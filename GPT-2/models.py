from os import cpu_count

import torch
from torch import nn
import torch.utils.data as Data
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1_score
import transformers
from transformers.modeling_utils import SequenceSummary
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from data import (
    ARCTDataMixin,
    ARCDataMixin,
    PIQADataMixin,
    CSQADataMixin,
)


class GPT2ForMultipleChoice(transformers.GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = transformers.GPT2Model(config)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) -
            1[``.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size - 1]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size - 1]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)

        Return:

        Example::

            >>> import torch
            >>> from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

            >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            >>> model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

            >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoded_choices = [tokenizer.encode(s) for s in choices]
            >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

            >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
            >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.logits
            >>> mc_logits = outputs.mc_logits

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        mc_loss = None
        if mc_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        # lm_loss = None
        # if labels is not None:
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     loss_fct = CrossEntropyLoss()
        #     lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (mc_logits,) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            #return ((lm_loss,) + output) if lm_loss is not None else output
            return output

        return transformers.modeling_gpt2.GPT2DoubleHeadsModelOutput(
            # loss=lm_loss,
            mc_loss=mc_loss,
            # logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class GPT2ForVariableMultipleChoice(GPT2ForMultipleChoice):
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) -
            1[``.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size - 1]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size - 1]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)

        Return:

        Example::

            >>> import torch
            >>> from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

            >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            >>> model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

            >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoded_choices = [tokenizer.encode(s) for s in choices]
            >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

            >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
            >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.logits
            >>> mc_logits = outputs.mc_logits

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        missing_choices = (input_ids.sum(dim=2) == 0)

        # lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
        mc_logits[missing_choices] = 0.0

        mc_loss = None
        if mc_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        # lm_loss = None
        # if labels is not None:
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     loss_fct = CrossEntropyLoss()
        #     lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (mc_logits,) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            #return ((lm_loss,) + output) if lm_loss is not None else output
            return output

        return transformers.modeling_gpt2.GPT2DoubleHeadsModelOutput(
            # loss=lm_loss,
            mc_loss=mc_loss,
            # logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )



class GPT2ForClassification(pl.LightningModule):
    def __init__(
        self,
        hparams,
        data_path=None,
        epochs=None,
        lr_schedule=True,
        num_classes=2,
        init_pretrained_gpt2=True,
    ):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            hparams.model_name,
            cache_dir="/tmp/",
        )
        self.tokenizer.add_special_tokens({
            "sep_token": "[SEP]",
            "cls_token": "[CLS]",
        })
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.data_path = data_path
        self.epochs = epochs
        # self.epoch = 0
        self.lr_schedule = lr_schedule
        self.num_classes = num_classes
        self.gpt2 = None
        if init_pretrained_gpt2:
            self.init_pretrained_gpt2()

    def train_dataloader(self):
        return Data.DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
        )

    def val_dataloader(self):
        return Data.DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
        )

    def test_dataloader(self):
        return Data.DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
        )

    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if self.lr_schedule:
            steps = len(self.train_dataloader()) * self.epochs
            scheduler = get_cosine_schedule_with_warmup(
                opt,
                int(steps * self.hparams.warmup_ratio),
                steps,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": scheduler,
            }
        return opt

    def training_step(self, batch, _):
        input_ids, attention_mask, mc_token_ids, y = batch
        loss, logits = self.forward(input_ids, attention_mask, mc_token_ids, y)
        accuracy_score = accuracy(logits, y, num_classes=self.num_classes)
        self.log("Accuracy/Train", accuracy_score)
        self.log("Loss/Train", loss)
        return {"loss": loss}

    def validation_step(self, batch, _):
        input_ids, attention_mask, mc_token_ids, y = batch
        loss, logits = self.forward(input_ids, attention_mask, mc_token_ids, y)
        return {
            "val_loss": loss,
            "val_tp": torch.sum(torch.eq(y, logits.detach()).view(-1)).item(),
            "batch_size": logits.shape[0],
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.FloatTensor([x["val_loss"] for x in outputs]).mean()
        val_accuracy = torch.FloatTensor([x["val_tp"] for x in outputs])
        total_examples = (
            torch.FloatTensor([x["batch_size"] for x in outputs]).sum().item()
        )
        val_accuracy = torch.FloatTensor([val_accuracy.sum().item() / total_examples])
        # print(f"Val Accuracy @ {self.epoch} epochs: {val_accuracy}")
        # self.epoch += 1

        self.log_dict(
            {
                "Loss/Validation": avg_val_loss,
                "Accuracy/Validation": val_accuracy,
            },
            prog_bar=True,
        )

    def test_step(self, batch, _):
        input_ids, attention_mask, mc_token_ids, y = batch
        logits = self.forward(input_ids, attention_mask, mc_token_ids)[0]
        return {"y_pred": logits, "y": y}

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        y = torch.cat([x["y"] for x in outputs], dim=0).flatten()
        accuracy_score = accuracy(y_pred, y, num_classes=self.num_classes)
        f1 = f1_score(y_pred, y)
        log = {"Test/Accuracy": accuracy_score, "Test/f1": f1}
        self.log_dict(log, prog_bar=True)

    def forward(self, input_ids, attention_mask, mc_token_ids, y=None, argmax=True):
        x = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mc_labels=y,
            mc_token_ids=mc_token_ids
        )
        logits = x["mc_logits"]
        out = ()
        if y is not None:
            out += (x["mc_loss"],)
        if argmax:
            logits = torch.argmax(logits, dim=1)
        out += (logits,)

        return out

    def prepare_data(self):
        raise NotImplementedError()

    def init_pretrained_gpt2(self):
        self.gpt2 = GPT2ForMultipleChoice.from_pretrained(
            self.hparams.model_name, cache_dir="/tmp/", return_dict=True
        )
        self.gpt2.resize_token_embeddings(len(self.tokenizer))


class GPT2ForVariableClassification(GPT2ForClassification):
    def init_pretrained_gpt2(self):
        self.gpt2 = GPT2ForVariableMultipleChoice.from_pretrained(
            self.hparams.model_name, cache_dir="/tmp/", return_dict=True
        )
        self.gpt2.resize_token_embeddings(len(self.tokenizer))


class GPT2ForARCT(ARCTDataMixin, GPT2ForClassification):
    pass


class GPT2ForARC(ARCDataMixin, GPT2ForVariableClassification):
    pass


class GPT2ForPIQA(PIQADataMixin, GPT2ForClassification):
    pass


class GPT2ForCSQA(CSQADataMixin, GPT2ForClassification):
    pass
