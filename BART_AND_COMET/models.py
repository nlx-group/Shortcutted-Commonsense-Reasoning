from os import cpu_count

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.utils.data as Data
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1_score
import transformers
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from transformers.optimization import (
    AdamW,
    get_polynomial_decay_schedule_with_warmup,
)

from data import (
    ARCTSeqDataMixin,
    ARCTRankingDataMixin,
    ARCDataMixin,
    PIQADataMixin,
    CSQADataMixin,
)


class HFBartForMultipleChoice(transformers.BartForSequenceClassification):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = transformers.BartModel(config)
        self.dropout = nn.Dropout(config.classif_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        num_choices = input_ids.shape[1]

        # Guessing decoder_input_ids is None as we are mirroring
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        outputs = self.model(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = flat_input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]
        sentence_representation = self.dropout(sentence_representation)

        logits = self.classifier(sentence_representation)
        if getattr(self, "zero_missing_choices", True):
            missing_choices = (input_ids.sum(dim=2) == 0).flatten()
            logits[missing_choices] = 0.0

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BartForClassification(pl.LightningModule):
    def __init__(
        self,
        hparams,
        data_path=None,
        epochs=None,
        lr_schedule=True,
        num_classes=2,
        init_pretrained_bart=True,
    ):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = transformers.BartTokenizer.from_pretrained(
            hparams.model_name,
            cache_dir="/tmp/",
        )
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.data_path = data_path
        self.epochs = epochs
        # self.epoch = 0
        self.lr_schedule = lr_schedule
        self.num_classes = num_classes
        self.bart = None
        if init_pretrained_bart:
            self.init_pretrained_bart()

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
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        if self.lr_schedule:
            steps = len(self.train_dataloader()) * self.epochs
            # scheduler = get_linear_schedule_with_warmup(
            #     opt,
            #     int(steps * self.hparams.warmup_ratio),
            #     steps,
            # )
            scheduler = get_polynomial_decay_schedule_with_warmup(
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
        input_ids, attention_mask, y = batch
        loss, logits = self.forward(input_ids, attention_mask, y)
        accuracy_score = accuracy(logits, y, num_classes=self.num_classes)
        self.log("Accuracy/Train", accuracy_score)
        self.log("Loss/Train", loss)
        return {"loss": loss}

    def validation_step(self, batch, _):
        input_ids, attention_mask, y = batch
        loss, logits = self.forward(input_ids, attention_mask, y)
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
        input_ids, attention_mask, y = batch
        logits = self.forward(input_ids, attention_mask)[0]
        return {"y_pred": logits, "y": y}

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        y = torch.cat([x["y"] for x in outputs], dim=0).flatten()
        accuracy_score = accuracy(y_pred, y, num_classes=self.num_classes)
        f1 = f1_score(y_pred, y)
        log = {"Test/Accuracy": accuracy_score, "Test/f1": f1}
        self.log_dict(log, prog_bar=True)

    def forward(self, input_ids, attention_mask, y=None, argmax=True):
        x = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=y)
        logits = x["logits"]
        out = ()
        if y is not None:
            out += (x["loss"],)
        if argmax:
            logits = torch.argmax(logits, dim=1)
        out += (logits,)

        return out

    def prepare_data(self):
        raise NotImplementedError()

    def init_pretrained_bart(self):
        self.bart = transformers.BartForSequenceClassification.from_pretrained(
            self.hparams.pretrained_weights,
            cache_dir="/tmp/",
            return_dict=True,
            num_labels=self.num_classes,
        )


class BartForMultipleChoice(BartForClassification):
    def init_pretrained_bart(self):
        self.bart = HFBartForMultipleChoice.from_pretrained(
            self.hparams.pretrained_weights,
            cache_dir="/tmp/",
            return_dict=True,
            num_labels=self.num_classes,
        )


class BARTSeqForARCT(ARCTSeqDataMixin, BartForClassification):
    pass


class BARTRankingForARCT(ARCTRankingDataMixin, BartForMultipleChoice):
    pass


class BARTForARC(ARCDataMixin, BartForMultipleChoice):
    pass


class BARTForPIQA(PIQADataMixin, BartForMultipleChoice):
    pass


class BARTForCSQA(CSQADataMixin, BartForMultipleChoice):
    pass
