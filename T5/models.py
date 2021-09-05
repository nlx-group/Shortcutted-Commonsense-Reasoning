from itertools import chain

import torch
import torch.utils.data as Data
import pytorch_lightning as pl
import transformers
from transformers.optimization import AdamW
from sklearn.metrics import accuracy_score, f1_score

from data import (
    ARCTDataMixin,
    ARCDataMixin,
    PIQADataMixin,
    CSQADataMixin
)


class T5ForGeneration(pl.LightningModule):
    def __init__(
        self,
        hparams,
        data_path=None,
        epochs=None,
        print_outputs=False,
    ):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            hparams.model_name,
            cache_dir="/tmp/",
        )
        self.t5 = transformers.T5ForConditionalGeneration.from_pretrained(
            hparams.model_name, return_dict=True
        )
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.data_path = data_path
        self.epochs = epochs
        self.print_outputs = print_outputs

    def train_dataloader(self):
        return Data.DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return Data.DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return Data.DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # HACK: Couldn't do num_workers > 0 with 16GB of RAM.
            num_workers=0,
        )

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, _):
        input_ids, attention_mask, y = batch
        loss, _ = self.forward(input_ids, attention_mask, y)
        self.log("Loss/Train", loss)
        return {"loss": loss}

    def validation_step(self, batch, _):
        input_ids, attention_mask, y = batch
        loss, _ = self.forward(input_ids, attention_mask, y)
        logits = self.generate(input_ids, attention_mask)
        gen_text = self.tokenizer.batch_decode(logits)
        y_text = self.tokenizer.batch_decode(y)

        return {
            "val_loss": loss,
            "val_tp": sum(map(lambda x: x[0] == x[1], zip(gen_text, y_text))),
            "batch_size": logits.shape[0],
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.FloatTensor([x["val_loss"] for x in outputs]).mean()
        val_accuracy = torch.FloatTensor([x["val_tp"] for x in outputs])
        total_examples = (
            torch.FloatTensor([x["batch_size"] for x in outputs]).sum().item()
        )
        val_accuracy = torch.FloatTensor([val_accuracy.sum().item() / total_examples])

        self.log_dict(
            {
                "Loss/Validation": avg_val_loss,
                "Accuracy/Validation": val_accuracy,
            },
            prog_bar=True,
        )

    def test_step(self, batch, _):
        input_ids, attention_mask, y = batch
        logits = self.generate(input_ids, attention_mask)
        gen_text = self.tokenizer.batch_decode(logits)
        y_text = self.tokenizer.batch_decode(y)
        return {"y_pred": gen_text, "y": y_text}

    def test_epoch_end(self, outputs):
        y_pred = list(chain(*[x["y_pred"] for x in outputs]))
        y = list(chain(*[x["y"] for x in outputs]))
        if self.print_outputs:
            print("Predictions:", y_pred)
            print("Reference:", y)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y_pred, y, average="macro")
        log = {"Test/Accuracy": accuracy, "Test/f1": f1}
        self.log_dict(log, prog_bar=True)

    def forward(self, input_ids, attention_mask, y=None):
        x = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=y)
        logits = x["logits"]
        out = ()
        if y is not None:
            out += (x["loss"],)
        out += (logits,)

        return out

    def generate(self, input_ids=None, attention_mask=None):
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **self.get_generation_config()
        )

    def prepare_data(self):
        raise NotImplementedError()

    def get_generation_config(self):
        raise NotImplementedError()


class T5ForARCT(ARCTDataMixin, T5ForGeneration):
    def get_generation_config(self):
        # This is the number of subwords needed to encode
        # warrant 1 and warrant 2
        return dict(
            max_length=self.OUTPUT_SIZE,
        )


class T5ForARC(ARCDataMixin, T5ForGeneration):
    def get_generation_config(self):
        # This is the number of subwords needed to encode
        # warrant 1 and warrant 2
        return dict(
            max_length=self.OUTPUT_SIZE,
        )


class T5ForPIQA(PIQADataMixin, T5ForGeneration):
    def get_generation_config(self):
        # This is the number of subwords needed to encode
        # warrant 1 and warrant 2
        return dict(
            max_length=self.OUTPUT_SIZE,
        )


class T5ForCSQA(CSQADataMixin, T5ForGeneration):
    def get_generation_config(self):
        # This is the number of subwords needed to encode
        # warrant 1 and warrant 2
        return dict(
            max_length=self.OUTPUT_SIZE,
        )
