from os import cpu_count

import torch
import torch.utils.data as Data
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1_score
import transformers
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import adapted_hf as hf_adaptations
from data import (
    BinaryARCTDataMixin,
    ARCTDataMixin,
    PIQADataMixin,
    MultipleChoicePIQADataMixin,
    ARCDataMixin,
    RankingARCDataMixin,
    CSQADataMixin,
)


class RobertaForClassification(pl.LightningModule):
    def __init__(
        self,
        hparams,
        data_path=None,
        epochs=None,
        lr_schedule=True,
        num_classes=2,
        init_pretrained_roberta=True,
    ):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(
            hparams.model_name,
            cache_dir="/tmp/",
        )
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.data_path = data_path
        self.epochs = epochs
        self.lr_schedule = lr_schedule
        self.num_classes = num_classes
        self.roberta = None
        if init_pretrained_roberta:
            self.init_pretrained_roberta()

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
            scheduler = get_linear_schedule_with_warmup(
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
        x = self.roberta(input_ids=input_ids, attention_mask=attention_mask, labels=y)
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

    def init_pretrained_roberta(self):
        raise NotImplementedError


class RobertaForSequenceClassification(RobertaForClassification):
    def init_pretrained_roberta(self):
        self.roberta = transformers.RobertaForSequenceClassification.from_pretrained(
            self.hparams.model_name,
            cache_dir="/tmp/",
            return_dict=True,
            num_labels=self.num_classes,
        )


class RobertaForVariableMultipleChoice(RobertaForClassification):
    """
    Useful for tasks such as ARC where each example can have a different
    number of choices.
    """

    def init_pretrained_roberta(self):
        self.roberta = hf_adaptations.RobertaForVariableMultipleChoice.from_pretrained(
            self.hparams.model_name, cache_dir="/tmp/", return_dict=True
        )


class RobertaForMultipleChoice(RobertaForClassification):
    def init_pretrained_roberta(self):
        self.roberta = transformers.RobertaForMultipleChoice.from_pretrained(
            self.hparams.model_name, cache_dir="/tmp/", return_dict=True
        )


class RobertaForBinaryARCT(BinaryARCTDataMixin, RobertaForSequenceClassification):
    pass


class RobertaForARCT(ARCTDataMixin, RobertaForMultipleChoice):
    pass


class RobertaForPIQA(PIQADataMixin, RobertaForSequenceClassification):
    pass


class RobertaForMultipleChoicePIQA(
    MultipleChoicePIQADataMixin, RobertaForMultipleChoice
):
    pass


class RobertaForARC(ARCDataMixin, RobertaForSequenceClassification):
    pass


class RobertaForRankingARC(RankingARCDataMixin, RobertaForVariableMultipleChoice):
    pass


class RobertaForCSQA(CSQADataMixin, RobertaForMultipleChoice):
    pass
