from argparse import Namespace
import json
import os

import click
from datasets import load_dataset
from textattack.attack_recipes import (
    BERTAttackLi2020,
    BAEGarg2019,
    ModifiedBERTAttackLi2020,
    ModifiedTextFoolerJin2019,
    CLARE2020,
)
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper
from textattack.goal_function_results import GoalFunctionResultStatus
from more_itertools import chunked
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from models import BARTRankingForARCT, BARTForARC, BARTForPIQA, BARTForCSQA


##################################
#                                #
#           Datasets             #
#                                #
##################################


class HfDatasetToDfMixin:
    def attack_result_to_df(self, result):
        adversarial = (
            result.perturbed_result.goal_status == GoalFunctionResultStatus.SUCCEEDED
        )
        original_inp = result.original_result.attacked_text._text_input
        perturbed_inp = result.perturbed_result.attacked_text._text_input
        data = {f"original_{k}": [original_inp[k]] for k in self.input_columns}
        data.update(
            {
                f"adv_{k}": [perturbed_inp[k]] if adversarial else "N/A"
                for k in self.input_columns
            }
        )
        data["label"] = [result.perturbed_result.ground_truth_output]
        data["adversarial"] = adversarial
        return pd.DataFrame(data=data)


class PIQADataset(HfDatasetToDfMixin, HuggingFaceDataset):
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._name = "piqa"
        self.input_columns = ["goal", "sol1", "sol2"]
        self.output_column = "label"
        self.examples = self._load_examples()

        self._i = 0
        self.label_map = None
        self.output_scale_factor = None

    def _load_examples(self):
        examples = []

        with open(os.path.join(self._root_dir, "valid.jsonl")) as fr:
            for line in fr:
                examples.append(json.loads(line))

        with open(os.path.join(self._root_dir, "valid-labels.lst")) as fr:
            for i, line in enumerate(fr):
                examples[i].update({"label": int(line.rstrip())})

        return examples


class ARCTDataset(HfDatasetToDfMixin, HuggingFaceDataset):
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._name = "arct"
        self.input_columns = ["reason", "claim", "warrant0", "warrant1"]
        self.output_column = "label"
        self.examples = self._load_examples()

        self._i = 0
        self.label_map = None
        self.output_scale_factor = None

    def _load_examples(self):
        examples = []

        df = pd.read_csv(os.path.join(self._root_dir, "test.csv"), sep="\t")
        for _, row in df.iterrows():
            ex = {k: row[k] for k in self.input_columns}
            ex["label"] = row["correctLabelW0orW1"]
            examples.append(ex)

        return examples


class ARCDataset(HfDatasetToDfMixin, HuggingFaceDataset):
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._name = "arc"
        self.input_columns = [
            "question",
            "choiceA",
            "choiceB",
            "choiceC",
            "choiceD",
            "choiceE",
        ]
        self.output_column = "label"
        self.answer_labels = ["A", "B", "C", "D", "E"]
        self.answer_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        self.examples = self._load_examples()

        self._i = 0
        self.label_map = None
        self.output_scale_factor = None

    def transform_labels(self, ex):
        for i, label in enumerate(ex["choices"]["label"]):
            if label not in self.answer_labels:
                ex["choices"]["label"][i] = self.answer_map[label]
        if ex["answerKey"] not in self.answer_labels:
            ex["answerKey"] = self.answer_map[ex["answerKey"]]
        return ex

    def _load_examples(self):
        examples = []

        ds = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
        for example in ds:
            example = self.transform_labels(example)
            ex = {"question": example["question"]}
            for i, label in enumerate(self.answer_labels):
                ex[f"choice{label}"] = (
                    example["choices"]["text"][i]
                    if i < len(example["choices"]["label"])
                    else ""
                )
            ex["label"] = self.answer_labels.index(example["answerKey"])
            examples.append(ex)

        return examples


class CSQADataset(HfDatasetToDfMixin, HuggingFaceDataset):
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._name = "csqa"
        self.input_columns = [
            "question",
            "choiceA",
            "choiceB",
            "choiceC",
            "choiceD",
            "choiceE",
        ]
        self.output_column = "label"
        self.answer_labels = ["A", "B", "C", "D", "E"]
        self.answer_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        self.examples = self._load_examples()

        self._i = 0
        self.label_map = None
        self.output_scale_factor = None

    def transform_labels(self, ex):
        for i, label in enumerate(ex["choices"]["label"]):
            if label not in self.answer_labels:
                ex["choices"]["label"][i] = self.answer_map[label]
        if ex["answerKey"] not in self.answer_labels:
            ex["answerKey"] = self.answer_map[ex["answerKey"]]
        return ex

    def _load_examples(self):
        examples = []

        ds = load_dataset("commonsense_qa", split="validation")
        for example in ds:
            example = self.transform_labels(example)
            ex = {"question": example["question"]}
            for i, label in enumerate(self.answer_labels):
                ex[f"choice{label}"] = (
                    example["choices"]["text"][i]
                    if i < len(example["choices"]["label"])
                    else ""
                )
            ex["label"] = self.answer_labels.index(example["answerKey"])
            examples.append(ex)

        return examples


##################################
#                                #
#           Models               #
#                                #
##################################


class BaseModelWrapper(ModelWrapper):
    def __init__(
        self,
        model_name,
        batch_size,
        num_choices,
        max_seq_len,
        ckpt_path,
        pretrained_weights=None,
    ):
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.pretrained_weights = pretrained_weights
        self.batch_size = batch_size
        self.num_choices = num_choices
        self.max_seq_len = max_seq_len
        self.model = self.get_model()
        self.tokenizer = self.model.tokenizer

    def get_grad(self):
        pass

    def get_model(self):
        raise NotImplementedError()

    def tokenize_input(self, _input):
        raise NotImplementedError()

    def tokenize_inputs(self, inputs):
        input_ids = torch.zeros(
            [len(inputs), self.num_choices, self.max_seq_len], dtype=torch.int64
        )
        attention_mask = torch.zeros(
            [len(inputs), self.num_choices, self.max_seq_len], dtype=torch.float32
        )

        for i, _input in enumerate(inputs):
            _input_ids, _attention_mask = self.tokenize_input(_input)
            input_ids[i] = _input_ids
            attention_mask[i] = _attention_mask

        return {
            "input_ids": input_ids.to("cuda"),
            "attention_mask": attention_mask.to("cuda"),
        }

    def __call__(self, text_inputs):
        chunks = chunked(text_inputs, self.batch_size)
        logits = torch.zeros([len(text_inputs), self.num_choices], dtype=torch.float32)

        for i, chunk in enumerate(chunks):
            tok_inputs = self.tokenize_inputs(chunk)
            with torch.no_grad():
                chunk_logits = self.model(**tok_inputs, argmax=False)[0].cpu()
            logits[i * self.batch_size : (i + 1) * self.batch_size] = chunk_logits

        return logits


class PIQAModelWrapper(BaseModelWrapper):
    def get_model(self):
        return BARTForPIQA.load_from_checkpoint(
            self.ckpt_path,
            hparams=Namespace(
                model_name=self.model_name,
                pretrained_weights=self.pretrained_weights,
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
            ),
            num_classes=self.num_choices,
        ).to("cuda")

    def tokenize_input(self, _input):
        input_ids = torch.zeros([self.num_choices, self.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros(
            [self.num_choices, self.max_seq_len], dtype=torch.float32
        )

        goal, sol1, sol2 = _input
        for i in range(2):
            x = self.tokenizer(
                self.tokenizer.sep_token.join(
                    [
                        goal,
                        sol1 if not i else sol2,
                    ]
                ),
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids[i] = x["input_ids"][0]
            attention_mask[i] = x["attention_mask"][0]

        eos_tokens = input_ids.eq(self.tokenizer.eos_token_id).sum(dim=1)
        truncated_choices = ~input_ids[:, -2].eq(
            self.tokenizer.pad_token_id
        ) & ~eos_tokens.eq(2)
        if truncated_choices.sum() > 0:
            input_ids[truncated_choices, -2] = self.tokenizer.eos_token_id
            attention_mask[truncated_choices, -2] = 1.0

        return input_ids, attention_mask


class ARCTModelWrapper(BaseModelWrapper):
    def get_model(self):
        return BARTRankingForARCT.load_from_checkpoint(
            self.ckpt_path,
            hparams=Namespace(
                model_name=self.model_name,
                pretrained_weights=self.pretrained_weights,
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
            ),
            num_classes=self.num_choices,
        ).to("cuda")

    def tokenize_input(self, _input):
        input_ids = torch.zeros([self.num_choices, self.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros(
            [self.num_choices, self.max_seq_len], dtype=torch.float32
        )
        reason, claim, warrant0, warrant1 = _input
        for i in range(2):
            x = self.tokenizer(
                self.tokenizer.sep_token.join(
                    [claim, warrant0 if not i else warrant1, reason]
                ),
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids[i] = x["input_ids"][0]
            attention_mask[i] = x["attention_mask"][0]

        return input_ids, attention_mask


class ARCModelWrapper(BaseModelWrapper):
    def get_model(self):
        return BARTForARC.load_from_checkpoint(
            self.ckpt_path,
            hparams=Namespace(
                model_name=self.model_name,
                pretrained_weights=self.pretrained_weights,
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
            ),
            num_classes=self.num_choices,
        ).to("cuda")

    def tokenize_input(self, _input):
        input_ids = torch.zeros([self.num_choices, self.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros(
            [self.num_choices, self.max_seq_len], dtype=torch.float32
        )

        (question, choiceA, choiceB, choiceC, choiceD, choiceE) = _input

        inp_dict = {
            "question": question,
            "choiceA": choiceA,
            "choiceB": choiceB,
            "choiceC": choiceC,
            "choiceD": choiceD,
            "choiceE": choiceE,
        }
        labels = ["A", "B", "C", "D", "E"]

        for i in range(self.num_choices):
            a = inp_dict[f"choice{labels[i]}"]
            if a:
                x = self.tokenizer(
                    self.tokenizer.sep_token.join([inp_dict["question"], a]),
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids[i] = x["input_ids"][0]
                attention_mask[i] = x["attention_mask"][0]

        # Make sure the same amount of EOS is present
        empty_choices = input_ids.sum(dim=1) == 0
        if empty_choices.sum() > 0:
            input_ids[empty_choices, -1] = self.tokenizer.eos_token_id
            attention_mask[empty_choices, -1] = 1.0

        eos_tokens = input_ids.eq(self.tokenizer.eos_token_id).sum(dim=1)
        truncated_choices = ~input_ids[:, -2].eq(
            self.tokenizer.pad_token_id
        ) & ~eos_tokens.eq(2)
        if truncated_choices.sum() > 0:
            input_ids[truncated_choices, -2] = self.tokenizer.eos_token_id
            attention_mask[truncated_choices, -2] = 1.0

        return input_ids, attention_mask


class CSQAModelWrapper(BaseModelWrapper):
    def get_model(self):
        return BARTForCSQA.load_from_checkpoint(
            self.ckpt_path,
            hparams=Namespace(
                model_name=self.model_name,
                pretrained_weights=self.pretrained_weights,
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
            ),
            num_classes=self.num_choices,
        ).to("cuda")

    def tokenize_input(self, _input):
        input_ids = torch.zeros([self.num_choices, self.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros(
            [self.num_choices, self.max_seq_len], dtype=torch.float32
        )

        (question, choiceA, choiceB, choiceC, choiceD, choiceE) = _input

        inp_dict = {
            "question": question,
            "choiceA": choiceA,
            "choiceB": choiceB,
            "choiceC": choiceC,
            "choiceD": choiceD,
            "choiceE": choiceE,
        }
        labels = ["A", "B", "C", "D", "E"]

        for i in range(self.num_choices):
            a = inp_dict[f"choice{labels[i]}"]
            if a:
                x = self.tokenizer(
                    self.tokenizer.sep_token.join([inp_dict["question"], a]),
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids[i] = x["input_ids"][0]
                attention_mask[i] = x["attention_mask"][0]

        empty_choices = input_ids.sum(dim=1) == 0
        if empty_choices.sum() > 0:
            input_ids[empty_choices, -1] = self.tokenizer.eos_token_id
            attention_mask[empty_choices, -1] = 1.0

        eos_tokens = input_ids.eq(self.tokenizer.eos_token_id).sum(dim=1)
        truncated_choices = ~input_ids[:, -2].eq(
            self.tokenizer.pad_token_id
        ) & ~eos_tokens.eq(2)
        if truncated_choices.sum() > 0:
            input_ids[truncated_choices, -2] = self.tokenizer.eos_token_id
            attention_mask[truncated_choices, -2] = 1.0

        return input_ids, attention_mask


##################################
#                                #
#           Attack               #
#                                #
##################################


def attack_dataset(recipe, ds, save_path, log_path):
    dfs = []
    y = []
    y_pred = []
    with open(log_path, "wt") as fw:
        for idx, result in enumerate(recipe.attack_dataset(ds)):
            header = " ".join(["-" * 20, f"Result {idx+1}", "-" * 20])
            print(header)
            print(result.__str__(color_method="ansi"))
            print()
            print(header, file=fw)
            print(str(result), file=fw)
            print("", file=fw)
            dfs.append(ds.attack_result_to_df(result))
            y.append(result.perturbed_result.ground_truth_output)
            y_pred.append(result.perturbed_result.output)

    pd.concat(dfs).to_csv(save_path, sep="\t", index=False)
    print(f"Accuracy score after attack: {accuracy_score(y, y_pred)}")
    print(f"F1 Score score after attack: {f1_score(y, y_pred, average='macro')}")


def get_dataset(name, root_dir):
    datasets = {
        "arct": ARCTDataset,
        "arc": ARCDataset,
        "piqa": PIQADataset,
        "csqa": CSQADataset,
    }
    return datasets[name](root_dir)


def get_recipe(name, model):
    recipes = {
        "bae": BAEGarg2019,
        "bert-attack": ModifiedBERTAttackLi2020,
        "textfooler": ModifiedTextFoolerJin2019,
        "clare": CLARE2020,
    }
    return recipes[name].build(model)


def get_model(
    name,
    model_name,
    batch_size,
    num_choices,
    max_seq_len,
    ckpt_path,
    pretrained_weights,
):
    models = {
        "arct": ARCTModelWrapper,
        "arc": ARCModelWrapper,
        "piqa": PIQAModelWrapper,
        "csqa": CSQAModelWrapper,
    }
    return models[name](
        model_name, batch_size, num_choices, max_seq_len, ckpt_path, pretrained_weights
    )


@click.command()
@click.option("--task", type=click.Choice(["arct", "arc", "piqa", "csqa"]))
@click.option(
    "--method", type=click.Choice(["bae", "bert-attack", "textfooler", "clare"])
)
@click.option("--model_name", type=str)
@click.option("--batch_size", type=int)
@click.option("--num_choices", type=int)
@click.option("--max_seq_len", type=int)
@click.option("--data_dir", type=click.Path())
@click.option("--checkpoint_path", type=click.Path())
@click.option("--pretrained_weights", type=click.Path())
@click.option("--save_dataset_path", type=click.Path())
@click.option("--log_path", type=click.Path())
def attack(
    task,
    method,
    model_name,
    batch_size,
    num_choices,
    max_seq_len,
    data_dir,
    checkpoint_path,
    pretrained_weights,
    save_dataset_path,
    log_path,
):
    ds = get_dataset(task, data_dir)
    model = get_model(
        task,
        model_name,
        batch_size,
        num_choices,
        max_seq_len,
        checkpoint_path,
        pretrained_weights,
    )
    recipe = get_recipe(method, model)
    attack_dataset(recipe, ds, save_dataset_path, log_path)


if __name__ == "__main__":
    attack()
