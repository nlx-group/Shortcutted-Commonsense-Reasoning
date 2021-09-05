import json
import logging
import os

from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np


LOG = logging.getLogger(__name__)


class ARCTDataMixin:
    INPUT_TEMPLATE = "arct claim: {} reason: {} warrant0: {} warrant1: {}"
    # This is the number of subwords needed to encode
    # warrant 1 and warrant 2
    OUTPUT_SIZE = 4

    def encode_example(self, row):
        x = self.tokenizer(
            self.INPUT_TEMPLATE.format(
                row["claim"], row["reason"], row["warrant0"], row["warrant1"]
            ),
            max_length=self.hparams.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        y = self.tokenizer(
            str(bool(row["correctLabelW0orW1"])),
            max_length=self.OUTPUT_SIZE,
            padding="max_length",
            return_tensors="pt",
        )

        return (x["input_ids"][0], x["attention_mask"][0], y["input_ids"][0])

    def encode_partition(self, df):
        input_ids = torch.zeros([len(df), self.hparams.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros(
            [len(df), self.hparams.max_seq_len], dtype=torch.float32
        )
        y = torch.zeros([len(df), self.OUTPUT_SIZE], dtype=torch.int64)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            _input_ids, _attention_mask, label = self.encode_example(row)
            input_ids[i] = _input_ids
            attention_mask[i] = _attention_mask
            y[i] = label

        return TensorDataset(input_ids, attention_mask, y)

    def get_train_df(self):
        return pd.read_csv(os.path.join(self.data_path, "train.csv"), sep="\t")

    def get_val_df(self):
        return pd.read_csv(os.path.join(self.data_path, "dev.csv"), sep="\t")

    def get_test_df(self):
        return pd.read_csv(os.path.join(self.data_path, "test.csv"), sep="\t")

    def prepare_data(self):
        LOG.info("Preparing data")
        if self.train_data is None:
            train = self.get_train_df()
            self.train_data = self.encode_partition(train)

        if self.val_data is None:
            val = self.get_val_df()
            self.val_data = self.encode_partition(val)

        if self.test_data is None:
            test = self.get_test_df()
            self.test_data = self.encode_partition(test)

    def get_max_sentence_sizes(self):
        dfs = {
            "train": self.get_train_df(),
            "val": self.get_val_df(),
            "test": self.get_test_df(),
        }
        out = {}

        for partition in dfs:
            _max = 0
            _max_id = 0
            for i, row in dfs[partition].iterrows():
                x = self.tokenizer(
                    self.INPUT_TEMPLATE.format(
                        row["claim"], row["reason"], row["warrant0"], row["warrant1"]
                    ),
                    return_tensors="pt",
                )
                size = x["input_ids"].size()[1]
                if size > _max:
                    _max = size
                    _max_id = i
            out[partition] = (_max, _max_id)
        return out


class ARCDataMixin:
    OUTPUT_SIZE = 2

    def encode_example(self, example):
        inp = f"arc: question: {example['question']}"

        for i, label in enumerate(example["choices"]["label"]):
            inp += f" {label}: {example['choices']['text'][i]}"

        x = self.tokenizer(
            inp,
            max_length=self.hparams.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        y = self.tokenizer(
            example["answerKey"],
            max_length=self.OUTPUT_SIZE,
            padding="max_length",
            return_tensors="pt",
        )

        return (x["input_ids"][0], x["attention_mask"][0], y["input_ids"][0])

    def encode_partition(self, examples):
        input_ids = torch.zeros(
            [len(examples), self.hparams.max_seq_len], dtype=torch.int64
        )
        attention_mask = torch.zeros(
            [len(examples), self.hparams.max_seq_len], dtype=torch.float32
        )
        y = torch.zeros([len(examples), self.OUTPUT_SIZE], dtype=torch.int64)

        for i, row in tqdm(enumerate(examples), total=len(examples)):
            _input_ids, _attention_mask, label = self.encode_example(row)
            input_ids[i] = _input_ids
            attention_mask[i] = _attention_mask
            y[i] = label

        return TensorDataset(input_ids, attention_mask, y)

    def prepare_data(self):
        LOG.info("Preparing data")
        ds = None
        if self.train_data is None:
            ds = load_dataset("ai2_arc", "ARC-Challenge")
            self.train_data = self.encode_partition(ds["train"])

        if self.val_data is None:
            self.val_data = self.encode_partition(ds["validation"])

        if self.test_data is None:
            self.test_data = self.encode_partition(ds["test"])

    def get_max_sentence_sizes(self):
        ds = load_dataset("ai2_arc", "ARC-Challenge")
        out = {}

        for partition in ds:
            max_x = 0
            max_x_id = 0
            max_y = 0
            max_y_id = 0

            for i, example in enumerate(ds[partition]):
                inp = f"arc: question: {example['question']}"

                for j, label in enumerate(example["choices"]["label"]):
                    inp += f" {label}: {example['choices']['text'][j]}"

                x = self.tokenizer(
                    inp,
                    return_tensors="pt",
                )
                y = self.tokenizer(
                    example["answerKey"],
                    return_tensors="pt",
                )
                size_x = x["input_ids"].size(1)
                size_y = y["input_ids"].size(1)
                if size_x > max_x:
                    max_x = size_x
                    max_x_id = i
                if size_y > max_y:
                    max_y = size_y
                    max_y_id = i

            out[partition] = dict(
                max_x=max_x, max_x_id=max_x_id, max_y=max_y, max_y_id=max_y_id
            )

        return out


class PIQADataMixin:
    INPUT_TEMPLATE = "piqa goal: {} sol0: {} sol1: {}"
    OUTPUT_SIZE = 4

    def encode_example(self, example, yy):
        x = self.tokenizer(
            self.INPUT_TEMPLATE.format(
                example["goal"],
                example["sol1"],
                example["sol2"],
            ),
            max_length=self.hparams.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        y = self.tokenizer(
            str(bool(yy)),
            max_length=self.OUTPUT_SIZE,
            padding="max_length",
            return_tensors="pt",
        )

        return (x["input_ids"][0], x["attention_mask"][0], y["input_ids"][0])

    def encode_partition(self, X, y):
        input_ids = torch.zeros([len(X), self.hparams.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros(
            [len(X), self.hparams.max_seq_len], dtype=torch.float32
        )
        y_input_ids = torch.zeros([len(X), self.OUTPUT_SIZE], dtype=torch.int64)

        zip_inputs = list(zip(X, y))

        for i, (xx, yy) in tqdm(enumerate(zip_inputs), total=len(zip_inputs)):
            _input_ids, _attention_mask, label = self.encode_example(xx, yy)
            input_ids[i] = _input_ids
            attention_mask[i] = _attention_mask
            y_input_ids[i] = label

        return TensorDataset(input_ids, attention_mask, y_input_ids)

    def get_train_data(self):
        X = None
        y = None

        with open(os.path.join(self.data_path, "train.jsonl")) as fr:
            X = list(map(lambda line: json.loads(line), fr))

        with open(os.path.join(self.data_path, "train-labels.lst")) as fr:
            y = list(map(lambda x: int(x.rstrip()), fr.readlines()))

        return X, y

    def get_test_data(self):
        # Test data in our case is actually the devset as we don't have access
        # to the labels of the testset

        X = None
        y = None

        with open(os.path.join(self.data_path, "valid.jsonl")) as fr:
            X = list(map(lambda line: json.loads(line), fr))

        with open(os.path.join(self.data_path, "valid-labels.lst")) as fr:
            y = list(map(lambda x: int(x.rstrip()), fr.readlines()))

        return X, y

    def prepare_data(self):
        LOG.info("Preparing data")
        if self.train_data is None:
            X, y = self.get_train_data()
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.1,
                random_state=self.hparams.seed,
                stratify=y,
            )
            self.train_data = self.encode_partition(X_train, y_train)
            self.val_data = self.encode_partition(X_val, y_val)

        if self.test_data is None:
            X, y = self.get_test_data()
            self.test_data = self.encode_partition(X, y)

    def get_max_sentence_sizes(self):
        dfs = {
            "train": self.get_train_data(),
            "test": self.get_test_data(),
        }
        out = {}

        for partition in dfs:
            sizes = []
            for example, _ in zip(*dfs[partition]):
                x = self.tokenizer(
                    self.INPUT_TEMPLATE.format(
                        example["goal"],
                        example["sol1"],
                        example["sol2"],
                    ),
                    return_tensors="pt",
                )
                size = x["input_ids"].size(1)

                if size < 512:
                    sizes.append(size)

            std = np.std(sizes)
            mean = np.mean(sizes)

            out[partition] = dict(
                mean=mean,
                std=std,
                one_std=mean + std,
                two_std=mean + 2 * std,
                three_std=mean + 3 * std,
            )

        return out


class CSQADataMixin:
    OUTPUT_SIZE = 2

    def encode_example(self, example):
        inp = f"csqa: question: {example['question']}"

        for i, label in enumerate(example["choices"]["label"]):
            inp += f" {label}: {example['choices']['text'][i]}"

        x = self.tokenizer(
            inp,
            max_length=self.hparams.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        y = self.tokenizer(
            example["answerKey"],
            max_length=self.OUTPUT_SIZE,
            padding="max_length",
            return_tensors="pt",
        )

        return (x["input_ids"][0], x["attention_mask"][0], y["input_ids"][0])

    def encode_partition(self, examples):
        input_ids = torch.zeros(
            [len(examples), self.hparams.max_seq_len], dtype=torch.int64
        )
        attention_mask = torch.zeros(
            [len(examples), self.hparams.max_seq_len], dtype=torch.float32
        )
        y = torch.zeros([len(examples), self.OUTPUT_SIZE], dtype=torch.int64)

        for i, row in tqdm(enumerate(examples), total=len(examples)):
            _input_ids, _attention_mask, label = self.encode_example(row)
            input_ids[i] = _input_ids
            attention_mask[i] = _attention_mask
            y[i] = label

        return TensorDataset(input_ids, attention_mask, y)

    def prepare_data(self):
        LOG.info("Preparing data")
        ds = None
        if self.train_data is None and self.val_data is None:
            ds = load_dataset("commonsense_qa")
            y = list(
                map(lambda x: x["choices"]["label"].index(x["answerKey"]), ds["train"])
            )
            X_train, X_val, _, _ = train_test_split(
                list(ds["train"]),
                y,
                test_size=0.1,
                random_state=self.hparams.seed,
                stratify=y,
            )
            self.train_data = self.encode_partition(X_train)
            self.val_data = self.encode_partition(X_val)

        if self.test_data is None:
            self.test_data = self.encode_partition(ds["validation"])

    def get_max_sentence_sizes(self):
        ds = load_dataset("commonsense_qa")
        out = {}

        for partition in ds:
            max_x = 0
            max_x_id = 0
            max_y = 0
            max_y_id = 0

            for i, example in enumerate(ds[partition]):
                inp = f"csqa: question: {example['question']}"

                for j, label in enumerate(example["choices"]["label"]):
                    inp += f" {label}: {example['choices']['text'][j]}"

                x = self.tokenizer(
                    inp,
                    return_tensors="pt",
                )
                y = self.tokenizer(
                    example["answerKey"],
                    return_tensors="pt",
                )
                size_x = x["input_ids"].size(1)
                size_y = y["input_ids"].size(1)
                if size_x > max_x:
                    max_x = size_x
                    max_x_id = i
                if size_y > max_y:
                    max_y = size_y
                    max_y_id = i

            out[partition] = dict(
                max_x=max_x, max_x_id=max_x_id, max_y=max_y, max_y_id=max_y_id
            )

        return out
