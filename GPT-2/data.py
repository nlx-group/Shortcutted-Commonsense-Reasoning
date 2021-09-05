import json
import logging
import random
import os

from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from tqdm import tqdm

LOG = logging.getLogger(__name__)


class ARCTDataMixin:
    def encode_example(self, row):
        input_ids = torch.zeros([2, self.hparams.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros([2, self.hparams.max_seq_len], dtype=torch.float32)
        for i in range(2):
            x = self.tokenizer(
                self.tokenizer.sep_token.join(
                    [
                        row["claim"],
                        row[f"warrant{i}"],
                        row["reason"],
                    ]
                ) + "[CLS]",
                max_length=self.hparams.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids[i] = x["input_ids"][0]
            attention_mask[i] = x["attention_mask"][0]

        mc_token_ids = torch.LongTensor(
            [t.index(self.tokenizer.cls_token_id) for t in input_ids.tolist()],
        )

        return input_ids, attention_mask, mc_token_ids, int(row["correctLabelW0orW1"])

    def encode_partition(self, df):
        input_ids = torch.zeros(
            [len(df), 2, self.hparams.max_seq_len], dtype=torch.int64
        )
        attention_mask = torch.zeros(
            [len(df), 2, self.hparams.max_seq_len], dtype=torch.float32
        )
        mc_token_ids = torch.zeros([len(df), 2], dtype=torch.int64)
        y = torch.zeros(len(df), dtype=torch.int64)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            _input_ids, _attention_mask, mc_token_id, label = self.encode_example(row)
            input_ids[i] = _input_ids
            attention_mask[i] = _attention_mask
            mc_token_ids[i] = mc_token_id
            y[i] = label

        return TensorDataset(input_ids, attention_mask, mc_token_ids, y)

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
                    self.tokenizer.sep_token.join(
                        [
                            row["claim"],
                            row["warrant0"],
                            row["warrant1"],
                            row["reason"],
                        ]
                    ) + "[CLS]",
                    return_tensors="pt",
                )
                size = x["input_ids"].size()[1]
                if size > _max:
                    _max = size
                    _max_id = i
            out[partition] = (_max, _max_id)
        return out


class ARCDataMixin:
    def encode_example(self, example):
        input_ids = torch.zeros([5, self.hparams.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros([5, self.hparams.max_seq_len], dtype=torch.float32)
        labels = example["choices"]["label"]

        for i in range(len(labels)):
            x = self.tokenizer(
                self.tokenizer.sep_token.join(
                    [example["question"], example["choices"]["text"][i]]
                ) + "[CLS]",
                max_length=self.hparams.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            pad_ids = x["input_ids"][0].eq(self.tokenizer.pad_token_id).nonzero(as_tuple=False)

            if pad_ids.sum() > 0:
               x["input_ids"][0, pad_ids[0]] = self.tokenizer.cls_token_id
            else:
               x["input_ids"][0, -1] = self.tokenizer.cls_token_id

            input_ids[i] = x["input_ids"][0]
            attention_mask[i] = x["attention_mask"][0]

        input_ids[input_ids.sum(dim=1) == 0, -1] = self.tokenizer.cls_token_id

        mc_token_ids = torch.LongTensor(
            [t.index(self.tokenizer.cls_token_id) for t in input_ids.tolist()],
        )

        return (
            input_ids,
            attention_mask,
            mc_token_ids,
            labels.index(example["answerKey"]),
        )

    def encode_partition(self, examples):
        input_ids = torch.zeros(
            [len(examples), self.num_classes, self.hparams.max_seq_len],
            dtype=torch.int64,
        )
        attention_mask = torch.zeros(
            [len(examples), self.num_classes, self.hparams.max_seq_len],
            dtype=torch.float32,
        )
        mc_token_ids = torch.zeros([len(examples), 5], dtype=torch.int64)
        y = torch.zeros(len(examples), dtype=torch.int64)

        for i, example in tqdm(enumerate(examples), total=len(examples)):
            _input_ids, _attention_mask, mc_token_id, label = self.encode_example(example)
            input_ids[i] = _input_ids
            attention_mask[i] = _attention_mask
            mc_token_ids[i] = mc_token_id
            y[i] = label

        return TensorDataset(input_ids, attention_mask, mc_token_ids, y)

    def prepare_data(self):
        LOG.info("Preparing data")
        ds = load_dataset("ai2_arc", "ARC-Challenge")
        if self.train_data is None:
            self.train_data = self.encode_partition(ds["train"])

        if self.val_data is None:
            self.val_data = self.encode_partition(ds["validation"])

        if self.test_data is None:
            self.test_data = self.encode_partition(ds["test"])

    def get_max_sentence_sizes(self):
        ds = load_dataset("ai2_arc", "ARC-Challenge")
        partitions = ["train", "validation", "test"]
        out = {}

        for partition in partitions:
            sizes = []
            _max_labels = 0
            for i, example in enumerate(ds[partition]):
                num_labels = len(example["choices"]["label"])
                for j in range(num_labels):
                    x = self.tokenizer(
                        self.tokenizer.sep_token.join(
                            [
                                example["question"],
                                example["choices"]["text"][j],
                            ]
                        ) + "[CLS]",
                        return_tensors="pt",
                    )
                    sizes.append(x["input_ids"].size()[1])
                    if num_labels > _max_labels:
                        _max_labels = num_labels

            std = np.std(sizes)
            mean = np.mean(sizes)

            out[partition] = (
                mean,
                std,
                mean + std,
                mean + 2 * std,
                mean + 3 * std,
                np.max(sizes),
                _max_labels,
            )

        return out



class PIQADataMixin:
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

    def encode_example(self, example):
        input_ids = torch.zeros([2, self.hparams.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros([2, self.hparams.max_seq_len], dtype=torch.float32)
        for i in range(2):
            x = self.tokenizer(
                self.tokenizer.sep_token.join(
                    [
                        example["goal"],
                        example[f"sol{i + 1}"],
                    ]
                ),
                max_length=self.hparams.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            pad_ids = x["input_ids"][0].eq(self.tokenizer.pad_token_id).nonzero(as_tuple=False)

            if pad_ids.sum() > 0:
                x["input_ids"][0, pad_ids[0]] = self.tokenizer.cls_token_id
            else:
                x["input_ids"][0, -1] = self.tokenizer.cls_token_id

            input_ids[i] = x["input_ids"][0]
            attention_mask[i] = x["attention_mask"][0]

        mc_token_ids = torch.LongTensor(
            [t.index(self.tokenizer.cls_token_id) for t in input_ids.tolist()],
        )

        return input_ids, attention_mask, mc_token_ids

    def encode_partition(self, X, y):
        input_ids = torch.zeros(
            [len(X), 2, self.hparams.max_seq_len], dtype=torch.int64
        )
        attention_mask = torch.zeros(
            [len(X), 2, self.hparams.max_seq_len], dtype=torch.float32
        )
        mc_token_ids = torch.zeros([len(X), 2], dtype=torch.int64)
        ys = torch.zeros(len(X), dtype=torch.int64)
        zip_inputs = list(zip(X, y))

        for i, (xx, yy) in tqdm(enumerate(zip_inputs), total=len(zip_inputs)):
            _input_ids, _attention_mask, mc_token_id = self.encode_example(xx)
            input_ids[i] = _input_ids
            attention_mask[i] = _attention_mask
            mc_token_ids[i] = mc_token_id
            ys[i] = yy

        return TensorDataset(input_ids, attention_mask, mc_token_ids, ys)

    def get_max_sentence_sizes(self):
        dfs = {
            "train": self.get_train_data(),
            "test": self.get_test_data(),
        }
        out = {}

        for partition in dfs:
            sizes = []
            for example, _ in zip(*dfs[partition]):
                for i in range(2):
                    x = self.tokenizer(
                        self.tokenizer.sep_token.join(
                            [
                                example["goal"],
                                example[f"sol{i + 1}"],
                            ]
                        ) + "[CLS]",
                        return_tensors="pt",
                    )
                    size = x["input_ids"].size()[1]

                    if size < 512:
                        sizes.append(size)

            std = np.std(sizes)
            mean = np.mean(sizes)

            out[partition] = (
                mean,
                std,
                mean + std,
                mean + 2 * std,
                mean + 3 * std,
                np.max(sizes),
            )

        return out


class CSQADataMixin:
    def encode_example(self, example):
        input_ids = torch.zeros([5, self.hparams.max_seq_len], dtype=torch.int64)
        attention_mask = torch.zeros([5, self.hparams.max_seq_len], dtype=torch.float32)
        labels = example["choices"]["label"]

        for i in range(len(labels)):
            x = self.tokenizer(
                self.tokenizer.sep_token.join(
                    [example["question"], example["choices"]["text"][i]]
                ) + "[CLS]",
                max_length=self.hparams.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids[i] = x["input_ids"][0]
            attention_mask[i] = x["attention_mask"][0]

        mc_token_ids = torch.LongTensor(
            [t.index(self.tokenizer.cls_token_id) for t in input_ids.tolist()],
        )

        return (
            input_ids,
            attention_mask,
            mc_token_ids,
            labels.index(example["answerKey"]),
        )

    def encode_partition(self, examples):
        input_ids = torch.zeros(
            [len(examples), self.num_classes, self.hparams.max_seq_len],
            dtype=torch.int64,
        )
        attention_mask = torch.zeros(
            [len(examples), self.num_classes, self.hparams.max_seq_len],
            dtype=torch.float32,
        )
        mc_token_ids = torch.zeros([len(examples), self.num_classes], dtype=torch.int64)
        y = torch.zeros(len(examples), dtype=torch.int64)

        for i, example in tqdm(enumerate(examples), total=len(examples)):
            _input_ids, _attention_mask, mc_token_id, label = self.encode_example(example)
            input_ids[i] = _input_ids
            attention_mask[i] = _attention_mask
            mc_token_ids[i] = mc_token_id
            y[i] = label

        return TensorDataset(input_ids, attention_mask, mc_token_ids, y)

    def prepare_data(self):
        LOG.info("Preparing data")
        ds = load_dataset("commonsense_qa")
        if self.train_data is None and self.val_data is None:
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
        partitions = ["train", "validation", "test"]
        out = {}

        for partition in partitions:
            sizes = []
            for i, example in enumerate(ds[partition]):
                num_labels = len(example["choices"]["label"])
                for j in range(num_labels):
                    x = self.tokenizer(
                        self.tokenizer.sep_token.join(
                            [
                                example["question"],
                                example["choices"]["text"][j],
                            ]
                        ) + "[CLS]",
                        return_tensors="pt",
                    )
                    sizes.append(x["input_ids"].size()[1])

            std = np.std(sizes)
            mean = np.mean(sizes)

            out[partition] = dict(
                mean=mean,
                std=std,
                one_std=mean + std,
                two_std=mean + 2 * std,
                three_std=mean + 3 * std,
                max=np.max(sizes),
            )

        return out
