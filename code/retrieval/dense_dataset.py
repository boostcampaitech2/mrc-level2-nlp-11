import torch
import os
import random
import json
import numpy as np
import pandas as pd

from datasets import load_from_disk
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from func import retrieve_from_embedding


def TrainRetrievalDataset(tokenizer_name: str, dataset_name: str) -> TensorDataset:
    print("get in-batch training sample")
    org_dataset = load_from_disk(dataset_name)
    train_data = org_dataset["train"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    question = train_data["question"]
    context = train_data["context"]

    p_seqs = tokenizer(
        context, padding="max_length", truncation=True, return_tensors="pt"
    )
    q_seqs = tokenizer(
        question, padding="max_length", truncation=True, return_tensors="pt"
    )

    return TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        q_seqs["token_type_ids"],
    )


class TrainRetrievalRandomDataset(torch.utils.data.Dataset):
    def __init__(
        self, tokenizer_name: str, dataset_name: str, num_neg: int, context_path: str
    ):
        org_dataset = load_from_disk(dataset_name)
        self.train_data = org_dataset["train"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_neg = num_neg

        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))

        print("negative sampling from wiki")
        self.in_batch_negative()

    def in_batch_negative(self):
        train_data = self.train_data
        num_neg = self.num_neg
        corpus = np.array(self.corpus)
        p_with_neg = []

        for c in train_data["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)

                    break
        self.p_with_neg = p_with_neg

    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        train_data = self.train_data
        question = train_data["question"][idx]
        num_neg = self.num_neg
        p_with_neg = self.p_with_neg[
            idx * (num_neg + 1) : idx * (num_neg + 1) + (num_neg + 1)
        ]

        p_seqs = tokenizer(
            p_with_neg, padding="max_length", truncation=True, return_tensors="pt"
        )
        q_seqs = tokenizer(
            question, padding="max_length", truncation=True, return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, num_neg + 1, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, num_neg + 1, max_len
        )

        p_input_ids = p_seqs["input_ids"]
        p_attention_mask = p_seqs["attention_mask"]
        p_token_type_ids = p_seqs["token_type_ids"]
        q_input_ids = q_seqs["input_ids"]
        q_attention_mask = q_seqs["attention_mask"]
        q_token_type_ids = q_seqs["token_type_ids"]

        return (
            p_input_ids,
            p_attention_mask,
            p_token_type_ids,
            q_input_ids,
            q_attention_mask,
            q_token_type_ids,
        )

    def __len__(self):
        return len(self.train_data)


class TrainRetrievalInBatchDatasetDenseTopk(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer_name: str,
        dataset_name: str,
        num_neg: int,
        context_path: str,
        q_encoder,
        emb_path: str,
    ):
        dataset = load_from_disk(dataset_name)
        self.train_data = dataset["train"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_neg = num_neg

        df_topk = retrieve_from_embedding(
            dataset_name, q_encoder, tokenizer_name, emb_path, num_neg * 3, context_path
        )
        self.topk_answer = df_topk.original_context
        self.topk_context = df_topk.context
        self.in_batch_negative()

    def in_batch_negative(self):
        train_data = self.train_data
        num_neg = self.num_neg
        topk_context = self.topk_context
        p_with_neg = []

        for c_idx in range(len(train_data["context"])):
            c = train_data["context"][c_idx]
            p_with_neg.append(c)

            # randomly pick num_neg docs in topk
            idx_list = list(range(num_neg * 3))
            np.random.shuffle(idx_list)
            neg_cnt = 0
            for neg_idx in idx_list:

                p_neg = topk_context[c_idx][neg_idx]

                if neg_cnt == num_neg:
                    break
                elif c != p_neg:
                    p_with_neg.extend([p_neg])
                    neg_cnt += 1

        self.p_with_neg = p_with_neg

    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        train_data = self.train_data
        num_neg = self.num_neg
        question = train_data["question"][idx]
        p_with_neg = self.p_with_neg[
            idx * (num_neg + 1) : idx * (num_neg + 1) + (num_neg + 1)
        ]

        p_seqs = tokenizer(
            p_with_neg, padding="max_length", truncation=True, return_tensors="pt"
        )
        q_seqs = tokenizer(
            question, padding="max_length", truncation=True, return_tensors="pt"
        )
        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, num_neg + 1, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, num_neg + 1, max_len
        )

        p_input_ids = p_seqs["input_ids"]
        p_attention_mask = p_seqs["attention_mask"]
        p_token_type_ids = p_seqs["token_type_ids"]
        q_input_ids = q_seqs["input_ids"]
        q_attention_mask = q_seqs["attention_mask"]
        q_token_type_ids = q_seqs["token_type_ids"]

        return (
            p_input_ids,
            p_attention_mask,
            p_token_type_ids,
            q_input_ids,
            q_attention_mask,
            q_token_type_ids,
        )

    def __len__(self):
        return len(self.train_data)
