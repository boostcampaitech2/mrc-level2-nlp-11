import torch
import os
import random
import json
import numpy as np
import pandas as pd
from datasets import (
    load_from_disk,
)

from transformers import (
    AutoTokenizer,
)

from retrieval import SparseRetrieval
from func import retrieve_from_embedding


class TrainRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer_name, dataset_name):
        org_dataset = load_from_disk(dataset_name)
        self.train_data = org_dataset["train"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        train_data = self.train_data
        question = train_data["question"][idx]
        context = train_data["context"][idx]

        p_seqs = tokenizer(
            context, padding="max_length", truncation=True, return_tensors="pt"
        )
        q_seqs = tokenizer(
            question, padding="max_length", truncation=True, return_tensors="pt"
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


class ValRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer_name, dataset_name):
        org_dataset = load_from_disk(dataset_name)
        self.val_data = org_dataset["validation"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        val_data = self.val_data
        question = val_data["question"][idx]
        context = val_data["context"][idx]

        p_seqs = tokenizer(
            context, padding="max_length", truncation=True, return_tensors="pt"
        )
        q_seqs = tokenizer(
            question, padding="max_length", truncation=True, return_tensors="pt"
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
        return len(self.val_data)


class TrainRetrievalInBatchOriginal(torch.utils.data.Dataset):
    def __init__(self, tokenizer_name, dataset_name, num_neg, context_path):
        org_dataset = load_from_disk(dataset_name)
        self.train_data = org_dataset["train"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_neg = num_neg
        self.corpus = WikiDataset(context_path, tokenizer_name).get_context()
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


class TrainRetrievalInBatchDatasetSparseTopk(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer_name,
        dataset_name,
        num_neg,
        context_path,
    ):
        dataset = load_from_disk(dataset_name)
        self.train_data = dataset["train"]
        # self.train_data = dataset["validation"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_neg = num_neg

        retriever = SparseRetrieval(
            tokenize_fn=self.tokenizer.tokenize,
            data_path="../../data",
            context_path="wikipedia_documents.json",
        )
        retriever.get_sparse_embedding()
        df_topk = retriever.retrieve_for_train(self.train_data, topk=num_neg * 2)
        self.topk_context = df_topk.context
        self.in_batch_negative()

    def in_batch_negative(self):
        train_data = self.train_data
        num_neg = self.num_neg
        topk_context = self.topk_context
        p_with_neg = []

        for c in train_data["context"]:
            p_with_neg.append(c)

            for p_neg in topk_context:
                if len(p_with_neg[-1]) == num_neg + 1:
                    break
                elif c != p_neg:
                    p_with_neg.extend(p_neg)

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


class TrainRetrievalInBatchDatasetDenseTopk(torch.utils.data.Dataset):
    def __init__(
        self, tokenizer_name, dataset_name, num_neg, context_path, q_encoder, emb_path
    ):
        dataset = load_from_disk(dataset_name)
        self.train_data = dataset["train"]
        # self.train_data = dataset["validation"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_neg = num_neg

        df_topk = retrieve_from_embedding(
            dataset_name, q_encoder, tokenizer_name, emb_path, num_neg * 3, context_path
        )

        self.topk_context = df_topk.context
        self.in_batch_negative()

    def in_batch_negative(self):
        train_data = self.train_data
        num_neg = self.num_neg
        topk_context = self.topk_context
        p_with_neg = []

        for c in train_data["context"]:
            p_with_neg.append(c)

            # for p_neg in topk_context:
            # randomly pick num_neg docs in topk
            idx_list = np.random.randint(num_neg * 3, size=num_neg)
            for i in idx_list:
                p_neg = topk_context[i]

                if len(p_with_neg[-1]) == num_neg + 1:
                    break
                elif c != p_neg:
                    p_with_neg.extend(p_neg)

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


class TrainRetrievalInBatchDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer_name, dataset_name, num_neg, context_path):
        org_dataset = load_from_disk(dataset_name)
        self.train_data = org_dataset["train"]
        # self.train_data = org_dataset["validation"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_neg = num_neg
        self.corpus = WikiDataset(context_path, tokenizer_name).get_context()
        print("in batch negative sampling from wiki")
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


class WikiDataset:
    def __init__(self, context_path, tokenizer_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.ids = list(range(len(self.contexts)))

    def get_context(self):
        return self.contexts

    def get_tokens(self):
        contexts = self.contexts
        return self.tokenizer(
            contexts, padding="max_length", truncation=True, return_tensors="pt"
        ).to("cuda")
