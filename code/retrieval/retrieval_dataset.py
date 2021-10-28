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


class TrainRetrievalInBatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer_name,
        dataset_name,
        num_neg,
        num_neg_sim,
        context_path,
        training_info,
    ):
        org_dataset = load_from_disk(dataset_name)
        self.train_data = org_dataset["train"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_neg = num_neg
        self.num_neg_sim = num_neg_sim
        with open(context_path, "r", encoding="utf-8") as f:
            self.wiki = json.load(f)
        if num_neg_sim < 0:
            self.df_info = None
        else:
            self.df_info = pd.read_pickle(training_info)

    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        train_data = self.train_data
        question = train_data["question"][idx]
        num_neg = self.num_neg + max(0, self.num_neg_sim)
        wiki = self.wiki
        df_info = self.df_info
        ans_id = np.array([train_data["document_id"][idx]])
        doc_id = np.random.randint(len(wiki), size=self.num_neg)
        doc_id = np.concatenate((ans_id, doc_id), axis=0)
        while self.num_neg_sim > 0:
            neg_sim_idxs = np.random.randint(
                len(df_info["docs_id"][idx]), size=self.num_neg_sim
            )
            check = False
            for neg_sim_id in neg_sim_idxs:
                if neg_sim_id in doc_id:
                    check = True
                    break
            if check == False:
                doc_id = np.concatenate((doc_id, neg_sim_idxs), axis=0)
                break

        p_with_neg = [wiki[str(sample_idx)]["text"] for sample_idx in doc_id]
        p_idxs = 0

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
        positive_ids = torch.tensor(p_idxs)

        return (
            p_input_ids,
            p_attention_mask,
            p_token_type_ids,
            q_input_ids,
            q_attention_mask,
            q_token_type_ids,
            positive_ids,
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
