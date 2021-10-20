import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
import wandb

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

import torch
import torch.nn.functional as F
from retrieval_model import BertEncoder
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)


class DenseRetrieval:
    def __init__(
        self,
        args,
        model_args,
        dataset,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder,
        context_path="wikipedia_documents.json",
        data_path="../../data/",
    ):

        self.args = args
        self.model_args = model_args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        torch.cuda.empty_cache()
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.ids = list(range(len(self.contexts)))
        # print(self.contexts)
        self.prepare_in_batch_negative(num_neg=num_neg)

    def get_dense_embedding(self) -> NoReturn:
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.args.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            p_embs = []
            with torch.no_grad():
                p_encoder.eval()

                for p in tqdm(self.contexts):
                    p = tokenizer(
                        p, padding="max_length", truncation=True, return_tensors="pt"
                    ).to("cuda")
                    p_emb = p_encoder(**p).to("cpu").detach().numpy()
                    p_embs.append(p_emb)

            p_embs = torch.Tensor(p_embs).squeeze()
            print(p_embs.size())

            self.p_embedding = p_embs
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset["context"]])))
        p_with_neg = []

        for c in dataset["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            p_with_neg, padding="max_length", truncation=True, return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, num_neg + 1, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, num_neg + 1, max_len
        )

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.model_args.per_device_train_batch_size,
        )

        valid_seqs = tokenizer(
            dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"],
            valid_seqs["attention_mask"],
            valid_seqs["token_type_ids"],
        )
        self.passage_dataloader = DataLoader(
            passage_dataset, batch_size=self.model_args.per_device_train_batch_size
        )

    def train(self, model_args=None):
        if model_args is None:
            model_args = self.model_args
        batch_size = model_args.per_device_train_batch_size

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": model_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": model_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=model_args.learning_rate,
            eps=model_args.adam_epsilon,
        )
        t_total = (
            len(self.train_dataloader)
            // model_args.gradient_accumulation_steps
            * model_args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=model_args.warmup_steps,
            num_training_steps=t_total,
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(model_args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                count = 0
                for batch in tepoch:
                    count += 1
                    p_encoder.train()
                    q_encoder.train()

                    # targets = torch.zeros(
                    #     batch_size
                    # ).long()  # positive example은 전부 첫 번째에 위치하므로
                    # targets = targets.to(model_args.device)
                    # 정상적으로 출력이 되는 코드에서도
                    targets = torch.arange(0, batch_size).long()  # 실습 코드에서 가져옴 5강
                    targets = targets.to(model_args.device)
                    p_inputs = {
                        "input_ids": batch[0]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(model_args.device),
                        "attention_mask": batch[1]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(model_args.device),
                        "token_type_ids": batch[2]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(model_args.device),
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(model_args.device),
                        "attention_mask": batch[4].to(model_args.device),
                        "token_type_ids": batch[5].to(model_args.device),
                    }

                    # (batch_size*(num_neg+1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size*, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, -1, self.num_neg + 1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(
                        q_outputs, p_outputs
                    ).squeeze()  # (batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    if count % 100 == 0:
                        wandb.log({"loss": loss.item()}, step=count)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

        p_encoder.save_pretrained(save_directory="./encoder/p_encoder_ver2")
        q_encoder.save_pretrained(save_directory="./encoder/q_encoder_ver2")

    def get_relevant_doc(
        self, query, k=1, model_args=None, p_encoder=None, q_encoder=None
    ):

        if model_args is None:
            model_args = self.model_args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors="pt"
            ).to(model_args.device)
            q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(model_args.device) for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                p_emb = p_encoder(**p_inputs).to("cpu")
                p_embs.append(p_emb)

        # (num_passage, emb_dim)
        p_embs = torch.stack(p_embs, dim=0).view(
            len(self.passage_dataloader.dataset), -1
        )  ## context 에서 안됨

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        return rank[:k]


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="../../data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument("--run_name", default="dense_retrieval", type=str, help="")

    args = parser.parse_args()

    org_dataset = load_from_disk(args.dataset_name)
    train_dataset = org_dataset["train"]
    validation_dataset = org_dataset["validation"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    p_encoder = BertEncoder.from_pretrained("./encoder/p_encoder_ver1")
    q_encoder = BertEncoder.from_pretrained("./encoder/q_encoder_ver1")
    p_encoder.cuda()
    q_encoder.cuda()

    model_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=500,
    )
    wandb.init(entity="ai_esg", name=args.run_name)
    wandb.config.update(model_args)

    retriever = DenseRetrieval(
        args, model_args, train_dataset, 3, tokenizer, p_encoder, q_encoder
    )
    retriever.train()

    retriever.get_dense_embedding()
