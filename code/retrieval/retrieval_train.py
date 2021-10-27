import os
import pickle
import numpy as np
import pandas as pd
import argparse
from pprint import pprint
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

import torch
import torch.nn.functional as F
from retrieval_dataset import (
    TrainRetrievalDataset,
    ValRetrievalDataset,
    TrainRetrievalInBatchDataset,
    WikiDataset,
)
from retrieval_model import BertEncoder

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    set_seed,
)


class DenseRetrieval:
    def __init__(
        self,
        args,
        model_args,
        train_dataset,
        val_dataset,
        num_neg,
        p_encoder,
        q_encoder,
    ) -> None:
        self.args = args
        self.model_args = model_args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_neg = num_neg
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        torch.cuda.empty_cache()

    def save_embedding(self):

        wiki_data_list = WikiDataset(self.args.context_path, self.args.tokenizer_name)

        emb_path = self.args.save_pickle_path
        p_encoder = self.p_encoder
        p_embs = []
        with torch.no_grad():
            p_encoder.eval()
            test_p = wiki_data_list.get_tokens()

            for p in tqdm(wiki_data_list.contexts):
                p = wiki_data_list.tokenizer(
                    p, padding="max_length", truncation=True, return_tensors="pt"
                ).to("cuda")
                p_emb = p_encoder(**p).to("cpu").detach().numpy()
                p_embs.append(p_emb)

        p_embs = torch.Tensor(p_embs).squeeze()

        p_embedding = p_embs
        with open(emb_path, "wb") as file:
            pickle.dump(p_embedding, file)
        print("Embedding pickle saved.")

    def train(self):
        model_args = self.model_args
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder
        num_neg = self.num_neg
        train_dataset = self.train_dataset
        batch_size = model_args.per_device_train_batch_size

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": model_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": model_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in q_encoder.named_parameters()
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
            ## train_dataloader
            len(train_dataset)
            // model_args.gradient_accumulation_steps
            * model_args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=model_args.warmup_steps,
            num_training_steps=t_total,
        )

        global_step = 0
        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size
        )

        for _ in tqdm(range(int(model_args.num_train_epochs)), desc="Epoch"):

            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    p_encoder.train()
                    q_encoder.train()

                    targets = batch[6].long()
                    targets = targets.to(model_args.device)
                    p_inputs = {
                        "input_ids": batch[0]
                        .view(batch_size * (num_neg + 1), -1)
                        .to(model_args.device),
                        "attention_mask": batch[1]
                        .view(batch_size * (num_neg + 1), -1)
                        .to(model_args.device),
                        "token_type_ids": batch[2]
                        .view(batch_size * (num_neg + 1), -1)
                        .to(model_args.device),
                    }

                    q_inputs = {
                        "input_ids": batch[3]
                        .view(batch_size, -1)
                        .to(model_args.device),
                        "attention_mask": batch[4]
                        .view(batch_size, -1)
                        .to(model_args.device),
                        "token_type_ids": batch[5]
                        .view(batch_size, -1)
                        .to(model_args.device),
                    }

                    p_outputs = p_encoder(**p_inputs)
                    q_outputs = q_encoder(**q_inputs)

                    p_outputs = torch.transpose(
                        p_outputs.view(batch_size, num_neg + 1, -1), 1, 2
                    )
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    p_encoder.zero_grad()
                    q_encoder.zero_grad()

                    torch.cuda.empty_cache()
                    global_step += 1
                    del p_inputs, q_inputs
                    if global_step % 100 == 0:  # args 추가 필요
                        wandb.log({"loss": loss}, step=global_step)

                    # if global_step % 50 == 0:
                    #    self.eval(p_encoder, q_encoder, global_step)

        p_encoder.save_pretrained(save_directory=args.save_path_p)  # args
        q_encoder.save_pretrained(save_directory=args.save_path_q)  # args

    def eval(self, p_encoder, q_encoder, global_step):
        val_dataset = self.val_dataset
        p_encoder.eval()
        q_encoder.eval()

        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=1)

        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):  # 데이터 셋 class
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                p_inputs = {
                    "input_ids": batch[0].view(1, -1),
                    "attention_mask": batch[1].view(1, -1),
                    "token_type_ids": batch[2].view(1, -1),
                }
                q_inputs = {
                    "input_ids": batch[3].view(1, -1),
                    "attention_mask": batch[4].view(1, -1),
                    "token_type_ids": batch[5].view(1, -1),
                }

                p_outputs = p_encoder(**p_inputs)
                q_outputs = q_encoder(**q_inputs)

                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))

            wandb.log({"Eval sim_scores": sim_scores[0]}, step=global_step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--save_path_q", default="./encoder/q_encoder", type=str, help=""
    )
    parser.add_argument(
        "--save_path_p", default="./encoder/p_encoder", type=str, help=""
    )
    parser.add_argument(
        "--dataset_name", default="../../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--tokenizer_name",
        default="bert-base-multilingual-cased",
        type=str,
        help="",
    )

    parser.add_argument(
        "--context_path",
        default="../../data/wikipedia_documents.json",
        type=str,
        help="context path for retrieval",
    )
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument(
        "--run_name", default="dense_retrieval", type=str, help="wandb run name"
    )
    parser.add_argument(
        "--num_train_epochs", default=10, type=int, help="number of epochs for train"
    )
    parser.add_argument(
        "--train_batch", default=2, type=int, help="batch size for train"
    )
    parser.add_argument(
        "--eval_batch", default=2, type=int, help="batch size for evaluation"
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="learning rate for train"
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="weight decay coeff for train"
    )
    parser.add_argument(
        "--num_neg", default=3, type=int, help="number of negative samples for training"
    )
    parser.add_argument(
        "--num_neg_sim", default=-1, type=int, help="number of random_neg_sample with similar docs"
    )
    parser.add_argument(
        "--random_seed", default=211, type=int, help="random seed for numpy and torch"
    )
    parser.add_argument(
        "--p_enc_name_or_path",
        default="bert-base-multilingual-cased",
        type=str,
        help="name or path for p_encoder",
    )
    parser.add_argument(
        "--q_enc_name_or_path",
        default="bert-base-multilingual-cased",
        type=str,
        help="name or path for q_encoder",
    )
    parser.add_argument(
        "--save_pickle_path",
        default="../../data/dense_embedding.bin",
        type=str,
        help="wiki embedding save path",
    )
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    set_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    p_encoder = BertEncoder.from_pretrained(args.p_enc_name_or_path).cuda()
    q_encoder = BertEncoder.from_pretrained(args.q_enc_name_or_path).cuda()

    # 이후 arg 뺄 수 있으면 빼기
    model_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
    )

    wandb.init(entity="ai_esg", name=args.run_name)
    wandb.config.update(model_args)

    # TrainRetrievalDataset
    train_dataset = TrainRetrievalInBatchDataset(

        args.model_name_or_path, args.dataset_name, args.num_neg, args.num_neg_sim

    )
    validation_dataset = ValRetrievalDataset(args.tokenizer_name, args.dataset_name)
    retriever = DenseRetrieval(
        args,
        model_args,
        train_dataset,
        validation_dataset,
        args.num_neg + max(0,args.num_neg_sim),
        p_encoder,
        q_encoder,
    )

    #    retriever.train()
    retriever.save_embedding()
