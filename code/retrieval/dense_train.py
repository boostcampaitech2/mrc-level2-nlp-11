import sys
import os
import pickle
import argparse
import json
import wandb

from typing import List, Tuple, NoReturn, Any, Optional, Union
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    set_seed,
)

from dense_dataset import (
    TrainRetrievalDataset,
    TrainRetrievalRandomDataset,
)
from dense_model import BertEncoder
from func import (
    neg_sample_input,
    neg_sample_sim_scores,
    inbatch_input,
    inbatch_sim_scores,
)

sys.path.append("../read")
from utils_qa import preprocess, get_preprocess_dataset, get_preprocess_wiki


class DenseRetrieval:
    def __init__(self, args, num_neg, p_encoder, q_encoder,) -> None:
        self.args = args
        self.num_neg = num_neg
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.cuda.empty_cache()

    def save_embedding(self, save_path):

        with open(self.args.context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name)
        p_encoder = self.p_encoder
        p_embs = []

        with torch.no_grad():
            p_encoder.eval()

            for p in tqdm(contexts):
                p = tokenizer(
                    p, padding="max_length", truncation=True, return_tensors="pt"
                ).to("cuda")
                p_emb = p_encoder(**p).to("cpu").detach().numpy()
                p_embs.append(p_emb)

        p_embs = torch.Tensor(p_embs).squeeze()

        p_embedding = p_embs
        with open(save_path, "wb") as file:
            pickle.dump(p_embedding, file)
        print("Embedding pickle saved.")

    def train(self):
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder
        num_neg = self.num_neg
        args = self.args
        batch_size = args.batch_size

        # get in-batch dataset
        if args.in_batch:
            train_dataset = TrainRetrievalDataset(
                args.tokenizer_name, args.dataset_name,
            )
            train_dataloader = DataLoader(
                train_dataset, shuffle=True, batch_size=batch_size, drop_last=True
            )

        ## get negative samples from wiki for first epoch
        else:
            train_dataset = TrainRetrievalRandomDataset(
                args.tokenizer_name, args.dataset_name, num_neg, args.context_path,
            )
            train_dataloader = DataLoader(
                train_dataset, shuffle=True, batch_size=batch_size
            )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
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
                "weight_decay": args.weight_decay,
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
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
        )
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
        )

        global_step = 0
        epoch = 0
        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        for _ in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):

            epoch += 1
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    p_encoder.train()
                    q_encoder.train()

                    global_step += 1

                    if args.in_batch:
                        q_inputs, p_inputs, targets = inbatch_input(
                            batch, batch_size, self.device
                        )

                        p_outputs = p_encoder(**p_inputs)
                        q_outputs = q_encoder(**q_inputs)

                        sim_scores = inbatch_sim_scores(q_outputs, p_outputs)

                    else:
                        q_inputs, p_inputs, targets = neg_sample_input(
                            batch, batch_size, self.device, num_neg
                        )

                        p_outputs = p_encoder(**p_inputs)
                        q_outputs = q_encoder(**q_inputs)

                        sim_scores = neg_sample_sim_scores(
                            q_outputs, p_outputs, batch_size, num_neg
                        )

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    p_encoder.zero_grad()
                    q_encoder.zero_grad()

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

                    if global_step % args.log_step == 0:
                        wandb.log({"loss": loss}, step=global_step)

            if epoch % args.save_epoch == 0:
                p_encoder.save_pretrained(
                    save_directory=args.save_path_p + "_" + str(epoch)
                )
                q_encoder.save_pretrained(
                    save_directory=args.save_path_q + "_" + str(epoch)
                )

                if args.in_batch:
                    self.save_embedding(
                        args.save_pickle_path + "_epoch_{}.bin".format(epoch)
                    )

            if not args.in_batch and epoch != args.num_train_epochs:

                ## get negative samples from dense top k
                self.save_embedding(
                    args.save_pickle_path + "_epoch_{}.bin".format(epoch)
                )

                train_dataset = TrainRetrievalRandomDatasetDenseTopk(
                    args.tokenizer_name,
                    args.dataset_name,
                    num_neg,
                    args.context_path,
                    q_encoder,
                    save_pickle_path_full,
                )
                train_dataloader = DataLoader(
                    train_dataset, shuffle=True, batch_size=batch_size
                )


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
        "--tokenizer_name", default="klue/bert-base", type=str, help="",
    )
    parser.add_argument(
        "--context_path",
        default="../../data/wikipedia_documents.json",
        type=str,
        help="context for retrieval",
    )
    parser.add_argument(
        "--run_name", default="dense_retrieval", type=str, help="wandb run name"
    )
    parser.add_argument(
        "--num_train_epochs", default=1, type=int, help="number of epochs for train"
    )
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size for train"
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
    parser.add_argument(
        "--save_epoch", default=10, type=int, help="save encoders per epoch"
    )
    parser.add_argument(
        "--log_step", default=100, type=int, help="log loss to wandb per step"
    )
    parser.add_argument(
        "--in_batch",
        default=False,
        type=bool,
        help="if True, training over in-batch sample",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="adam epsilon for optimizer"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="gradient accumulation steps for scheduler",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="warmup steps for scheduler"
    )

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    set_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    get_preprocess_dataset("../../../data/")
    get_preprocess_wiki("../../../data/")

    p_encoder = BertEncoder.from_pretrained(args.p_enc_name_or_path).cuda()
    q_encoder = BertEncoder.from_pretrained(args.q_enc_name_or_path).cuda()

    wandb.init(entity="ai_esg", name=args.run_name)

    retriever = DenseRetrieval(args, args.num_neg, p_encoder, q_encoder,)
    retriever.train()
    retriever.save_embedding(args.save_pickle_path + ".bin")
