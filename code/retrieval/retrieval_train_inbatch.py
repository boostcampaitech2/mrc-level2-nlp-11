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
import json
import torch
import torch.nn.functional as F
from retrieval_dataset import (
    TrainRetrievalDataset,
    ValRetrievalDataset,
    TrainRetrievalRandomDataset,
    WikiDataset,
)
from retrieval_model import BertEncoder, RobertaEncoder

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    set_seed,
)

# from func import retrieve_from_embedding, retrieval_acc
from func import inbatch_input, inbatch_sim_scores

import sys

sys.path.append("../read")

from utils_qa import preprocess, get_preprocess_dataset, get_preprocess_wiki


class DenseRetrieval:
    def __init__(
        self,
        args,
        model_args,
        val_dataset,
        num_neg,
        p_encoder,
        q_encoder,
    ) -> None:
        self.args = args
        self.model_args = model_args
        self.val_dataset = val_dataset
        self.num_neg = num_neg
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        torch.cuda.empty_cache()

    def save_embedding(self, save_path):

        wiki_data_list = WikiDataset(self.args.context_path, self.args.tokenizer_name)

        # emb_path = self.args.save_pickle_path
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
        with open(save_path, "wb") as file:
            pickle.dump(p_embedding, file)
        print("Embedding pickle saved.")

    def train(self):
        model_args = self.model_args
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder
        num_neg = self.num_neg
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
            # len(train_dataset) ######### should to change ########
            3952
            # 240
            // model_args.gradient_accumulation_steps
            * model_args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=model_args.warmup_steps,
            num_training_steps=t_total,
        )

        global_step = 0
        epoch = 0

        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        # get in-batch dataset
        train_dataset = TrainRetrievalDataset(
            self.args.tokenizer_name,
            self.args.dataset_name,
        )
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size, drop_last=True
        )

        for _ in tqdm(range(int(model_args.num_train_epochs)), desc="Epoch"):

            epoch += 1
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    p_encoder.train()
                    q_encoder.train()

                    global_step += 1

                    q_inputs, p_inputs, targets = inbatch_input(
                        batch, batch_size, model_args.device
                    )

                    p_outputs = p_encoder(**p_inputs)
                    q_outputs = q_encoder(**q_inputs)

                    sim_scores = inbatch_sim_scores(q_outputs, p_outputs)

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
                retriever.save_embedding(
                    args.save_pickle_path + "_" + str(epoch) + ".bin"
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
        "--tokenizer_name",
        default="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument(
        "--context_path",
        default="../../data/wikipedia_documents.json",
        type=str,
        help="context for retrieval",
    )
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument(
        "--run_name", default="dense_retrieval", type=str, help="wandb run name"
    )
    parser.add_argument(
        "--num_train_epochs", default=1, type=int, help="number of epochs for train"
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
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    set_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    get_preprocess_dataset("/opt/ml/data/")
    get_preprocess_wiki("/opt/ml/data/")

    p_encoder = BertEncoder.from_pretrained(args.p_enc_name_or_path).cuda()
    q_encoder = BertEncoder.from_pretrained(args.q_enc_name_or_path).cuda()

    # 이후 arg 뺄 수 있으면 빼기
    model_args = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
    )

    wandb.init(entity="ai_esg", name=args.run_name)
    wandb.config.update(model_args)

    validation_dataset = ValRetrievalDataset(args.tokenizer_name, args.dataset_name)

    retriever = DenseRetrieval(
        args,
        model_args,
        validation_dataset,
        args.num_neg,
        p_encoder,
        q_encoder,
    )
    retriever.train()
    # retriever.save_embedding(args.save_pickle_path + ".bin")
