import os
import pickle
import argparse
import json
from tqdm.auto import tqdm

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from dense_model import BertEncoder
from datasets import load_from_disk


class RetrievalInference:
    def __init__(
        self, args, q_encoder: BertEncoder, tokenizer: AutoTokenizer, context_path: str
    ):
        self.pickle_path = args.pickle_path
        self.q_encoder = q_encoder
        self.tokenizer = tokenizer

        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    def get_dense_embedding(self):
        emd_path = self.pickle_path

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)

            print("Embedding pickle load.")

    def retrieval(self, query_or_dataset, topk=1):
        assert self.p_embedding is not None
        total = []
        doc_scores, doc_indices = self.get_relevant_doc_bulk(
            query_or_dataset["question"], k=topk
        )
        for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
            context_array = []
            for pid in doc_indices[idx]:
                context = "".join(self.contexts[pid])
                context_array.append(context)
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context_id": doc_indices[idx],
                "context": context_array,
                "scores": doc_scores[idx],
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas

    def get_relevant_doc_bulk(self, queries: list, k=1):
        q_encoder = self.q_encoder
        q_embs = []

        with torch.no_grad():
            q_encoder.eval()

            for q in tqdm(queries):
                q = tokenizer(
                    q, padding="max_length", truncation=True, return_tensors="pt"
                ).to("cuda")
                q_emb = q_encoder(**q).to("cpu").detach().numpy()
                q_embs.append(q_emb)

        q_embs = np.array(q_embs)
        q_embs = torch.Tensor(q_embs).squeeze()
        result = torch.matmul(q_embs, torch.transpose(self.p_embedding, 0, 1))

        if not isinstance(result, np.ndarray):
            result = np.array(result.tolist())
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices

    def get_acc_score(self, df: pd.DataFrame):
        df["correct"] = False
        df["correct_rank"] = 0
        for i in tqdm(range(len(df)), desc="check tok_n"):
            count = 0
            for context in df.iloc[i]["context"]:
                count += 1
                if df.iloc[i]["original_context"] == context:
                    df.at[i, "correct"] = True
                    df.at[i, "correct_rank"] = count

        return df

    def print_result(self, df: pd.DataFrame, length: int):
        # for i in range(length):
        # print("=======================================")
        # f = df.iloc[i]
        # print(f'Question         : {f["question"]}')
        # print(f'original_context : {f["original_context"]}')
        # print("\n\n")
        # for i in range(len(f["context"])):
        #     print(f'score\t:{f["scores"][i]},\ncontext\t: {f["context"][i]}\n')
        # print("=======================================")

        print(
            "correct retrieval result by exhaustive search",
            df["correct"].sum() / len(df),
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
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
        "--pickle_path",
        default="../../data/dense_embedding.bin",
        type=str,
        help="wiki embedding path",
    )
    parser.add_argument(
        "--context_path",
        default="../../data/wikipedia_documents.json",
        type=str,
        help="context for retrieval",
    )
    parser.add_argument(
        "--load_path_q",
        default="./encoder/q_encoder",
        type=str,
        help="q_encoder saved path",
    )

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    q_encoder = BertEncoder.from_pretrained(args.load_path_q).cuda()

    org_dataset = load_from_disk(args.dataset_name)
    train_dataset = org_dataset["train"]
    validation_dataset = org_dataset["validation"]

    retrieval = RetrievalInference(args, q_encoder, tokenizer, args.context_path)
    retrieval.get_dense_embedding()

    print("----- val top-5 -----")
    df = retrieval.retrieval(validation_dataset, topk=5)
    df = retrieval.get_acc_score(df)
    retrieval.print_result(df, 5)
    print("----- val top-10 -----")
    df = retrieval.retrieval(validation_dataset, topk=10)
    df = retrieval.get_acc_score(df)
    retrieval.print_result(df, 10)

    print("----- train top-5 -----")
    df = retrieval.retrieval(train_dataset, topk=5)
    df = retrieval.get_acc_score(df)
    retrieval.print_result(df, 5)
    print("----- train top-10 -----")
    df = retrieval.retrieval(train_dataset, topk=10)
    df = retrieval.get_acc_score(df)
    retrieval.print_result(df, 10)
