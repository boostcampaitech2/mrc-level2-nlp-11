import os
import torch
import json
import numpy as np
import pickle
from tqdm.auto import tqdm
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, NoReturn, Optional, Union
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)


class BM25Retrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../../../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        is_retrain=False,
        k1=1.2,
        b=0.75,
    ):

        self.is_retrain = is_retrain
        save_dir = os.path.join(data_path, "bm25")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.embed_path = os.path.join(save_dir, "embedding.bin")
        self.encoder_path = os.path.join(save_dir, "encoder.bin")
        self.idf_encoder_path = os.path.join(save_dir, "idf_encoder.bin")

        self.tokenizer = tokenize_fn
        self.idf_path = os.path.join(save_dir, "idf.bin")

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["title"] + ": " + v["text"] for v in wiki.values()])
        )

        self.b = b
        self.k1 = k1
        self.encoder = TfidfVectorizer(
            tokenizer=self.tokenizer, ngram_range=(1, 2), use_idf=False, norm=None
        )
        self.idf_encoder = TfidfVectorizer(
            tokenizer=self.tokenizer, ngram_range=(1, 2), use_idf=True, norm=None
        )

        self.dls = np.zeros(len(self.contexts))

        for idx, context in enumerate(self.contexts):
            self.dls[idx] = len(context)

        self.avdl = np.mean(self.dls)
        self.p_embedding = None
        self.idf = None

    def get_scores(self, p_embedding, query_vec):
        b, k1, avdl, = (
            self.b,
            self.k1,
            self.avdl,
        )
        len_p = self.dls
        p_emb_for_q = p_embedding[:, query_vec.indices]
        denom = p_emb_for_q + (k1 * (1 - b + b * len_p / avdl))[:, None]
        idf = self.idf[None, query_vec.indices] - 1.0
        idf_broadcasted = np.broadcast_to(idf, p_emb_for_q.shape)
        numer = p_emb_for_q * (k1 + 1)

        result = (np.multiply((numer / denom), idf_broadcasted)).sum(1).A1

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        return result

    def _exec_embedding(self):
        self.p_embedding = self.encoder.fit_transform(
            tqdm(self.contexts, desc="TF calculation: ")
        )
        self.idf = self.idf_encoder.fit(
            tqdm(self.contexts, desc="IDF calculation: ")
        ).idf_

        return self.p_embedding, self.encoder, self.idf, self.idf_encoder

    def get_sparse_embedding(self) -> NoReturn:
        if (
            os.path.isfile(self.embed_path)
            and os.path.isfile(self.encoder_path)
            and os.path.isfile(self.idf_encoder_path)
            and os.path.isfile(self.idf_path)
            and not self.is_retrain
        ):
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            with open(self.encoder_path, "rb") as f:
                self.encoder = pickle.load(f)

            with open(self.idf_encoder_path, "rb") as f:
                self.idf_encoder = pickle.load(f)

            with open(self.idf_path, "rb") as f:
                self.idf = pickle.load(f)
        else:
            (
                self.p_embedding,
                self.encoder,
                self.idf,
                self.idf_encoder,
            ) = self._exec_embedding()

            with open(self.embed_path, "wb") as f:
                pickle.dump(self.p_embedding, f)

            with open(self.encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)

            with open(self.idf_path, "wb") as f:
                pickle.dump(self.idf, f)

            with open(self.idf_encoder_path, "wb") as f:
                pickle.dump(self.idf_encoder, f)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        result = self.get_scores(query)
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, topk: Optional[int] = 5
    ) -> Tuple[List, List]:
        query_vecs = self.encoder.transform(queries)

        doc_scores = []
        doc_indices = []

        p_embedding = self.p_embedding.tocsc()

        self.results = []

        for query_vec in tqdm(query_vecs):

            result = self.get_scores(p_embedding, query_vec)
            self.results.append(result)

            sorted_result_idx = np.argsort(result)[::-1]
            doc_score, doc_indice = (
                result[sorted_result_idx].tolist()[:topk],
                sorted_result_idx.tolist()[:topk],
            )
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)

        if not isinstance(self.results, np.ndarray):
            self.results = np.array(self.results)

        return doc_scores, doc_indices

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.idf is not None
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        else:
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            print("-----Retrieve Start-----")
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], topk=topk
            )

            for idx in tqdm(range(len(query_or_dataset["question"]))):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": query_or_dataset["question"][idx],
                    "id": query_or_dataset["id"][idx],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": [self.contexts[pid] for pid in doc_indices[idx]],
                }
                if (
                    "context" in query_or_dataset[idx].keys()
                    and "answers" in query_or_dataset[idx].keys()
                ):
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = query_or_dataset["context"][idx]
                    tmp["answers"] = query_or_dataset["answers"][idx]
                    tmp["original_context_id"] = query_or_dataset["document_id"][idx]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--random_seed", default=211, type=int, help="random seed for numpy and torch"
    )
    parser.add_argument(
        "--dataset_name", default="../../../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument(
        "--data_path", default="../../../data", type=str, help="dataset directory path"
    )
    parser.add_argument(
        "--context_path",
        default="wikipedia_documents.json",
        type=str,
        help="",
    )
    parser.add_argument(
        "--retrain", default=False, type=bool, help="to retrain(1) or not(0)"
    )
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    print("*" * 40, "query dataset loaded", "*" * 40)

    retriever = BM25Retrieval(tokenize_fn=tokenizer.tokenize, is_retrain=args.retrain)

    retriever.get_sparse_embedding()

    # top-200 에 대해 몇번째에서 맞추는지 테스트
    df = retriever.retrieve(org_dataset["train"], topk=200)
    count = 0
    cnt = [0 for _ in range(201)]
    for i in range(len(df)):
        for n, j in enumerate(df["context"][i]):
            if df["original_context"][i] == ": ".join(j.split(": ")[1:]):
                cnt[n + 1] += 1
                count += 1
                break

    for n, i in enumerate(cnt):
        if i != 0:
            print(f"{n} : {i}")

    print(
        "correct retrieval result by exhaustive search",
        count / len(df),
    )
