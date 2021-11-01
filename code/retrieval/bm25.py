import os
import torch
import json
import numpy as np
import pickle
from tqdm.auto import tqdm
import pandas as pd
import argparse
from typing import List, Tuple, NoReturn, Any, Optional, Union
from rank_bm25 import BM25Okapi
from retrieval_dataset import (
    TrainRetrievalDataset,
    ValRetrievalDataset,
    TrainRetrievalInBatchDataset,
    WikiDataset,
)
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
import retrieval


class BM25Retrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../../../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ):
        self.data_path = data_path
        self.tokenize_fn = tokenize_fn
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        # self.contexts = [wiki[str(i)]["text"] for i in range(len(wiki))]
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        k1 = 1.2
        b = 0.75
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq * (k1 + 1) / (q_freq + k1 * (1 - b + b * doc_len / self.avgdl))
            )
        return score

    def get_sparse_embedding(self) -> NoReturn:
        idf_name = f"idf.bin"
        doc_freqs_name = f"doc_freqs.bin"
        avgdl_name = f"avgdl.bin"
        corpus_size_name = f"corpus_size.bin"
        doc_len_name = f"doc_len.bin"
        idf_path = os.path.join(self.data_path, idf_name)
        doc_freqs_path = os.path.join(self.data_path, doc_freqs_name)
        avgdl_path = os.path.join(self.data_path, avgdl_name)
        corpus_size_path = os.path.join(self.data_path, corpus_size_name)
        doc_len_path = os.path.join(self.data_path, doc_len_name)

        if (
            os.path.isfile(doc_freqs_path)
            and os.path.isfile(idf_path)
            and os.path.isfile(avgdl_path)
            and os.path.isfile(corpus_size_path)
            and os.path.isfile(doc_len_path)
        ):
            with open(doc_freqs_path, "rb") as file:
                self.doc_freqs = pickle.load(file)
            with open(idf_path, "rb") as file:
                self.idf = pickle.load(file)
            with open(avgdl_path, "rb") as file:
                self.avgdl = pickle.load(file)
            with open(corpus_size_path, "rb") as file:
                self.corpus_size = pickle.load(file)
            with open(doc_len_path, "rb") as file:
                self.doc_len = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.bm25 = BM25Okapi(
                self.contexts, tokenizer=self.tokenize_fn, k1=1.2, b=0.75
            )
            self.doc_freqs = self.bm25.doc_freqs
            self.idf = self.bm25.idf
            self.avgdl = self.bm25.avgdl
            self.corpus_size = self.bm25.corpus_size
            self.doc_len = self.bm25.doc_len
            with open(doc_freqs_path, "wb") as file:
                pickle.dump(self.doc_freqs, file)
            with open(idf_path, "wb") as file:
                pickle.dump(self.idf, file)
            with open(avgdl_path, "wb") as file:
                pickle.dump(self.avgdl, file)
            with open(corpus_size_path, "wb") as file:
                pickle.dump(self.corpus_size, file)
            with open(doc_len_path, "wb") as file:
                pickle.dump(self.doc_len, file)
            print("Embedding pickle saved.")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        result = self.get_scores(query)
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        result = [self.get_scores(query) for query in queries]
        print(result)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
            print(doc_scores[-1], doc_indices[-1])

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
            # elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            print("-----Retrieve Start-----")
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )
            print("-------test line--------")

            for idx in range(len(query_or_dataset["question"])):

                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": query_or_dataset["question"][idx],
                    "id": query_or_dataset["id"][idx],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": [self.contexts[pid] for pid in doc_indices[idx]],
                }
                if (
                    "context" in query_or_dataset.keys()
                    and "answers" in query_or_dataset.keys()
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
        # default="klue/bert-base",
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
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    # train_dataset = TrainRetrievalDataset(args.model_name_or_path, args.dataset_name)
    # validation_dataset = ValRetrievalDataset(args.model_name_or_path, args.dataset_name)

    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    # print(full_ds["question"])

    retriever = BM25Retrieval(tokenize_fn=tokenizer.tokenize)
    # retriever = retrieval.SparseRetrieval(
    #     tokenize_fn=tokenizer.tokenize,
    #     data_path=args.data_path,
    #     context_path=args.context_path,
    # )
    retriever.get_sparse_embedding()
    # retriever.retrieve(full_ds, topk=5)
    # retriever.retrieve(org_dataset["train"]["question"][20], topk=5)
    # print(org_dataset["train"]["context"][20])
    # for query in full_ds["question"]:
    #     print(retriever.retrieve(query, topk=5))
    #     break
    # print(type(retriever.contexts))

    df = retriever.retrieve(org_dataset["train"], topk=5)
    print(df)
    count = 0
    # [print(f"{text}\n\n\n\n") for text in df["context"][0]]
    for i in range(len(df)):
        # print(df["original_context_id"][i], df["context_id"][i])
        # if df["original_context_id"][i] in df["context_id"][i]:
        if df["original_context"][i] in df["context"][i]:
            count += 1

    print(
        "correct retrieval result by exhaustive search",
        count / len(df),  # acc
    )
