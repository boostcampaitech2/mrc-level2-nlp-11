import pickle
import torch
import numpy as np
import pandas as pd
import json

from datasets import load_from_disk
from tqdm.auto import tqdm

from transformers import AutoTokenizer


def retrieve_from_embedding(
    dataset_path, q_encoder, tokenizer_name, emb_path: str, topk: int, context_path: str
):

    """
    Input : Dataset path(question), Question encoder, Tokenizer, Embedding path, number of top score documents (topk), wikipedia json file path
    Output : Dataframe of topk dense retrieval documents
    """

    ## get relevant doc bulk
    dataset = load_from_disk(dataset_path)
    # dataset = dataset['validation']
    dataset = dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(emb_path, "rb") as file:
        p_embedding = pickle.load(file)
    q_embs = []

    with torch.no_grad():
        q_encoder.eval()

        for q in tqdm(dataset["question"]):
            q = tokenizer(
                q, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = q_encoder(**q).to("cpu").detach().numpy()
            q_embs.append(q_emb)

    q_embs = torch.Tensor(q_embs).squeeze()
    result = torch.matmul(q_embs, torch.transpose(p_embedding, 0, 1))

    if not isinstance(result, np.ndarray):
        result = np.array(result.tolist())
    doc_scores = []
    doc_indices = []
    for i in range(result.shape[0]):
        sorted_result = np.argsort(result[i, :])[::-1]
        doc_scores.append(result[i, :][sorted_result].tolist()[:topk])
        doc_indices.append(sorted_result.tolist()[:topk])
    ## get relevant doc bulk

    with open(context_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    total = []
    for idx, example in enumerate(tqdm(dataset, desc="Dense retrieval: ")):
        context_array = []
        for pid in doc_indices[idx]:
            context = "".join(contexts[pid])
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


def retrieval_acc(df, topk):
    df["correct"] = False
    df["correct_rank"] = 0
    for i in tqdm(range(len(df)), desc="check tok_n"):
        count = 0
        for context in df.iloc[i]["context"]:
            count += 1
            if df.iloc[i]["original_context"] == context:
                df.at[i, "correct"] = True
                df.at[i, "correct_rank"] = count

    accuracy = (df["correct"].sum() / len(df),)

    return accuracy
