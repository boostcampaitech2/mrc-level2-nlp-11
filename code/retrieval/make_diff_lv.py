import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from retrieval import *
import argparse

def main_fun(args):
    org_dataset = load_from_disk(args.train_dataset)
    dt = org_dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,       
        use_fast=False,
    )   #저장된 토크나이저를 로드할 수 있도록 수정 필요

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.datapath,
        context_path=args.wiki,
    )

    retriever.get_sparse_embedding()
    retriever.build_faiss(64)

    '''
    윗부분을 수정시 충분히 다른 모델에 대해서도 오답노트 생성 가능.
    지금은 sparse 버전으로 만들었음.
    피클 파일은 id-question(문장 전체)-doc_id_list(오답으로 나오기 쉬운 문서 id)-ans_id 순.
    doc_id_list에서 정답 id와 title이 완전히 같은 것은 제외.
    만약 그냥 생짜를 원한다면 위키를 불러올 필요도 없고, title을 비교할 필요도 없음.
    다만 -1이 나오는 경우 때문에 docs_out에서 j>0조건은 그대로 두는 것이 좋음.
    '''


    print("get wiki data")
    with open("../../data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)


    print("====ready=====")


    listform = []
    ct = 0
    N = len(dt)
    minvar = 100
    for i in range(N):
        if i%100 == 0:
            print(f'==={i}/{N} 진행중===')
        query = dt['question'][i]
        ans_id = dt['document_id'][i]
        title = dt['title'][i]
        _, docs = retriever.get_relevant_doc_faiss(query,100)
        docs_out = [j for j in docs if j > 0 and wiki[str(j)]['title'] != title]
        minvar = min(minvar,len(docs_out))
        listform.append([i,query,docs_out,ans_id])


    data_to_save = pd.DataFrame(listform,columns = ['id','question','docs_id','ans_id'])
    data_to_save.to_pickle(args.pkl_save_path)
    print(f'유사 답안 최소 길이는 {minvar}입니다.')
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--datapath", default="../../data", type=str, help=""
    )
    parser.add_argument(
        "--wiki", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument(
        "--tokenizer", default="bert-base-multilingual-cased", type=str, help=""
    )     #저장된 토크나이저 사용하기를 위해서는 수정 필요
    
    parser.add_argument(
        "--train_dataset", default="../../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--pkl_save_path", default="../../data/difficult_sparse_docs.pkl", type=str, help=""
    )
    args = parser.parse_args()
    main_fun(args)