# Open Domain Question Answering
A solution for MRC Competition in the 2nd BoostCamp AI Tech by team AI-ESG 

## Content

## Background - MRC Tasks
Open Domain Question Answering (ODQA) is a task to find an exact answer to any question in Wikipedia articles. Thus, given only a question, the system outputs the best answer it can find. The default ODQA implementation takes a batch of queries as input and returns the best answer.



**This is an example:**

**Question**: 서울의 GDP는 세계 몇 위인가?
**Answer**: 세계 4위





## Project Outline


## Team

### Members of Team AI-ESG
| Name | github | contact |
| -------- | -------- | -------- |
| 문석암     | [Link](https://github.com/mon823) | mon823@naver.com |
| 박마루찬 | [Link](https://github.com/MaruchanPark) | shaild098@naver.com |
| 박아멘 | [Link](https://github.com/AmenPark) | puzzlistpam@gmail.com |
| 우원진 | [Link](https://github.com/woowonjin) | dndnjswls613@naver.com |
| 윤영훈 | [Link](https://github.com/wodlxosxos) | wodlxosxos73@gmail.com |
| 장동건 | [Link](https://github.com/mycogno) | jdg4661@gmail.com |
| 홍현승 | [Link](https://github.com/Hong-Hyun-Seung) | honghyunseung100@gmail.com |

## Structure
```
├── code
│   ├── read
│   │   ├── mrc_reader.py
│   │   ├── train.py
│   │   ├── arguments.py
│   │   ├── utils_qa.py
│   │   ├── utils_qa_save.py
│   │   └── trainers_qa.py
│   ├── reatrieval
│   │   ├── bm25.py
│   │   ├── elastic_search.py
│   │   ├── dense_train.py
│   │   ├── dense_inference.py
│   │   ├── dense_dataset.py
│   │   ├── dense_model.py
│   │   └── func.py
│   ├── ensemble.ipynb
│   ├── inference.py
│   └── arguments.py
└── data
    ├── wikipedia_document.json
    ├── test_dataset
    └── train_dataset
```

* ```read/``` : Train reader model
* ```retrieval/bm25.py``` : Make TF-IDF embedding and get bm25 scores
* ```retrieval/elastic_search.py``` : Build elastic search module
* ```retrieval/dense_*``` : Train, inference dense embedding retrieval


## Getting Started
### Hardware
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

### Dependencies
- datasets==1.5.0
- transformers==4.5.0
- tqdm==4.41.1
- pandas==1.1.4
- scikit-learn==0.24.1
- konlpy==0.5.2
- wandb==0.12.4
- elasticsearch==7.9.2

### Install Requirements
```
pip install -r install/install_requirements.txt
source ./code/retrieval/elastic.sh
```

### Reader

#### Train
```
python train.py --model_name_or_path pre_xlm-roberta-large \
    --train_batch_size 4 \
    --epochs 5 --run_name {wandb run-name} 
```
### Retriever

#### Sparse Embedding (BM25)
```
python3 bm25.py --dataset_name ../../data/train_dataset \
    --model_name_or_path klue/bert-base --context_path wikipedia_documents.json
```

#### Elastic Search
```
python elastic_search.py (ElasticSearch 테스트때, 실제 사용할때는 실행할 필요없음.)
```

#### Dense Embedding
##### Train
in_batch - False(default) : training over negative docs from wiki
in_batch - True : in-batch training

```
python3 dense_train.py --p_enc_name_or_path klue/bert-base \
    --q_enc_name_or_path klue/bert-base \
    --run_name dense_train --tokenizer_name klue/bert-base \
    --save_path_p ./encoder/p_enc --save_path_q ./encoder/q_enc \
    --save_pickle_path ../../data/refac_test --num_neg 3  --in_batch True \
    --dataset_name ../../data/train_dataset \
    --context_path ../../data/preprocess_wikipedia_documents.json \
    --save_epoch 10 --batch_size 2 --num_train_epochs 10 --random_seed 452
```
##### Inference
```
python3 dense_inference.py --load_path_q klue/bert-base \
    --tokenizer_name klue/bert-base \
    --pickle_path ../../data/refac_test.bin \
    --context_path ../../data/preprocess_wikipedia_documents.json \
    --dataset_name ../../data/train_dataset
```
### Inference
```
python inference.py --output_dir ./outputs/test_dataset/ \
    --dataset_name ../data/test_dataset/ \
    --model_name_or_path {모델저장된 path} \
    --retrieval_name elastic --do_predict
```
- 만약 custom모델을 사용해 inference하는 경우 "--model_name custom_{custom모델 모듈 이름}"을 추가


