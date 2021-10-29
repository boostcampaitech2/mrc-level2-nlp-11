from elasticsearch import Elasticsearch
import json
from tqdm import tqdm
import time
from subprocess import Popen, PIPE, STDOUT
import os
import pandas as pd
from datasets import Dataset, DatasetDict, Value, Features, load_from_disk

class ElasticSearch:
    '''
    처음 실행하거나, Connection Error발생시
    elastic_search 폴더삭제(안해도 되는거 같아요.), 서버 껐다가
    elastic.sh 쉘스크립트 실행이후 사용
    '''
    def __init__(self):
        self.config : dict= {'host':'localhost', 'port':9200}
        self.wiki_path : str= "/opt/ml/mrc-level2-nlp-11/data/wikipedia_documents.json"
        self.wiki_contexts : list = None
        self.get_wiki_parsed()
        self.index_config : dict = self.get_index_config()
        print(self.index_config)
        self.index_name = "wiki_4"
        self.es = self.set_elastic_server()
        if not self.es:
            exit()
        is_already_exist = self.set_mapping()
        if not is_already_exist:
            self.insert_data_to_elastic()

    def get_wiki_parsed(self) -> None:
        with open(self.wiki_path, "r", encoding="utf-8") as f:
            wiki_json = json.load(f)
            self.wiki_contexts = [{"id": val["document_id"], "title": val["title"], "context" : val["text"]} for key, val in wiki_json.items()]

    def set_elastic_server(self):
        path_to_elastic = "/opt/ml/mrc-level2-nlp-11/code/retrieval/elasticsearch-7.9.2/bin/elasticsearch"
        es_server = Popen([path_to_elastic],
                            stdout=PIPE, stderr=STDOUT,
                            preexec_fn=lambda: os.setuid(1)  # as daemon
                            )
        config = {'host':'localhost', 'port':9200}
        print("You have to wait 20secs for connecting to elastic server")
        for _ in tqdm(range(20)):
            time.sleep(1)
        es = Elasticsearch([config], timeout=30)
        ping_result = es.ping()
        if ping_result:
            print("Connecting Success !!")
            return es
        else:
            print("Connecting Failed.. You have to try again. You have to follow class description")
            return None
    
    def get_index_config(self) -> str:
        index_config_1 = {
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "nori_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "nori_tokenizer",
                                    "decompound_mode": "mixed",
                                }
                            }
                        }
                    },
                    "mappings": {
                        "dynamic": "strict", 
                        "properties": {
                            "id": {"type": "int"},
                            "title": {"type": "text", "analyzer":"nori_analyzer"},
                            "context": {"type": "text", "analyzer":"nori_analyzer"},
                            }
                        }
                    }
        index_config_2 = {
                'settings':{
                    'analysis':{
                        'analyzer':{
                          'my_analyzer':{
                              "type": "custom",
                              'tokenizer':'nori_tokenizer',
                              'decompound_mode':'mixed',
                              'stopwords':'_korean_',
                              'synonyms':'_korean_',
                              "filter": ["lowercase",
                                         "my_shingle_f",
                                         "nori_readingform",
                                         "nori_number",
                                         "cjk_bigram",
                                         "decimal_digit",
                                         "stemmer",
                                         "trim"]
                          }
                      },
                      'filter':{
                          'my_shingle_f':{
                              "type": "shingle"
                          }
                      }
                  },
                  'similarity':{
                      'my_similarity':{
                          'type':'BM25',
                      }
                  }
              },
            'mappings':{
                  'properties':{
                      'title':{
                          'type':'text',
                          'analyzer':'my_analyzer',
                          'similarity':'my_similarity'
                      },
                      'context':{
                          'type':'text',
                          'analyzer':'my_analyzer',
                          'similarity':'my_similarity'
                      },
                        "id": {"type": "int"},
                  }
              }
        }

        index_config_3 = {
        "settings": {
            "analysis": {
                "filter":{
                    "my_stop_filter": {
                        "type" : "stop",
                        "stopwords_path" : "my_stop_dic.txt"
                    }
                },
                "analyzer": {
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "filter" : ["my_stop_filter"]
                    }
                }
            }
        },
        "mappings": {
            "dynamic": "strict", 
            "properties": {
                    'title':{
                          'type':'text',
                          'analyzer':'nori_analyzer',
                    },
                    'context':{
                        'type':'text',
                        'analyzer':'nori_analyzer',
                    },
                      "id": {"type": "int"},
                }
            }
        }

        return index_config_3

    def set_mapping(self) -> bool:
        if self.es.indices.exists(index=self.index_name):
            print("Index Mapping already exists.")
            return True
            # self.es.indices.delete(index=self.index_name, ignore=[400, 404])
            # print("Previous Index was dropped.")
            # self.es.indices.create(index=self.index_name, body=self.index_config, ignore=400)
            # print("Index Mapping Created.")
        else:
            self.es.indices.create(index=self.index_name, body=self.index_config, ignore=400)
            print("Index Mapping Created.")
            return False

    def insert_data_to_elastic(self) -> None:
        for i, rec in enumerate(tqdm(self.wiki_contexts)):
            try:
                index_status = self.es.index(index=self.index_name, id=i, body=rec)
            except Exception as e:
                print(f'Unable to load document {i}. because of {e}')   
        n_records = self.es.count(index=self.index_name)['count']
        print(f'Succesfully loaded {n_records} into {self.index_name}')
    
    # def get_top_k_passages(self, question:str, k:int) -> list:
    #     query = {
    #             'query': {
    #                     'match': {
    #                         'context': question
    #                      }
    #                 }
    #             }
    #     result = self.es.search(index=self.index_name, body=query, size=k)
    #     dataset = [{"question": question, 
    #                 "title": doc["_source"]["title"],
    #                 "context": doc["_source"]["context"],
    #                 "score": doc["_score"]
    #                 } for doc in result["hits"]["hits"]]
    #     return dataset

    def get_top_k_passages(self, question:str, k:int) -> list:
        query = {
                'query': {
                        'match': {
                            'context': question
                         }
                    }
                }
        result = self.es.search(index=self.index_name, body=query, size=k)
        return result["hits"]["hits"]

    def run_retrieval(self, q_dataset: Dataset, topk: int) -> DatasetDict:
        results = []
        for idx, q_data in enumerate(q_dataset):
            question = q_data["question"]
            question_id = q_data["id"]
            passages = self.get_top_k_passages(question=question, k=topk)
            # examples = [{"context": passages[i]["_source"]["context"], "id": question_id+f"_{i}", "question": question, "score":passages[i]["_score"]} for i in range(len(passages))]
            context = ""
            for passage in passages:
                context += passage["_source"]["context"] + " "
            example = {"context": context, "id":question_id, "question":question, "score":0} 
            # results.extend(examples)
            results.append(example)
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "score": Value(dtype="float", id=None)
            }
        )
        df = pd.DataFrame(results)
        
        return DatasetDict({"validation": Dataset.from_pandas(df, features=f)})

if __name__ == "__main__":
    retriever = ElasticSearch()
    train_path = "/opt/ml/mrc-level2-nlp-11/data/train_dataset"
    data = load_from_disk(train_path)
    train_data = data["train"]
    match_cnt = 0
    k = 5
    for datum in tqdm(train_data):
        question_text = datum["question"]
        context = datum["context"]
        result = retriever.get_top_k_passages(question_text, k)
        for res in result:
            if res["_source"]["context"] == context:
                match_cnt += 1
                break
    print(f"matching score is {match_cnt/3952:.3f}")