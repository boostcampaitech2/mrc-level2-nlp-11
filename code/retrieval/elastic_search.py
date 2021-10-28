from elasticsearch import Elasticsearch
import json
from tqdm import tqdm
import time
from subprocess import Popen, PIPE, STDOUT
import os
import pandas as pd

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
        self.index_name = "wiki"
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
                            preexec_fn=os.setuid(1)  # as daemon
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
        index_config = {
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
        return index_config

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
    
    def get_top_k_passages(self, question:str, k:int) -> list:
        query = {
                'query': {
                        'match': {
                            'context': question
                         }
                    }
                }
        result = self.es.search(index=self.index_name, body=query, size=k)
        dataset = [{"question": question, 
                    "title": doc["_source"]["title"],
                    "context": doc["_source"]["context"],
                    "score": doc["_score"]
                    } for doc in result["hits"]["hits"]]
        return dataset

if __name__ == "__main__":
    retriever = ElasticSearch()
    res = retriever.get_top_k_passages("미국의 수도는?", 5)
    print(res)