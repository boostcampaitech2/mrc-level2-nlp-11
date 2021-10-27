from elasticsearch import ElasticSearch
import json
from tqdm import tqdm
import time
from subprocess import Popen, PIPE, STDOUT
import os

class ElasticSearch:
    '''
    처음 실행하거나, Connection Error발생시
    elastic_search 폴더와, tar.gz파일을 삭제후
    elastic.sh 쉘스크립트 실행
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
            return
        self.set_mapping()
        self.insert_data_to_elastic()

    def get_wiki_parsed(self) -> None:
        with open(self.wiki_path, "r", encoding="utf-8") as f:
            wiki_json = json.load(f)
            self.wiki_contexts = [{"id": val["document_id"], "title": val["title"], "context" : val["text"]} for key, val in wiki_json.items()]

    def set_elastic_server(self):
        path_to_elastic = "elasticsearch-7.9.2/bin/elasticsearch"
        es_server = Popen([path_to_elastic],
                            stdout=PIPE, stderr=STDOUT,
                            preexec_fn=os.setuid(1)  # as daemon
                            )
        config = {'host':'localhost', 'port':9200}
        print("You have to wait 10secs for connecting to elastic server")
        time.sleep(10)
        es = Elasticsearch([config])
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

    def set_mapping(self) -> None:
        if self.es.indices.exists(index=self.index_name):
            print("Index Mapping already exists.")
        else:
            self.es.indices.create(index=self.index_name, body=self.index_config, ignore=400)
            print("Index Mapping Created.")

    def insert_data_to_elastic(self) -> None:
        for i, rec in enumerate(tqdm(self.wiki_contexts)):
            try:
                index_status = self.es.index(index=INDEX_NAME, id=i, body=rec)
            except:
                print(f'Unable to load document {i}.')    
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
        #result["hits"]["hits"]