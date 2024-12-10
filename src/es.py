from elasticsearch import Elasticsearch
from datetime import datetime
from pprint import pprint
class ES:
    
    def __init__(self):
        self.es = Elasticsearch("http://localhost:9200")
        self.INDEX = "wiki_chunks"
        
        if not self.es.indices.exists(index=self.INDEX):
            mappings = {
                "properties": {
                    "source_id": {"type": "text", "analyzer": "standard"},
                    "source_title": {"type": "text", "analyzer": "standard"},
                    "source_url": {"type": "text", "analyzer": "standard"},
                    "text": {"type": "text", "analyzer": "english"},
                    "embedding": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"}
                    }
                }
            self.create_index(mappings, self.INDEX)


    def create_index(self, mappings, index_name):
        self.es.indices.create(index=index_name, mappings=mappings)
        
    
    def insert(self, source_doc_id, source_doc_title, source_url, chunk_text,  embedding):  
        doc = dict()
        doc["source_id"] = source_doc_id
        doc["source_title"] = source_doc_title
        doc["source_url"] = source_url
        doc["text"] = chunk_text
        doc["embedding"] = embedding
        resp = self.es.index(index=self.INDEX, document=doc)
    