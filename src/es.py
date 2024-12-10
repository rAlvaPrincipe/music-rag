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
                    "page_id": {"type": "text", "analyzer": "standard"},
                    "page": {"type": "text", "analyzer": "standard"},
                    "text": {"type": "text", "analyzer": "english"},
                    "embedding": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"}
                    }
                }
            self.create_index(mappings, self.INDEX)


    def create_index(self, mappings, index_name):
        self.es.indices.create(index=index_name, mappings=mappings)
        
    