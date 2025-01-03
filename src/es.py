from elasticsearch import Elasticsearch
         
class ES:

    
    def __init__(self, conf):
        self.es = Elasticsearch("http://localhost:9200")
        self.INDEX = conf["index_name"]        
        self.conf = conf

        if not self.es.indices.exists(index=self.INDEX):
            self.create_index()


    def create_index(self):
        self.es.indices.create(index=self.INDEX, mappings=self.conf["mappings"], settings=self.conf["settings"])
        
    
    
    def insert(self, source_doc_id, source_doc_title, source_url, chunk_text,  embedding):  
        doc = dict()
        doc["source_id"] = source_doc_id
        doc["source_title"] = source_doc_title
        doc["source_url"] = source_url
        doc["text"] = chunk_text
        doc["embedding"] = embedding
        resp = self.es.index(index=self.INDEX, document=doc)
    
    
    
    
    def get_rag_contex_only_embeddings(self, question_vector):
        query = {
            "knn": {
            "field": "embedding",
            "inner_hits": {
                "_source": False,
                "fields": ["text"]
            },
            "query_vector": question_vector,
            "k": 5,
            }
        }

        resp = self.es.search(index=self.INDEX, body=query)
    
    
    
    def get_rag_contex(self, question_vector):
        fields = ["source_title", "text"]

        es_query = {
            "size": 15,
            "knn": {  # k-NN is a top-level query
                "field": "embedding",
                "query_vector": question_vector,
                "k": 15,
                "num_candidates": 200
            },
            "query": {  # Add boosting with a bool query
                "bool": {
                    "should": [
                        {
                            "match": {
                                "text": {
                                    "query": "Pink Floyd",
                                    "fuzziness": "AUTO",
                                    "boost": 1.3
                                }
                            }
                        }
                    ]
                }
            }
        }

        

        results = self.es.search(index=self.INDEX, body=es_query, _source_includes=fields)
        results = results["hits"]["hits"]  
        
        outs = list()
        for result in results:
            out = { "score": result["_score"]}
            for field in fields:
                out[field] = result["_source"][field]
            outs.append(out)
        return outs          


# da provare: filtra prima quelli che contengon pink floyd, su questo sottoinsieme fai la knn
#             cerca solo nella pagina di pink floyd