from elasticsearch import Elasticsearch
import hashlib 
         
class ES:

    
    def __init__(self, conf):
        self.es = Elasticsearch("http://localhost:9200", timeout= 600)
        self.INDEX = conf["index_name"]        
        self.conf = conf
        

    def create_index(self):
        self.es.indices.create(index=self.INDEX, mappings=self.conf["es_conf"]["mappings"], settings=self.conf["es_conf"]["settings"])
        
    
    def insert(self, source_doc_id, source_doc_title, source_url, chunk_text):  
        doc = dict()
        id = source_doc_id + "_" + hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
        doc["source_id"] = source_doc_id
        doc["source_title"] = source_doc_title
        doc["source_url"] = source_url
        doc["text"] = chunk_text

        resp = self.es.index(index=self.INDEX, id=id, document=doc)
    
    
    def update_embedding(self, id, embedding_field, embedding):
        update_query = {
            "doc": { embedding_field: embedding }
        }
        self.es.update(index=self.INDEX, id=id, body=update_query)

    
    def get_rag_contex_only_embeddings(self, question_vector,embedder_name,include_metadata):
        fields = ["source_title", "text"]
        
        query = {
            "knn": {
                "field": embedder_name,
                "query_vector": question_vector,
                "k": 15,
                "num_candidates": 200
            }
        }
        results = self.es.search(index=self.INDEX, body=query, _source_includes=fields)
        return self.format_output(results, fields, include_metadata) 
    
    
    def get_rag_contex(self, question_vector, embedder_name, entities, include_metadata):
        fields = ["source_title", "text"]

        query = {
            "size": 15,
            "knn": {  # k-NN is a top-level query
                "field": embedder_name,
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
                                    "query": entities[0],
                                    "fuzziness": "AUTO",
                                    "boost": 1.3
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        results = self.es.search(index=self.INDEX, body=query, _source_includes=fields)
        return self.format_output(results, fields, include_metadata)


    def format_output(self, results, fields, include_metadata):
        results = results["hits"]["hits"]  
        
        outs = list()
        for result in results:
            if include_metadata:
                out = { "score": result["_score"]}
                for field in fields:
                    out[field] = result["_source"][field]
            else:
                out = result["_source"]["text"]
            outs.append(out)
        return outs         


# TODO:
#- in alcuni casi vale la pena filtrare prima sulla pagina del musicista e poi fare la ricerca dentro
#- quatno pesare titolo/entità/embedding?
# da provare: filtra prima quelli che contengon pink floyd, su questo sottoinsieme fai la knn
#             cerca solo nella pagina di pink floyd
