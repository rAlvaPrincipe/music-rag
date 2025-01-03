
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

class Vectorizer():

    #available_models = ["sbert/all-MiniLM-L6-v2"] 
    
    AVAILABLE_MODELS = {
        "sbert/all-MiniLM-L6-v2": {
            "label": "sbert",
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "dims": 384
        },
        "sbert/all-mpnet-base-v2": {
            "label": "sbert",
            "id": "sentence-transformers/all-mpnet-base-v2",
            "dims": 768
        }
    }
    
    
    def __init__(self, model):            
        if model in self.AVAILABLE_MODELS.keys():
            if "sbert" in model:
                self.model = SentenceTransformer(self.AVAILABLE_MODELS[model]["id"])
        else:
            raise Exception("model not available")
            


    def get_embeddings(self, chunks):
        return self.model.encode(chunks)        

