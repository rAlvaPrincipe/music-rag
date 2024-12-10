
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

class Vectorizer():

    available_models = ["sentence-transformers/all-MiniLM-L6-v2"] 

    def __init__(self, model):            
        if model in self.available_models:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            raise Exception("model not available")
            


    def get_embeddings(self, chunks):
        return self.model.encode(chunks)        

