from sentence_transformers import SentenceTransformer
import torch

# https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
class Vectorizer():    
    AVAILABLE_MODELS = {
        "sbert/all-MiniLM-L6-v2": {
            "label": "sbert",
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "dims": 384,
            "max_input_length": 256
        },
        "sbert/all-mpnet-base-v2": {
            "label": "sbert",
            "id": "sentence-transformers/all-mpnet-base-v2",
            "dims": 768,
            "max_input_length": 384
        }
    }
    
    
    def __init__(self, model):            
        if model in self.AVAILABLE_MODELS.keys():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(device)

            if "sbert" in model:
                self.model = SentenceTransformer(self.AVAILABLE_MODELS[model]["id"], device=device)
        else:
            raise Exception("model not available")
            


    def get_embeddings(self, chunks):
        return self.model.encode(chunks)        

