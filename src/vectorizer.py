
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch


class Vectorizer():

    available_models = ["nickprock/sentence-bert-base-italian-uncased"] 

    def __init__(self, model):
        self.embedders = dict()
            
        if model in self.available_models:
            self.tokenizer = AutoTokenizer.from_pretrained(model)     
            self.embedder = AutoModel.from_pretrained(model, output_hidden_states = True).to("cuda")
        else:
            raise Exception("model not available")
            


    def get_embeddings(self, chunks, is_pooled_output=False):
        encodings = self.tokenizer(chunks, truncation=True, padding=True, max_length=512, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.embedder(input_ids=encodings["input_ids"].to("cuda"), attention_mask=encodings["attention_mask"].to("cuda"))
            if is_pooled_output:
                return outputs.pooler_output
            else:
                return  outputs.last_hidden_state[:, 0, :] 
                


