from langchain_groq import ChatGroq
import os 
import json


with open(os.path.join("apikeys.json"), "r", encoding="utf-8") as f:
    keys = json.load(f) 

os.environ["GROQ_API_KEY"] = keys["groq"]



def get_llm(provider, model):
    if provider == "groq":
        return  ChatGroq( model=model, temperature=0)