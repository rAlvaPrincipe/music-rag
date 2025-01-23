from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os 
import json


with open(os.path.join("apikeys.json"), "r", encoding="utf-8") as f:
    keys = json.load(f) 

os.environ["GROQ_API_KEY"] = keys["groq"]
os.environ["OPENAI_API_KEY"] = keys["openai"]


def get_llm(provider, model):
    if provider == "groq":
        return  ChatGroq( model=model, temperature=0)
    elif provider == "openai":
        return ChatOpenAI( model=model, temperature=0)