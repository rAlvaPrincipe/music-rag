import os
import json 
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
#from langchain_community.document_transformers import LongContextReorder
from vectorizer import Vectorizer
from es import ES
from langchain_core.prompts import ChatPromptTemplate
from conf_rag import parse, build_conf
import sys

class Rag():


    with open(os.path.join("apikeys.json"), "r", encoding="utf-8") as f:
        keys = json.load(f) 

    os.environ["GROQ_API_KEY"] = keys["groq"]


    def __init__(self, conf):
        self.conf = conf
        self.embedder = conf["embedder"]
        self.question = conf["question"]
        
        with open("indexes/" + conf["index_name"] + "/conf.json" , 'r', encoding='utf-8') as file:
            conf_indexing = json.load(file) 
        
        self.es = ES(conf_indexing)
        self.vectorizer  = Vectorizer(self.embedder)
        self.groq_model = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
        )

        self.template = ChatPromptTemplate([
            ('system',
            "You are a highly knowledgeable assistant tasked with answering questions based on the provided context. Use only the provided text snippets and do not include information from outside sources. Keep your answer concise and directly address the question in two to three sentences." +
            "If the provided context does not contain enough information to answer the question, respond with: 'The provided context does not contain sufficient information to answer this question.'"),
            ('human', 'Question: {question} Context: {context} Answer:')
        ])


    def run_validation(self):
        3
        

    def run_inference(self):
        #question = "Who was influenced by Pink Floyd?"
        #question = "Who influenced Pink Floyd?"
        question_embedding = self.vectorizer.get_embeddings(self.question)
        context = self.es.get_rag_contex(question_embedding, self.conf["embedder"])
        chain = self.template | self.groq_model
        answer = chain.invoke({"context": context, "question": self.question})

        full_prompt = self.template.format(question=self.question, context=context)
        print(full_prompt)
        print(answer.content)




def main():
    args = parse()
    conf = build_conf(args)
        
    rag = Rag(conf)
    rag.run_inference()
        
    



if __name__ == "__main__":
    main()

