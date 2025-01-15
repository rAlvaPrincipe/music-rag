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
from ner import NER
from dataset import get_dataset
import sys

class Rag():


    with open(os.path.join("apikeys.json"), "r", encoding="utf-8") as f:
        keys = json.load(f) 

    os.environ["GROQ_API_KEY"] = keys["groq"]


    def __init__(self, conf):
        self.conf = conf
        self.embedder = conf["embedder"]
        self.mode = conf["retrieval_mode"]
        self.include_metadata = conf["include_metadata"]
        
        with open("indexes/" + conf["index_name"] + "/conf.json" , 'r', encoding='utf-8') as file:
            conf_indexing = json.load(file) 
        
        self.ner = NER()
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
        dataset = get_dataset(self.conf["dataset"])
        for item in dataset:
            question = item["q"]
            gt = item["a"]
            
            answer, full_prompt, context = self.run_inference(question)        
            print(answer)
            
        


    def run_inference(self, question):
        question_embedding = self.vectorizer.get_embeddings(question)
        if self.mode == "dense":
            context = self.es.get_rag_contex_only_embeddings(question_embedding, self.conf["embedder"], self.include_metadata)
        else:
            entities = self.ner.get_entities(question)
            context = self.es.get_rag_contex(question_embedding, self.conf["embedder"], entities, self.include_metadata)
    
        chain = self.template | self.groq_model
        answer = chain.invoke({"context": context, "question": question})
        full_prompt = self.template.format(question=question, context=context)
        
        return answer.content, full_prompt, context





def main():
    args = parse()
    conf = build_conf(args)
    
    rag = Rag(conf)
    if conf["mode"] == "inference":
        question = conf["question"]
        answer, full_prompt, context = rag.run_inference(question)
        
        for el in context:
            if isinstance(el, str):
                print(el+ "\n----------------------------------------- \n")
            else:
                print("score: " + str(el["score"]) + "  " + "page: " + el["source_title"])
                print(el["text"]+ "\n")    
        print("\n" + answer)
    else:
        rag.run_validation()
        



if __name__ == "__main__":
    main()


# TODO
#- il prompt template non mi sembra ottimale: il contesto lo spara come un array e avolte non si capisce la fine e l'inizio di uno.