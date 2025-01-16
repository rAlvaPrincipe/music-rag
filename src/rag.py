import json 
from langchain_core.prompts import ChatPromptTemplate
from vectorizer import Vectorizer
from es import ES
from langchain_core.prompts import ChatPromptTemplate
from conf_rag import parse, build_conf
from ner import NER
from metrics import validate
from dataset import get_dataset
import llms

class Rag():

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
        self.llm = llms.get_llm("groq", "llama-3.1-70b-versatile")

        self.template = ChatPromptTemplate([
            ('system',
            "You are a highly knowledgeable assistant tasked with answering questions based on the provided context. Use only the provided text snippets and do not include information from outside sources. Keep your answer concise and directly address the question in two to three sentences." +
            "If the provided context does not contain enough information to answer the question, respond with: 'The provided context does not contain sufficient information to answer this question.'"),
            ('human', 'Question: {question} Context: {context} Answer:')
        ])




    def validation(self):
        dataset = get_dataset(self.conf["dataset"])
        
        questions, contexts, answers, ground_truths = [], [], [], []
        for item in dataset:
            question = item["q"]
            gt = item["a"]
            answer, full_prompt, context = self.inference(question)     
   
            questions.append(question)
            ground_truths.append(gt)
            contexts.append(context)
            answers.append(answer)
            print(answer)
            
        metrics = validate(questions, contexts, answers, ground_truths)
        print(metrics)
            
        


    def inference(self, question):
        question_embedding = self.vectorizer.get_embeddings(question)
        if self.mode == "dense":
            context = self.es.get_rag_contex_only_embeddings(question_embedding, self.conf["embedder"], self.include_metadata)
        else:
            entities = self.ner.get_entities(question)
            context = self.es.get_rag_contex(question_embedding, self.conf["embedder"], entities, self.include_metadata)
    
        chain = self.template | self.llm
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
        rag.validation()
        



if __name__ == "__main__":
    main()


# TODO
#- il prompt template non mi sembra ottimale: il contesto lo spara come un array e avolte non si capisce la fine e l'inizio di uno.

