import json 
from langchain_core.prompts import ChatPromptTemplate
from vectorizer import Vectorizer
from es import ES
from langchain_core.prompts import ChatPromptTemplate
from conf_rag import build_conf_from_args
from ner import NER
from dataset import get_dataset
import llms
from validator import Validator


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
        self.llm = llms.get_llm(conf["llm"]["inference"]["provider"], conf["llm"]["inference"]["model"])
        self.validator = Validator(conf["llm"]["evaluation"]["provider"], conf["llm"]["evaluation"]["model"])

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
            id = item["id"]
            q = item["q"]
            gt= item["a"]
            answer, full_prompt, context = self.inference(q)     
   
            questions.append(q)
            ground_truths.append(gt)
            contexts.append(context)
            answers.append(answer)
            print(answer)
            
            
            self.validator.save_requests_responses(full_prompt, self.conf["output_dir"] + "/logs/" + id, "full_prompt.txt")
            self.validator.save_requests_responses(item, self.conf["output_dir"] + "/logs/" + id, "gt.json")
            self.validator.save_requests_responses(context, self.conf["output_dir"] + "/logs/" + id, "context.json")
            self.validator.save_requests_responses(answer, self.conf["output_dir"] + "/logs/" + id, "answer.txt")

        metrics = self.validator.validate(questions, contexts, answers, ground_truths)
        self.validator.save_metrics(metrics, self.conf["output_dir"] + "/metrics.json")
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
    conf = build_conf_from_args()
    
    rag = Rag(conf)
    if conf["mode"] == "inference":
        question = conf["question"]
        answer, full_prompt, context = rag.inference(question)
        
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

