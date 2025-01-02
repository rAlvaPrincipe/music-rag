import os
import json 
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
#from langchain_community.document_transformers import LongContextReorder
from vectorizer import Vectorizer
from es import ES
from langchain_core.prompts import ChatPromptTemplate

with open(os.path.join("apikeys.json"), "r", encoding="utf-8") as f:
    keys = json.load(f) 

os.environ["GROQ_API_KEY"] = keys["groq"]


vectorizer  = Vectorizer("sentence-transformers/all-MiniLM-L6-v2")
es = ES()

groq_model = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
)

template = ChatPromptTemplate([
    ('system',
     "You are a highly knowledgeable assistant tasked with answering questions based on the provided context. Use only the provided text snippets and do not include information from outside sources. Keep your answer concise and directly address the question in two to three sentences." +
     "If the provided context does not contain enough information to answer the question, respond with: 'The provided context does not contain sufficient information to answer this question.'"),
    ('human', 'Question: {question} Context: {context} Answer:')
])




def main():
    #question = "Who were influenced by Pink Floyd?"
    question = "Who influenced Pink Floyd?"

    question_embedding = vectorizer.get_embeddings(question)
    context = es.get_rag_contex(question_embedding)
    chain = template | groq_model
    answer = chain.invoke({"context": context, "question": question})

    full_prompt = template.format(question=question, context=context)
    print(full_prompt)
    print(answer.content)


if __name__ == "__main__":
    main()

