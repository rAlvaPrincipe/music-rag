from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from vectorizer import Vectorizer
from es import ES
from tqdm import tqdm
from conf_indexing import parse, build_conf


def get_docs(source_path):
    docs = []
    for filename in os.listdir(source_path):
        file_path = os.path.join(source_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f) 
            docs.append(data)  
    return docs


def doc2chunks(doc_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(doc_text)
    return chunks



def main():
    args = parse()
    conf = build_conf(args)
    
    source_path = "./data/english_alternative_rock_groups"
    docs = get_docs(source_path)
    
    vectorizer = Vectorizer("sentence-transformers/all-MiniLM-L6-v2")
    es = ES(conf)
    
    #TODO gestisci zero o molteplici vectorizers
    
    with tqdm(total=len(docs)) as pbar:
        for doc in docs:
            chunks = doc2chunks(doc["text"])
            embeddings = vectorizer.get_embeddings(chunks)
            for chunk, embedding in zip(chunks, embeddings):
                es.insert(doc["page_id"], doc["title"], doc["url"], chunk, embedding)
            
            pbar.update(1)

if __name__ == "__main__":
    main()

