from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from vectorizer import Vectorizer
from es import ES
from tqdm import tqdm
from conf_indexing import parse, build_conf
import hashlib

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


def index_text(es, docs):
    with tqdm(total=len(docs)) as pbar:
        for doc in docs:
            chunks = doc2chunks(doc["text"])
            for chunk in chunks:
                es.insert(doc["page_id"], doc["title"], doc["url"], chunk)
            
            pbar.update(1)


def index_embeddings(es, docs, embedder_model):
    vectorizer = Vectorizer(embedder_model)
    with tqdm(total=len(docs)) as pbar:
        for doc in docs:
            chunks = doc2chunks(doc["text"])
            embeddings = vectorizer.get_embeddings(chunks)
            
            for chunk, embedding in zip(chunks, embeddings):
                id = doc["page_id"] + "_" + hashlib.md5(chunk.encode('utf-8')).hexdigest()
                es.update_embedding(id, embedder_model, embedding)
                pbar.update(1)


def main():
    args = parse()
    conf = build_conf(args)
    es = ES(conf)

    docs = get_docs(conf["data_source"])[:30]
    index_text(es, docs)
    
    embedders = conf["embedders"]
    for embedder in embedders:
        index_embeddings(es, docs, embedder)


if __name__ == "__main__":
    main()

