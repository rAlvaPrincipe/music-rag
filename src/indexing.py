from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from vectorizer import Vectorizer
from es import ES
from tqdm import tqdm
from conf_indexing import parse, build_conf
import hashlib


class Indexer:
    def __init__(self, es, conf):
        self.es = es
        self.chunk_size = conf["chunk_size"]
        self.chunk_overlap = conf["chunk_overlap"]
        self.docs = self.get_docs(conf["data_source"])


    def get_docs(self, source_path):
        docs = []
        for filename in os.listdir(source_path):
            file_path = os.path.join(source_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f) 
                docs.append(data)  
        return docs


    def doc2chunks(self, doc_text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_text(doc_text)
        return chunks


    def index_text(self):
        with tqdm(total=len(self.docs)) as pbar:
            for doc in self.docs:
                chunks = self.doc2chunks(doc["text"])
                for chunk in chunks:
                    self.es.insert(doc["page_id"], doc["title"], doc["url"], chunk)
                
                pbar.update(1)


    def index_embeddings(self, embedder_model):
        vectorizer = Vectorizer(embedder_model)
        with tqdm(total=len(self.docs)) as pbar:
            for doc in self.docs:
                chunks = self.doc2chunks(doc["text"])
                embeddings = vectorizer.get_embeddings(chunks)
                
                for chunk, embedding in zip(chunks, embeddings):
                    id = doc["page_id"] + "_" + hashlib.md5(chunk.encode('utf-8')).hexdigest()
                    self.es.update_embedding(id, embedder_model, embedding)
                
                pbar.update(1)


def main():
    args = parse()
    conf = build_conf(args)
    es = ES(conf)
    es.create_index()

    indexer = Indexer(es, conf)
    indexer.index_text()
    
    embedders = conf["embedders"]
    for embedder in embedders:
        indexer.index_embeddings(embedder)


if __name__ == "__main__":
    main()

