from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from vectorizer import Vectorizer


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
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(doc_text)
    return chunks



def main():
    source_path = "./english_alternative_rock_groups"
    docs = get_docs(source_path)
    
    vectorizer = Vectorizer("nickprock/sentence-bert-base-italian-uncased")

    for doc in docs:
        chunks = doc2chunks(doc["text"])
        embeddings = vectorizer.get_embeddings(chunks)
        print(doc["title"])
                  

if __name__ == "__main__":
    main()

