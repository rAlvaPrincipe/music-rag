# 🎵 Music-RAG: Music Q&A with LLMs  

Music-RAG is a **Retrieval-Augmented Generation (RAG)** system leveraging the power of **Large Language Models (LLMs)** to answer music-related questions.  
Ever wondered:  
- *Who influenced Muse?*  
- *Who has collaborated with Daft Punk?*  
- *Which band is based in Seattle?*  


## 📄 Requirements
-  Python (tested on 3.11)
-  Docker
-  Docker Compose


## ⚙️ Set Up Your Environment  

Run the following commands to set up a Python virtual environment:  

```bash
$ python3 -m venv venv             # Create virtual environment  
$ source venv/bin/activate         # Activate the virtual environment  
$ pip3 install -r requirements.txt # Install dependencies  
```

Set up ElasticsSearch and Kibana containers. The Kibana GUI is optional:

```bash
$ docker-compose up 
```
Kibana will be available at http://localhost:5601/app/home#/.

## 📚  Fetch Documents from Wikipedia

To retrieve the dataset:
```bash
$ python ./src/corpus_fetcher.py  
```

## 🗄️ Index Data on Elasticsearch

Index the documents into Elasticsearch for efficient retrieval:

```bash
$ python ./src/indexing.py --text_sim <bm25|tfidf> --name_prefix <prefix> --data_source <path> --embedders <emb1 emb2 ...>
```

### Parameters

- **`--text_sim`** (required):  
  Select the similarity metric for text-based search:  
  - `bm25`: Use BM25.  
  - `tfidf`: Use a custom TF-IDF with a scripted implementation.

- **`--name_prefix`** (required):  
  Specify a prefix for the Elasticsearch index name.

- **`--data_source`** (required):  
  Path to the directory containing documents to be indexed.

- **`--embedders`** (optional):  
  List of embedder names for dense vector indexing. Use space as a separator.

- **`--chunk_size`** (optional, default: `500`):  
  Size of text chunks for indexing.

- **`--chunk_overlap`** (optional, default: `50`):  
  Number of overlapping characters between consecutive chunks.

Indexing configurations will be saved in the ./indexes folder.


## 🔎 RAG Inference

Run the RAG system:
```bash
$ python ./src/rag.py --index_name <index_name> --question <question> --embedder <embedder_model> --retrieval_mode <dense|hybrid> --include_metadata <yes|no>
```

### Parameters

- **`--index_name`** (required):  
  The name of the Elasticsearch index to use for retrieval.

- **`--question`** (required):  
  The music-related question you want to ask.

- **`--embedder`** (required):  
  Specifies the embedding model for encoding the question and retrieving relevant chunks. It must match one of the models used during corpus vectorization.

- **`--retrieval_mode`** (required):  
  The retrieval strategy to use:  
  - `dense`: Retrieval using dense embeddings only.  
  - `hybrid`: Combines dense embeddings and named entity recognition (NER) for retrieval.

- **`--include_metadata`** (required):  
  Specify `yes` to include metadata (e.g., `{score: 9.47, source_title: Radiohead, text: ...}`) with each chunk presented to the LLM, or `no` to provide only the plain chunk text.
