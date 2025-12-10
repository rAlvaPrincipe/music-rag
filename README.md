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

Set your apykeys in apikeys.json

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

Example:

```bash
$ python ./src/indexing.py --text_sim bm25 --name_prefix ai2025 --data_source ./data/musical_groups_by_genre --embedders sbert/all-mpnet-base-v2
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


## 🔎 Running the RAG System  

The RAG system supports two modes of operation:  
- **Inference**: Ask music-related questions to retrieve relevant information.  
- **Evaluation**: Evaluate the system on a dataset of questions and answers.  

Use the following command:  

```bash
$ python ./src/rag.py --mode <inference|evaluation> --index_name <index_name> --embedder <embedder_model> --retrieval_mode <dense|hybrid> --include_metadata <yes|no> --inf_llm_provider <llm_provider> --inf_llm_model <llm_model> [--question <question>] [--dataset <dataset_path>] [--eval_llm_provider <eval_llm_provider>] [--eval_llm_model <eval_llm_model>]
```

Example:
```bash
python ./src/rag.py --mode inference  --index_name ai2025_2e79 --embedder sbert/all-mpnet-base-v2 --retrieval_mode dense --include_metadata yes --inf_llm_provider openai --inf_llm_model gpt-4o-mini --question "Did Pink Floyd influence Radiohead?"
```

### Parameters  

- `--mode` (required):  
  Specify the mode of operation:  
  - `inference`: To ask a question.  
  - `evaluation`: To evaluate the system on a dataset.

- `--index_name` (required):  
  The name of the Elasticsearch index to use for retrieval.

- `--embedder` (required):  
  Specifies the embedding model used for encoding.

- `--retrieval_mode` (required):  
  The retrieval strategy to use:  
  - `dense`: Retrieval using dense embeddings only.  
  - `hybrid`: Combines dense embeddings and named entity recognition (NER) for retrieval.

- `--include_metadata` (required):  
  Include metadata with chunks:  
  - `yes`: Metadata is included (e.g., `{score: 9.47, source_title: Radiohead, text: ...}`).  
  - `no`: Only plain text chunks are provided.

- `--inf_llm_provider` (required):  
  Specify the LLM service provider to use at inference time, such as `groq`, `aws`, `openai`.

- `--inf_llm_model` (required):  
  Specify the LLM model to use at inference time, such as: `llama-3.1-70b-versatile`, `gpt-4-turbo`.

#### Mode-Specific Parameters  

- For `inference` mode:  
  - `--question` (required):  
    The music-related question you want to ask.

- For `evaluation` mode:  
  - `--dataset` (required):  
    The dataset name containing questions and answers for evaluation.

  - `--eval_llm_provider` (required):  
  Specify the LLM service provider at evaluation time, such as `groq`, `aws`, `openai`.

  - `--eval_llm_model` (required):  
  Specify the LLM model to use at evaluation time, such as: `llama-3.1-70b-versatile`, `gpt-4-turbo`.