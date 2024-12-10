# 🎵 Music-RAG: Answering Music Questions with LLMs  

Music-RAG is a **Retrieval-Augmented Generation (RAG)** system leveraging the power of **Large Language Models (LLMs)** to answer music-related questions.  
Ever wondered:  
- *Who influenced Muse?*  
- *Who has collaborated with Daft Punk?*  
- *Which band is based in Seattle?*  


##  Requirements
-  Python (tested on 3.11)
-  Docker
-  Docker Compose


##  Set Up Your Environment  

Run the following commands to set up a Python virtual environment:  

```bash
$ python3 -m venv venv             # Create virtual environment  
$ source venv/bin/activate         # Activate the virtual environment  
$ pip3 install -r requirements.txt # Install dependencies  
```

Set up ElasticsSearch and Kibana containers. The Kibana GUI is optional:

```bash
% docker-compose up 
```

##  Fetch Documents from Wikipedia

To retrieve the dataset:
```bash
$ python ./src/dataset.py  
```

## Index Data on Elasticsearch

Index the documents into Elasticsearch for efficient retrieval:
```bash
$ python ./src/indexing.py  
```