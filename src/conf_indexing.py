import argparse
from pathlib import Path
import json
import os
import hashlib
import time
from datetime import datetime

SCRIPTED_TFIDF = {
    "type": "scripted",
    "weight_script": {
        "source": "double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; return query.boost * idf;"
    },
    "script": {
        "source": "double tf = Math.sqrt(doc.freq); double norm = 1/Math.sqrt(doc.length); return weight * tf * norm;"
    }
}

BASE_MAPPINGS = {
    "dynamic": "strict",  # Disallow indexing of undefined fields
    "properties": {
        "source_id": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
        "source_title": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
        "source_url": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
    }
}
    

def build_settings_mappings(conf):
    settings = {}
    mappings = {**BASE_MAPPINGS}  # Make a copy of base mappings

    text_sim = conf["text_sim"]
    if text_sim == "bm25":
        mappings["properties"]["text"] = {"type": "text", "analyzer": "english", "similarity": "BM25"}
    elif text_sim == "tfidf":
        settings["similarity"] = {"scripted_tfidf": SCRIPTED_TFIDF}
        mappings["properties"]["text"] = {"type": "text", "analyzer": "english", "similarity": "scripted_tfidf"}

    dense_embeddings = conf.get("dense_embeddings", [])
    for embedding in dense_embeddings:
        embedder = embedding.get("embedder")
        if embedder:
            mappings["properties"][embedder] = {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"}

    return settings, mappings



    


def parse():
    parser = argparse.ArgumentParser(description="Indexing", allow_abbrev=False)
    parser.add_argument('--text_sim', help='bm25 or tfidf')
    parser.add_argument('--index_name', help='index name')
    args = parser.parse_args()
    if not args.text_sim  :
        parser.error('please specify the similarity scoring')
    return args


def build_conf(args):
    conf = personalize(args)
    save(conf)
    return conf


def personalize(args):
    conf = {}
    conf["text_sim"] = args.text_sim
    conf["index_name"] = args.index_name
    conf["time"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    conf["version"] = hashlib.sha256(str(time.time()).encode()).hexdigest()[:4]
    
    settings, mappings = build_settings_mappings(conf)
    conf["settings"] = settings
    conf["mappings"]  = mappings

    output_dir = "indexes/" + conf["index_name"].replace("/","-")  + "/"+ conf["text_sim"] + "__" +  conf["version"]
    conf["output_dir"] = output_dir
    print(conf["output_dir"])
    return conf


def save(conf):
    f_out = conf["output_dir"] + "/" + "conf.json"
    Path(os.path.dirname(f_out)).mkdir(parents=True, exist_ok=True)
    with open(f_out, 'w') as fp:
        json.dump(conf, fp, indent=4)



