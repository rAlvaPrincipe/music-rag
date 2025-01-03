import argparse
from pathlib import Path
import json
import os
import hashlib
import time
from datetime import datetime
from vectorizer import Vectorizer


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

    embedders = conf.get("embedders", [])
    for embedder in embedders:
        dims = Vectorizer.AVAILABLE_MODELS[embedder]["dims"]
        mappings["properties"][embedder] = {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"}

    return settings, mappings



    


def parse():
    parser = argparse.ArgumentParser(description="Indexing", allow_abbrev=False)
    parser.add_argument('--text_sim', help='bm25 or tfidf')
    parser.add_argument('--name_prefix', help='prefix for the index name')
    parser.add_argument('--data_source', help='the path of the data source directory')
    parser.add_argument('--embedders', nargs='+', help='the embedders for dense indexing')
    args = parser.parse_args()
    if not args.text_sim or not args.name_prefix or not args.data_source :
        parser.error('please specify all the arguments')
    if args.text_sim not in ["bm25", "tfidf"]:
        parser.error('text_sim must be bm25 or tfidf')
    #if args.embedders not in Vectorizer.AVAILABLE_MODELS.keys():
    #    parser.error('embedders must be chosen from the available ones: ' + str(list(Vectorizer.AVAILABLE_MODELS.keys())))
    if args.embedders:
        invalid_embedders = [e for e in args.embedders if e not in Vectorizer.AVAILABLE_MODELS.keys()]
        if invalid_embedders:
            parser.error('Invalid embedders: ' + ', '.join(invalid_embedders))
    return args


def build_conf(args):
    conf = personalize(args)
    save(conf)
    return conf


def personalize(args):
    conf = {}
    conf["text_sim"] = args.text_sim
    conf["index_name"] = args.name_prefix + "_" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:4]
    conf["data_source"] = args.data_source
    conf["embedders"] = args.embedders or [] 
    conf["time"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    conf["output_dir"] = "indexes/" + conf["index_name"]
    settings, mappings = build_settings_mappings(conf)
    conf["es_conf"] = {"settings": settings, "mappings": mappings}
    
    print(conf["output_dir"])
    return conf


def save(conf):
    f_out = conf["output_dir"] + "/" + "conf.json"
    Path(os.path.dirname(f_out)).mkdir(parents=True, exist_ok=True)
    with open(f_out, 'w') as fp:
        json.dump(conf, fp, indent=4)



