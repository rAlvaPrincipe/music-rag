import argparse
from pathlib import Path
import json
import os
from datetime import datetime
import hashlib
import time


def build_output_file_path(conf):
    output_dir = "results/" + conf["dataset"] + "/" + conf["llm"]["inference"]["provider"] + "_" + conf["llm"]["inference"]["model"].replace(":", "_").replace("/", "_") + "_"
    output_dir += "__V" + conf["version"]
    return output_dir     


def parse():
    parser = argparse.ArgumentParser(description="RAG Configuration", allow_abbrev=False)
    
    # Shared arguments
    parser.add_argument('--mode', required=True, choices=['evaluation', 'inference'], help='Specify whether to run in evaluation or inference mode.')
    parser.add_argument('--index_name', required=True, help='The name of the Elasticsearch index.')
    parser.add_argument('--embedder', required=True, help='specifies the model used for embedding the question and retrieving chunks. It must match one of the models used to vectorize the corpus.')
    parser.add_argument('--retrieval_mode', required=True, choices=['dense', 'hybrid', 'dense_plus_kg', 'hybrid_plus_kg'], help='Choose between dense or hybrid retrieval strategies.')
    parser.add_argument('--include_metadata', required=True, choices=['yes', 'no'], help='Should each chunk presented to the LLM include metadata (e.g., {score: 9.47, source_title: Radiohead, text: ...}) or just the plain chunk text?')
    parser.add_argument('--inf_llm_provider', required=True, help='e.g., groq, aws, openai')
    parser.add_argument('--inf_llm_model', required=True, help='e.g., llama-3.1-70b-versatile')

    # Mode-specific arguments
    parser.add_argument('--dataset', help='Path to the dataset (required for evaluation mode).')
    parser.add_argument('--question', help='The question to ask (required for inference mode).')
    parser.add_argument('--eval_llm_provider', help='e.g., groq, aws, openai (required for evaluation mode)')
    parser.add_argument('--eval_llm_model', help='e.g., llama-3.1-70b-versatile (required for evaluation mode)')

    args = parser.parse_args() 

    # Validate mode-specific arguments
    if args.mode == "evaluation" and (not args.dataset or not args.inf_llm_provider or not args.inf_llm_model or not args.eval_llm_provider or not args.eval_llm_model):
        parser.error('--dataset is required for evaluation mode.')
    if args.mode == "inference" and (not args.question or not args.inf_llm_provider or not args.inf_llm_model):
        parser.error('--question is required for inference mode.')

    if not args.index_name or not args.embedder or not args.retrieval_mode or not args.include_metadata :
        parser.error('please specify all the arguments.')     
    return args


def build_conf(args):
    conf = personalize(args)
    return conf


def personalize(args):
    conf = {}
    conf["mode"] = args.mode
    conf["index_name"] = args.index_name
    if args.mode == "evaluation":
        conf["dataset"] = args.dataset
    else:
        conf["question"] = args.question
    conf["embedder"] = args.embedder
    conf["retrieval_mode"] = args.retrieval_mode
    conf["include_metadata"] = True if args.include_metadata.lower() == "yes" else False
    conf["llm"] = { "inference": {"provider": args.inf_llm_provider, "model": args.inf_llm_model}, "evaluation": {"provider": args.eval_llm_provider, "model": args.eval_llm_model}}
    
    if conf["mode"] == "evaluation":
        conf["time"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
        conf["version"] = hashlib.sha256(str(time.time()).encode()).hexdigest()[:4]
        conf["output_dir"] = build_output_file_path(conf)
        print(conf["output_dir"])
    return conf




def save(conf):
    f_out = conf["output_dir"] + "/conf.json"
    Path(os.path.dirname(f_out)).mkdir(parents=True, exist_ok=True)
    with open(f_out, 'w') as fp:
        json.dump(conf, fp, indent=4)
        


def build_conf_from_args():
    args = parse()
    conf = personalize(args)
    if conf["mode"] == "evaluation":
        save(conf)
    return conf
    