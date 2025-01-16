import argparse


def parse():
    parser = argparse.ArgumentParser(description="RAG Configuration", allow_abbrev=False)
    
    # Shared arguments
    parser.add_argument('--mode', required=True, choices=['evaluation', 'inference'], help='Specify whether to run in evaluation or inference mode.')
    parser.add_argument('--index_name', required=True, help='The name of the Elasticsearch index.')
    parser.add_argument('--embedder', required=True, help='specifies the model used for embedding the question and retrieving chunks. It must match one of the models used to vectorize the corpus.')
    parser.add_argument('--retrieval_mode', required=True, choices=['dense', 'hybrid'], help='Choose between dense or hybrid retrieval strategies.')
    parser.add_argument('--include_metadata', required=True, choices=['yes', 'no'], help='Should each chunk presented to the LLM include metadata (e.g., {score: 9.47, source_title: Radiohead, text: ...}) or just the plain chunk text?')

    # Mode-specific arguments
    parser.add_argument('--dataset', help='Path to the dataset (required for evaluation mode).')
    parser.add_argument('--question', help='The question to ask (required for inference mode).')

    args = parser.parse_args() 

    # Validate mode-specific arguments
    if args.mode == "evaluation" and not args.dataset:
        parser.error('--dataset is required for evaluation mode.')
    if args.mode == "inference" and not args.question:
        parser.error('--question is required for inference mode.')

    if not args.index_name or not args.embedder or not args.retrieval_mode or not args.include_metadata:
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
    return conf




