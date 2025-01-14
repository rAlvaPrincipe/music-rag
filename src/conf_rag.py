import argparse


def parse():
    parser = argparse.ArgumentParser(description="Indexing", allow_abbrev=False)
    parser.add_argument('--index_name', help='the name of the eslasticsearch index')
    parser.add_argument('--question', help='question about musical artists')
    parser.add_argument('--embedder', help='specifies the model used for embedding the question and retrieving chunks. It must match one of the models used to vectorize the corpus.')
    parser.add_argument('--retrieval_mode', help='dense or hybrid')
    parser.add_argument('--include_metadata', help='yes or no. Should each chunk presented to the LLM include metadata (e.g., {score: 9.47, source_title: Radiohead, text: ...}) or just the plain chunk text?')

    args = parser.parse_args()
    
    if not args.index_name or not args.question or not args.embedder or not args.retrieval_mode or not args.include_metadata:
        parser.error('please specify all the arguments')
    return args


def build_conf(args):
    conf = personalize(args)
    return conf


def personalize(args):
    conf = {}
    conf["index_name"] = args.index_name
    conf["question"] = args.question
    conf["embedder"] = args.embedder
    conf["retrieval_mode"] = args.retrieval_mode
    conf["include_metadata"] = True if args.include_metadata.lower() == "yes" else False
    return conf




