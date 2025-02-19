import os
import json 
from pathlib import Path


######################## Instance-wise Metrics ####################################

# https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k
# Precision@K
def p_at_k(k, relevant_set, retrieved_docs):
    retrieved_set = set(retrieved_docs[:k])  # take only the top-k retrieved documents
    precision = len(retrieved_set & relevant_set) / k  # compute Precision@k
    return precision


# https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k
# Recall@K
def r_at_k(k, relevant_set, retrieved_docs):
    retrieved_set = set(retrieved_docs[:k])  # take only the top-k retrieved documents
    recall = len(retrieved_set & relevant_set) / len(relevant_set) # compute Recall@k
    return recall


# AveragePrecision@K
def ap_at_k(k, relevant_set, retrieved_docs):
    precisions = []
    for i, doc_id in enumerate(retrieved_docs[:k]):  # iterate through top-k results
        if doc_id in relevant_set:
            precision_at_i = p_at_k(i+1, relevant_set, retrieved_docs)  # precision at rank i
            precisions.append(precision_at_i)
    
    average_precision = sum(precisions) / len(relevant_set)  # average precision
    return average_precision


# ReciprocalRank@K
def rr_at_k(k, relevant_set, retrieved_docs):
    rr = 0                                               # if there is no relevant item
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in  relevant_set:
            rr = 1 / (i+1)                               # calculate reciprocal rank 
            break
    return rr

    
######################## Dataset-wise metrics   ##################################### 

def dataset_precision_at_k(precision_scores):
    scores = [prec_k for q, prec_k in precision_scores.items()]
    return sum(scores) / len(scores)


def dataset_recall_at_k(recall_scores):
    scores = [recall_k for q, recall_k in recall_scores.items()]
    return sum(scores) / len(scores)


# https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map
# MeanAveragePrecision@K
def map_at_k(ap_scores, relevant_docs):
    """
    MAP metric rewards the system’s ability to place relevant items at the top. 
    MAP equals 1 in the case of perfect ranking when all relevant documents or items are at the top of the list. 
    MAP equals 0 when no relevant objects are retrieved.    
    MAP can be between 0 and 1 in all other cases. The closer the MAP score is to 1, the better the ranking performance.   
    """
    scores = [recall_k for q, recall_k in ap_scores.items()]
    return sum(scores) / len(relevant_docs)


# https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr
# MeanReciprocalRank@K
def mrr_at_k(rr_scores):
    scores = [rr_k  for q, rr_k in rr_scores.items()]
    return sum(scores) / len(scores)
    


def save_requests_responses(content, output_dir, f_name):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
    
    if "json" in f_name:
        with open(output_dir + "/" + f_name, "w", encoding='utf-8') as json_file:
            json.dump(content, json_file, indent=4)
    else:
        with open(output_dir + "/" + f_name, "w", encoding='utf-8') as text_file:
            text_file.write(content)
        

######################## generator metrics   ##################################### 


def generator_accuracy(answers, ground_truths):
        correct_answers = 0
        total_questions = len(ground_truths)

        for answer, correct in zip(answers, ground_truths):
            if answer == correct:
                correct_answers += 1

        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        return accuracy

            
def validate(questions, contexts, answers, ground_truths):
    metrics = {}
    metrics["accuracy"] = self.generator_accuracy(answers, ground_truths)
    
    ############
    answers=[]
    for item in contexts:
        tmp = []
        for candidate in item:
            tmp.append(candidate["symptom_id"])
        answers.append(tmp)
    
    retrieved_docs = {}
    for question, answer in zip(questions, answers):
        retrieved_docs[question] = answer
        
    relevant_docs = {}
    for question, gt in zip(questions, ground_truths):
        relevant_docs[question] = {gt}
        
        
    k=5    
    precisions = dict()
    recalls = dict()
    aps = dict()
    rrs = dict()
    for query_id in retrieved_docs.keys():
        precision = p_at_k(k, relevant_docs[query_id], retrieved_docs[query_id])
        recall = r_at_k(k, relevant_docs[query_id], retrieved_docs[query_id])
        ap = ap_at_k(k, relevant_docs[query_id], retrieved_docs[query_id])
        rr = rr_at_k(k, relevant_docs[query_id], retrieved_docs[query_id])
        
        precisions[query_id] = precision
        recalls[query_id] = recall
        aps[query_id] = ap
        rrs[query_id] = rr
        
        print(f"{query_id}: Precision@{k}: {precision}, Recall@{k}: {recall}, AP@{k}: {ap}, RR@{k}: {rr}")

        print(f"\nDataset Precision@{k}: {dataset_precision_at_k(precisions)}")
        print(f"Dataset Recall@{k}: {dataset_recall_at_k(recalls)}")
        print(f"MAP@{k}: {map_at_k(aps, relevant_docs)}")
        print(f"MRR@{k}: {mrr_at_k(rrs)}")
        
        
        
    return metrics





if True:
    # Example usage
    relevant_docs = {"q1": {"d1", "d3"}, "q2": {"d2", "d4"}}
    retrieved_docs = {"q1": ["d1", "d2", "d3"], "q2": ["d5", "d2", "d4"]}
    k = 3


    precisions = dict()
    recalls = dict()
    aps = dict()
    rrs = dict()
    for query_id in retrieved_docs.keys():
        precision = p_at_k(k, relevant_docs[query_id], retrieved_docs[query_id])
        recall = r_at_k(k, relevant_docs[query_id], retrieved_docs[query_id])
        ap = ap_at_k(k, relevant_docs[query_id], retrieved_docs[query_id])
        rr = rr_at_k(k, relevant_docs[query_id], retrieved_docs[query_id])
        
        precisions[query_id] = precision
        recalls[query_id] = recall
        aps[query_id] = ap
        rrs[query_id] = rr
        
        print(f"{query_id}: Precision@{k}: {precision}, Recall@{k}: {recall}, AP@{k}: {ap}, RR@{k}: {rr}")

    print(f"\nDataset Precision@{k}: {dataset_precision_at_k(precisions)}")
    print(f"Dataset Recall@{k}: {dataset_recall_at_k(recalls)}")
    print(f"MAP@{k}: {map_at_k(aps, relevant_docs)}")
    print(f"MRR@{k}: {mrr_at_k(rrs)}")