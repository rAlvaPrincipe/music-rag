from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from langchain_core.prompts.prompt import PromptTemplate
from tqdm import tqdm
import llms
from ragas import evaluate  as ragas_evaluate
import evaluate as hf_evaluate
import os
import json
from pathlib import Path


class Validator:
    
    def __init__(self, llm_provider, llm_model):    
        self.llm = llms.get_llm(llm_provider, llm_model)
        
            

    def create_ragas_dataset(self, questions, contexts, answers, ground_truths):
        samples = []
        for i in range(len(questions)):
            samples.append(
                SingleTurnSample(
                    user_input=questions[i],
                    retrieved_contexts=contexts[i],
                    response=answers[i],
                    reference=ground_truths[i]
                )
            )
            
        return EvaluationDataset(samples=samples)


    def ragas_evaluation(self, dataset):
        evaluator_llm = LangchainLLMWrapper(self.llm)
        metrics = [LLMContextRecall(), LLMContextPrecisionWithReference()]
        return ragas_evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)


    def generation_evaluation(self, predicted_answers, ground_truths):
        bleu = hf_evaluate.load("bleu")
        bleu_results = bleu.compute(predictions=predicted_answers, references=ground_truths)

        rouge = hf_evaluate.load('rouge')
        rouge_results = rouge.compute(predictions=predicted_answers, references=ground_truths)

        meteor = hf_evaluate.load('meteor')
        meteor_results = meteor.compute(predictions=predicted_answers, references=ground_truths)

        return {'bleu': bleu_results['bleu'], 'rouge': rouge_results, 'meteor': meteor_results['meteor']}


    def llm_as_a_judge(self, questions, predicted_answers, ground_truths):
        _PROMPT_TEMPLATE = """You are an expert professor specialized in grading students' answers to questions.
        You are grading the following question:
        {question}
        Here is the real answer:
        {ground_truth}
        You are grading the following predicted answer:
        {predicted_answer}
        Respond only with CORRECT, INCORRECT or PARTIALLY CORRECT:
        Grade:
        """

        prompt = PromptTemplate(input_variables=["question", "ground_truth", "predicted_answer"], template=_PROMPT_TEMPLATE)
        grades =  {"correct": 0, "incorrect": 0, "partially correct": 0}
        
        for x in tqdm(range(len(questions))):
            chain = prompt | self.llm
            #grades.append(chain.invoke({"question": questions[x], "ground_truth": ground_truths[x], "predicted_answer": predicted_answers[x]}))
            grade = chain.invoke({"question": questions[x], "ground_truth": ground_truths[x], "predicted_answer": predicted_answers[x]})
            grade = grade.content.lower()
            grades[grade] += 1
            
        return {'llm-as-a-judge': grades}



    def validate(self, questions, contexts, answers, ground_truths):
        dataset = self.create_ragas_dataset(questions, contexts, answers, ground_truths)
        
        metrics = {}
        metrics.update({'ragas': eval(str(self.ragas_evaluation(dataset)))})
        metrics.update(self.generation_evaluation(answers, ground_truths))
        metrics.update(self.llm_as_a_judge(questions, answers, ground_truths))

        return metrics
    
    
    def save_metrics(self, metrics, output_f):
        output_dir = Path(output_f).parent.absolute()
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir) 
            
        with open(output_f, "w", encoding='utf-8') as json_file:
            json.dump(metrics, json_file, indent=4)
        
    
    def save_requests_responses(self, content, output_dir, f_name):
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir) 
        
        if "json" in f_name:
            with open(output_dir + "/" + f_name, "w", encoding='utf-8') as json_file:
                json.dump(content, json_file, indent=4)
        else:
            with open(output_dir + "/" + f_name, "w", encoding='utf-8') as text_file:
                text_file.write(content)