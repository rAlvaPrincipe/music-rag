from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from langchain_core.prompts.prompt import PromptTemplate
from tqdm import tqdm
import llms
from ragas import evaluate  as ragas_evaluate
import evaluate as hf_evaluate


llm = llms.get_llm("groq", "llama-3.1-70b-versatile")

def create_ragas_dataset(questions, contexts, answers, ground_truths):
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


def ragas_evaluation(dataset):
    evaluator_llm = LangchainLLMWrapper(llm)
    metrics = [LLMContextRecall(), LLMContextPrecisionWithReference()]
    return ragas_evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)


def generation_evaluation(predicted_answers, ground_truths):
    bleu = hf_evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=predicted_answers, references=ground_truths)

    rouge = hf_evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predicted_answers, references=ground_truths)

    meteor = hf_evaluate.load('meteor')
    meteor_results = meteor.compute(predictions=predicted_answers, references=ground_truths)

    return {'bleu': bleu_results['bleu'], 'rouge': rouge_results, 'meteor': meteor_results['meteor']}


def llm_as_a_judge(questions, predicted_answers, ground_truths):
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

    prompt = PromptTemplate( input_variables=["question", "ground_truth", "predicted_answer"], template=_PROMPT_TEMPLATE)
    grades =  {"correct": 0, "incorrect": 0, "partially correct": 0}
    
    for x in tqdm(range(len(questions))):
        chain = prompt | llm
        #grades.append(chain.invoke({"question": questions[x], "ground_truth": ground_truths[x], "predicted_answer": predicted_answers[x]}))
        grade = chain.invoke({"question": questions[x], "ground_truth": ground_truths[x], "predicted_answer": predicted_answers[x]})
        grade = grade.content.lower()
        grades[grade] += 1
        
    return {'llm-as-a-judge': grades}



def validate(questions, contexts, answers, ground_truths):
    dataset = create_ragas_dataset(questions, contexts, answers, ground_truths)
    
    metrics = {}
    metrics.update({'ragas': str(ragas_evaluation(dataset))})
    metrics.update(generation_evaluation(answers, ground_truths))
    metrics.update(llm_as_a_judge(questions, answers, ground_truths))
    
    return metrics