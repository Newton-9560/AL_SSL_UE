import argparse
import numpy as np
import os
from lm_polygraph import estimate_uncertainty
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import SemanticEntropy, SAR, MaximumSequenceProbability, LexicalSimilarity, MonteCarloSequenceEntropy
from tqdm import tqdm
import uuid
import torch
from dataset.utils import fix_seed, cal_similarity
from llm_models.models import LLMs, get_model_path
from dataset.dataset import TriviaQA, CoQA, SciQ, TruthfulQA, Tydiqa, AmbigQA, Squad, SimpleQA
from llm_models.prompt import get_prompt_template
from transformers import logging, DebertaForSequenceClassification, DebertaTokenizer
import gc
import time
logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset with model outputs')
    parser.add_argument('--model', type=str, default='llama3.1',
                        choices=['llama3', 'opt', 'qwen', 'mistral', 'llama3.1'],
                        help='Model name to use for generation')
    parser.add_argument('--model_size', type=str, default='70',
                        help='Model size to use for generation')
    parser.add_argument('--dataset', type=str, default='trivia_qa',
                        choices=['coqa', 'trivia_qa', 'sciq', 'truthful_qa', 'tydiqa', 'ambig_qa', 'squad', 'simple_qa'],
                        help='Dataset to use')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--num_samples', type=int, default=4000,
                        help='Number of samples to generate')
    parser.add_argument('--num_splits', type=int, default=1,
                        help='Number of splits to divide the dataset into')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--serial_id', type=int, default=0,
                        help='Serial ID to use for the dataset')
    parser.add_argument('--save_path', type=str, default='/home/hanwenli/work/2025/AL_SSL/data',
                        help='Path to save the dataset')
    
    args = parser.parse_args()
    
    print("Working on model: ", get_model_path(args.model, args.model_size), " Dataset: ", args.dataset)

    return args

def load_dataset(model, dataset, split):
    prompt_template = get_prompt_template(model, dataset)
    print('prompt_template: ', prompt_template)
    if dataset == 'coqa':
        dataset = CoQA(batch_size=1, prompt_template=prompt_template, test=split == 'test')
    elif dataset == 'trivia_qa':
        dataset = TriviaQA(batch_size=1, prompt_template=prompt_template, test=split == 'test')
    elif dataset == 'sciq':
        dataset = SciQ(batch_size=1, prompt_template=prompt_template, test=split == 'test')
    elif dataset == 'truthful_qa':
        dataset = TruthfulQA(batch_size=1, prompt_template=prompt_template)
    elif dataset == 'tydiqa':
        dataset = Tydiqa(batch_size=1, prompt_template=prompt_template)
    elif dataset == 'ambig_qa':
        dataset = AmbigQA(batch_size=1, prompt_template=prompt_template)
    elif dataset == 'squad':
        dataset = Squad(batch_size=1, prompt_template=prompt_template)
    elif dataset == 'simple_qa':
        dataset = SimpleQA(batch_size=1, prompt_template=prompt_template)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    return dataset

def subsample_dataset(model, dataset, split, num_samples, num_splits, serial_id=0):
    dataset = load_dataset(model, dataset, split)
    if num_samples > 0:
        sample_idx = np.random.choice(np.arange(0, len(dataset)), size=num_samples, replace=False)
        print(sample_idx)
        split_size = len(sample_idx) // num_splits
        sample_idx = sample_idx[serial_id * split_size:(serial_id + 1) * split_size]
        dataset.subsample(sample_idx)
    print("The length of the dataset is: ", len(dataset))
    return dataset

def ue_method():
    return [
        MonteCarloSequenceEntropy(),
        LexicalSimilarity(),
        SemanticEntropy(),
        MaximumSequenceProbability(),
        SAR()
    ]
    
def load_deberta(model_path = 'microsoft/deberta-large-mnli'):
    deberta = DebertaForSequenceClassification.from_pretrained(
                model_path, problem_type="multi_label_classification"
            )
    deberta.to('cuda')
    deberta.eval()
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    return {'deberta': deberta, 'tokenizer': tokenizer}
    
def generate_result(dataset, ue_model, ue_methods):
    deberta = load_deberta()
    result = []
    for i, (q, a) in enumerate(dataset):
        data_point = {}
        for method in ue_methods:
            start_time = time.time()
            try:
                output = estimate_uncertainty(ue_model, method, q)
                data_point[method.__class__.__name__.lower()] = output.uncertainty
            except Exception as e:
                print(f"Error in {method.__class__.__name__}: {e}")
                data_point[method.__class__.__name__.lower()] = None
                
            generation_time = time.time() - start_time
            print(f"{method.__class__.__name__} time: {generation_time:.4f}s")
        align_scores = [cal_similarity(output.generation_text, target_text) for target_text in a['correct_answer']]
        data_point['align'] = max(align_scores)
        data_point['inputs'] = q
        data_point['target_texts'] = a
        data_point['answer'] = output.generation_text
        print(f"Processing {i+1}/{len(dataset)}: ", output.generation_text)
        result.append(data_point)
        
        # DEBUG
        print(data_point['inputs'])
        print(data_point['answer'])
        print(a['correct_answer'])
        print('*'*100)
        if i > 10:
            break
        
        gc.collect()
        torch.cuda.empty_cache()
    return result

def save_result(result, save_path, EID):
    result_array = np.array(result)
    output_file = os.path.join(save_path, f'{EID}.npy')
    np.save(output_file, result_array)
    print(f"Results saved to {output_file}")

def main():
    EID = str(uuid.uuid4())
    print(f"Experiment ID: {EID}")
    args = parse_args()
    fix_seed(args.seed)
    dataset = subsample_dataset(args.model, args.dataset, args.split, args.num_samples, args.num_splits, args.serial_id)
    llm = LLMs(args.model, args.model_size)
    
    ue_model = WhiteboxModel(llm.get_model(), llm.get_tokenizer())
    ue_methods = ue_method()
    
    result = generate_result(dataset, ue_model, ue_methods)
    
    save_path = os.path.join(args.save_path, args.model + '_' + args.model_size, args.dataset)
    os.makedirs(save_path, exist_ok=True)
    
    save_result(result, save_path, EID)
    # print(result)

if __name__ == "__main__":
    print("Starting to generate dataset")
    main()