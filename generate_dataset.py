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
from llm_models.models import LLMs, model_path_dict
from dataset.dataset import TriviaQA, CoQA, SciQ
from llm_models.prompt import get_prompt_template
from transformers import logging, DebertaForSequenceClassification, DebertaTokenizer
import gc

# logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset with model outputs')
    parser.add_argument('--model', type=str, default='opt',
                        choices=['llama3', 'opt', 'qwen', 'llama2-13b'],
                        help='Model name to use for generation')
    parser.add_argument('--dataset', type=str, default='trivia_qa',
                        choices=['coqa', 'trivia_qa', 'sciq'],
                        help='Dataset to use')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate')
    parser.add_argument('--num_splits', type=int, default=1,
                        help='Number of splits to divide the dataset into')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--serial_id', type=int, default=0,
                        help='Serial ID to use for the dataset')
    
    args = parser.parse_args()
    
    print("Working on model: ", model_path_dict[args.model], " Dataset: ", args.dataset)

    return args

def load_dataset(model_path, args):
    prompt_template = get_prompt_template(model_path, args.dataset)
    print('prompt_template: ', prompt_template)
    if args.dataset == 'coqa':
        dataset = CoQA(batch_size=1, prompt_template=prompt_template, test=args.split == 'test')
    elif args.dataset == 'trivia_qa':
        dataset = TriviaQA(batch_size=1, prompt_template=prompt_template, test=args.split == 'test')
    elif args.dataset == 'sciq':
        dataset = SciQ(batch_size=1, prompt_template=prompt_template, test=args.split == 'test')
    return dataset

def subsample_dataset(model_name, args):
    model_path = model_path_dict[model_name]
    dataset = load_dataset(model_path, args)
    sample_idx = np.random.choice(np.arange(0, len(dataset)), size=args.num_samples, replace=False)
    print(sample_idx)
    split_size = len(sample_idx) // args.num_splits
    sample_idx = sample_idx[args.serial_id * split_size:(args.serial_id + 1) * split_size]
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
            try:
                output = estimate_uncertainty(ue_model, method, q, deberta)
                data_point[method.__class__.__name__.lower()] = output.uncertainty
            except Exception as e:
                print(f"Error in {method.__class__.__name__}: {e}")
                data_point[method.__class__.__name__.lower()] = None
        align_scores = [cal_similarity(output.generation_text, target_text) for target_text in a]
        data_point['align'] = max(align_scores)
        data_point['inputs'] = q
        data_point['target_texts'] = a
        data_point['answer'] = output.generation_text
        print(f"Processing {i+1}/{len(dataset)}: ", output.generation_text)
        result.append(data_point)
        
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
    dataset = subsample_dataset(args.model, args)
    llm = LLMs(args.model)
    
    ue_model = WhiteboxModel(llm.get_model(), llm.get_tokenizer())
    ue_methods = ue_method()
    
    result = generate_result(dataset, ue_model, ue_methods)
    
    save_path = os.path.join('/home/hanwenli/work/2025/AL_SSL/data', args.model, args.dataset)
    os.makedirs(save_path, exist_ok=True)
    
    save_result(result, save_path, EID)
    # print(result)

if __name__ == "__main__":
    print("Starting to generate dataset")
    main()