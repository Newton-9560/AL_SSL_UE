from lm_polygraph.utils.dataset import Dataset
from datasets import load_dataset

class TriviaQA:
    def __init__(self, batch_size: int, prompt_template: str, test=False):
        tqa = load_dataset("trivia_qa", "rc.nocontext")
        self.dataset = tqa["train"] if test is not True else tqa["test"]
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def subsample(self, sample_idx: list[int], seed=42):
        self.dataset = self.dataset.select(sample_idx)
        
    def __iter__(self):
        for data in self.dataset:
            prompt = self.prompt_template.format(question=data["question"])
            answer = {'correct_answer': data["answer"]['normalized_aliases'], 'incorrect_answers': None}
            yield prompt, answer
    
class CoQA:
    def __init__(self, batch_size: int, prompt_template: str, test=False):
        coqa = load_dataset("coqa")
        self.dataset = coqa["train"] if test is not True else coqa["test"]
        self.prompt_template = prompt_template
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.dataset)
    
    def subsample(self, sample_idx: list[int]):
        self.dataset = self.dataset.select(sample_idx)
    
    def __iter__(self):
        for data in self.dataset:
            for q, a in zip(data["questions"], data["answers"]["input_text"]):
                prompt = self.prompt_template.format(story=data["story"], question=q, answer=a)
                answer = {'correct_answer': [a], 'incorrect_answers': None}
                yield prompt, answer
                
class SciQ:
    def __init__(self, batch_size: int, prompt_template: str, test=False):
        sci_q = load_dataset("sciq")
        self.dataset = sci_q["train"] if test is not True else sci_q["test"]
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def subsample(self, sample_idx: list[int]):
        self.dataset = self.dataset.select(sample_idx)
        
    def __iter__(self):
        for data in self.dataset:
            prompt = self.prompt_template.format(question=data["question"], answer=data["correct_answer"])
            answer = {'correct_answer': data["correct_answer"], 'incorrect_answers': None}
            yield prompt, answer
        
class TruthfulQA:
    def __init__(self, batch_size: int, prompt_template: str, test=False):
        truthful_qa = load_dataset("truthful_qa", "generation")
        self.dataset = truthful_qa['validation']
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def subsample(self, sample_idx: list[int]):
        self.dataset = self.dataset.select(sample_idx)
        
    def __iter__(self):
        for data in self.dataset:
            prompt = self.prompt_template.format(question=data["question"])
            answer = {'correct_answer': data["correct_answers"], 'incorrect_answers': data["incorrect_answers"]}
            yield prompt, answer
            
class Tydiqa:
    def __init__(self, batch_size: int, prompt_template: str, test=False):
        tydiqa = load_dataset("tydiqa", "secondary_task")
        self.dataset = tydiqa["train"] if test is not True else tydiqa["validation"]
        self.dataset = self.dataset.filter(lambda x: 'english' in x['id'])
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def subsample(self, sample_idx: list[int]):
        self.dataset = [self.dataset[i] for i in sample_idx]
        
    def __iter__(self):
        for data in self.dataset:
            prompt = self.prompt_template.format(context=data["context"], question=data["question"])
            answer = {'correct_answer': data['answers']["text"], 'incorrect_answers': None}
            yield prompt, answer
            
class AmbigQA:
    def __init__(self, batch_size: int, prompt_template: str, test=False):
        ambig_qa = load_dataset("ambig_qa")
        self.dataset = ambig_qa["train"] if test is not True else ambig_qa["test"]
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def subsample(self, sample_idx: list[int]):
        self.dataset = self.dataset.select(sample_idx)
        
    def __iter__(self):
        for data in self.dataset:
            prompt = self.prompt_template.format(question=data["question"])
            answer = {'correct_answer': data["nq_answer"], 'incorrect_answers': None}
            yield prompt, answer


# TODO: test this dataset
class SimpleQA:
    def __init__(self, batch_size: int, prompt_template: str, test=False):
        simple_qa = load_dataset("basicv8vc/SimpleQA")
        self.dataset = simple_qa["test"]
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def subsample(self, sample_idx: list[int]):
        self.dataset = self.dataset.select(sample_idx)
        
        
    def __iter__(self):
        for data in self.dataset:
            prompt = self.prompt_template.format(question=data["problem"])
            answer = {'correct_answer': [data["answer"]], 'incorrect_answers': None}
            yield prompt, answer
            
# TODO: Squad
class Squad:
    def __init__(self, batch_size: int, prompt_template: str, test=False):
        squad = load_dataset("rajpurkar/squad")
        self.dataset = squad["train"] if test is not True else squad["validation"]
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def subsample(self, sample_idx: list[int]):
        self.dataset = self.dataset.select(sample_idx)
        
    def __iter__(self):
        for data in self.dataset:
            prompt = self.prompt_template.format(context=data["context"], question=data["question"])
            answer = {'correct_answer': data["answers"]['text'], 'incorrect_answers': None}
            yield prompt, answer
        
if __name__ == "__main__":
    # from llm_models.prompt import get_prompt_template
    # import torch
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # from lm_polygraph.utils.model import WhiteboxModel
    # from lm_polygraph.estimators import LexicalSimilarity, SemanticEntropy, SAR
    # import numpy as np
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # import random
    # random.seed(seed)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
    # model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    # # model_name = 'facebook/opt-6.7b'
    
    # tqa = TriviaQA(batch_size=1, prompt_template=get_prompt_template(model_name, "trivia_qa"))
    # sciq = SciQ(batch_size=1, prompt_template=get_prompt_template(model_name, "sciq"))
    # coqa = CoQA(batch_size=1, prompt_template=get_prompt_template(model_name, "coqa"))
    # from llm_models.models import LLMs
    # LLM = LLMs(model_name)
    # from lm_polygraph.utils.manager import estimate_uncertainty
    # ue_model = WhiteboxModel(LLM.get_model(), LLM.get_tokenizer())
    # sar = SAR()
    # for q, a in coqa:
    #     # output = estimate_uncertainty(ue_model, sar, input_text=q)
    #     # print(q)
    #     # print(output.generation_text)
    #     print(a)
    #     print('-'*100)
    # dataset = load_dataset("truthful_qa",  "generation")
    # dataset = dataset['validation']
    # print(dataset[0])

    dataset = load_dataset("basicv8vc/SimpleQA")

    print(dataset)
    print(dataset["test"][0])

    



    
    
