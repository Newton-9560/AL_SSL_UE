from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, AutoConfig
import torch

model_path_dict = {
    'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'opt': 'facebook/opt-6.7b',
    'qwen': 'Qwen/Qwen2.5-7B-Instruct-1M',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'mistral_7b': 'mistralai/Mistral-7B-Instruct-v0.3'
}

class LLMs:
    def __init__(self, model_name: str, device='cuda'):
        self.model_name = model_name
        self.device = device
        self.init_model()
        
    def init_model(self):
        if 'llama' in self.model_name:
            model = LlamaForCausalLM.from_pretrained(model_path_dict[self.model_name], device_map=self.device, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(model_path_dict[self.model_name])
            if 'llama3' in self.model_name:
                model.config.pad_token_id = 128001
                model.generation_config.pad_token_id = 128001
                tokenizer.pad_token = tokenizer.eos_token
            self.model = model
            self.tokenizer = tokenizer
        elif 'opt' in self.model_name:
            model = AutoModelForCausalLM.from_pretrained(model_path_dict[self.model_name], device_map=self.device, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(model_path_dict[self.model_name], use_fast=False)
            model.config.eos_token_id = 50118
            model.generation_config.eos_token_id = 50118
            self.model = model
            self.tokenizer = tokenizer
        elif 'qwen' in self.model_name:
            model = AutoModelForCausalLM.from_pretrained(model_path_dict[self.model_name], device_map=self.device, torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(model_path_dict[self.model_name])
            self.model = model
            self.tokenizer = tokenizer
        elif 'mistral' in self.model_name:
            model = AutoModelForCausalLM.from_pretrained(model_path_dict[self.model_name], device_map=self.device, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(model_path_dict[self.model_name])
            model.config.pad_token_id = 2
            model.generation_config.pad_token_id = 2
            tokenizer.pad_token = tokenizer.eos_token
            print(next(model.parameters()).dtype)
            self.model = model
            self.tokenizer = tokenizer
            
        self.show_model_info()
        
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def show_model_info(self):
        config = AutoConfig.from_pretrained(model_path_dict[self.model_name])
        self.num_layers = config.num_hidden_layers
        print(f"Number of layers: {self.num_layers}")
    
    def generate_response(self, prompt: str) -> str:
        return self.model_name.generate_response(prompt)
    
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=inputs.input_ids.shape[1]+20)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)