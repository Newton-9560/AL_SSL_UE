from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
import gc
model_path_dict = {
    'llama3': 'meta-llama/Meta-Llama-3-{size}B-Instruct',
    'opt': 'facebook/opt-{size}b',
    'qwen': 'Qwen/Qwen2.5-{size}B-Instruct',
    'mistral': 'mistralai/Mistral-{size}B-Instruct-v0.3',
    'llama3.1': "meta-llama/Meta-Llama-3.1-{size}B-Instruct"
}

def get_model_path(model_name: str, model_size: str):
    return model_path_dict[model_name].format(size=model_size)

class LLMs:
    def __init__(self, model_name: str, model_size: str, device='cuda', load_model=True):
        self.model_path = get_model_path(model_name, model_size)
        if 'Qwen' in self.model_path and '7B' in self.model_path:
            self.model_path = self.model_path + '-1M'
        self.model_name = self.model_path.split('/')[-1]
        self.device = device
        self.load_model = load_model
        if load_model:
            self.init_model()
        
    def delete_model(self):
        if self.load_model:
            del self.model
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
    def init_model(self):
        print("Initializing model: ", self.model_path)
        if 'llama' in self.model_path:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True) if '70B' in self.model_path else None
            model = LlamaForCausalLM.from_pretrained(self.model_path, device_map='auto', torch_dtype=torch.float16, quantization_config=quantization_config)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if 'llama' in self.model_path:
                model.config.pad_token_id = 128001
                model.generation_config.pad_token_id = 128001
                tokenizer.pad_token = tokenizer.eos_token
            self.model = model
            self.tokenizer = tokenizer
        elif 'opt' in self.model_path:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            model.config.eos_token_id = 50118
            model.generation_config.eos_token_id = 50118
            self.model = model
            self.tokenizer = tokenizer
        elif 'Qwen' in self.model_path:
            self.model_path = self.model_path + '-1M' if '7B' in self.model_path else self.model_path
            model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map='auto', torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = model
            self.tokenizer = tokenizer
        elif 'mistral' in self.model_path:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model.config.pad_token_id = 2
            model.generation_config.pad_token_id = 2
            tokenizer.pad_token = tokenizer.eos_token
            self.model = model
            self.tokenizer = tokenizer
            
        self.show_model_info()
        
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def show_model_info(self):
        config = AutoConfig.from_pretrained(self.model_path)
        self.num_layers = config.num_hidden_layers
        print(f"Number of layers: {self.num_layers}")
    
    def generate_response(self, prompt: str) -> str:
        return self.model.generate_response(prompt)
    
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=inputs.input_ids.shape[1]+20)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)