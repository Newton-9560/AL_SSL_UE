from tqdm import tqdm
import pickle
import os
import torch

from .utils import load_generated_result

def generate_hidden_state(inp_text, llm, layer_id):
    with torch.no_grad():
        input = llm.tokenizer(inp_text, return_tensors="pt").to('cuda')
        result = llm.model(input_ids=input.input_ids, output_hidden_states=True)
        hs = result['hidden_states'][layer_id][0][-1].cpu().detach().to(torch.float32).numpy()
    return hs

def generate_dataset(llm, dataset, layer_id, save=False):
    file_name = f'{llm.model_name}_{dataset}_layer_{layer_id}'
    file_path = os.path.join('/home/hanwenli/work/2025/AL_SSL/cache', f'{file_name}.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            print(f'Dataset {file_name} loaded from {file_path}')
            return pickle.load(f), file_name
    generated_result = load_generated_result(llm.model_name, dataset)
    dataset = []
    for id in tqdm(range(len(generated_result))):
        result = generated_result[id]
        inp_text = result['inputs']+result['answer']
        if llm.model_name == 'llama3':
            inp_text += '<|eot_id|>'
        if llm.model_name == 'qwen':
            inp_text += '<|im_end|>'
        result['hidden_state'] = generate_hidden_state(inp_text, llm, layer_id)
        result['id'] = id
        dataset.append(result)
    if save:
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
            print(f'Dataset {file_name} saved to {file_path}')
    return dataset, file_name
    
if __name__ == "__main__":
    from llm_models.models import LLMs
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    llm = LLMs('llama3')
    dataset = generate_dataset(llm, 'trivia_qa', layer_id=16)
    print(dataset)
