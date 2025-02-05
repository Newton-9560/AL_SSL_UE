import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer
import pickle
from main import train_loop, validation_loop, run, get_split, MLP
from loader import generate_data_and_labels
class HiddenStatesDataset(Dataset):
    def __init__(self, data, label, threshold=0.7):
        self.data = data
        self.label = label
        self.threshold = threshold

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.threshold is None:
            label = self.label[idx]['align_score']
        else:
            label = 1 if self.label[idx]['align_score'] >= self.threshold else 0
            
        return self.data[idx], label
    
    def add(self, new_data, new_label):
        self.data = np.concatenate((self.data, new_data.reshape(1, -1)), axis=0)
        self.label = np.concatenate((self.label, np.array([new_label])))
        
    def delete(self, idx):
        if 0 <= idx < len(self.data):
            self.data = np.delete(self.data, idx, axis=0)
            self.label = np.delete(self.label, idx, axis=0)
        else:
            raise IndexError("Index out of range")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
TRAIN_NUM = 2000
VAL_NUM = 500
THRESHOLD = 0.7
MODEL = 'qwen'
DATASET = 'sciq'
if MODEL == 'llama3':
    model = LlamaForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', device_map='cuda', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = 128001
    model.generation_config.pad_token_id = 128001
elif MODEL == 'qwen':
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct-1M', device_map='cuda', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct-1M')

if DATASET == 'tqa':    
    train_raw = []
    for i in range(4):
        with open('/home/hanwenli/work/llm-early-exit/SSL/dataset/generate/output/num_3000split_trainid_'+str(i)+'.pkl', 'rb') as f:
            train_raw += pickle.load(f)
    with open('/home/hanwenli/work/llm-early-exit/SSL/dataset/generate/output/num_17944split_validationid_0.pkl', 'rb') as f:
        val_raw = pickle.load(f)
    
    train_raw = train_raw[:TRAIN_NUM]
    val_raw = val_raw[:VAL_NUM]
    
elif DATASET == 'coqa':
    coqa_dataset = {}
    for i in range(4):
        with open('/home/hanwenli/work/UQ_SSL/dataset/output/result'+str(i+1)+'.pkl', 'rb') as f:
            temp_data = pickle.load(f)
            coqa_dataset.update(temp_data)
                
    train_keys = list(coqa_dataset.keys())[:130]
    val_keys = list(coqa_dataset.keys())[520:]
    train_raw = []
    for key in train_keys:
        train_raw.extend([data for data in coqa_dataset[key]])
    val_raw = []
    for key in val_keys:
        val_raw.extend([data for data in coqa_dataset[key]])
        
    print(len(train_raw), len(val_raw))

elif DATASET == 'sciq':
    with open('/home/hanwenli/work/2025/AL_SSL/data/qwen/sciq/a18a8260-2a7a-42a7-9234-5d977e218b46.npy', 'rb') as f:
        raw_data = np.load(f, allow_pickle=True)
    train_raw = raw_data[:TRAIN_NUM]
    val_raw = raw_data[-VAL_NUM:]
    

layers = np.arange(8, 24, 2)
print(layers)

result = {}
for layer in layers:
    train_data, train_label = generate_data_and_labels(model, tokenizer, train_raw, layer=layer)
    val_data, val_label = generate_data_and_labels(model, tokenizer, val_raw, layer=layer)

    labeled_dataset = HiddenStatesDataset(train_data, train_label, threshold=THRESHOLD)
    val_dataset = HiddenStatesDataset(val_data, val_label, threshold=THRESHOLD)
    print('The length of labeled dataset is', len(labeled_dataset))
    print('The length of validation dataset is', len(val_dataset))

    # Define model
    mlp_model = MLP().cuda()

    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_roc, best_accuracy = run(mlp_model, optimizer, criterion, labeled_dataset, 
                                val_dataset, batch_size=BATCH_SIZE, epoch=EPOCHS, threshold=THRESHOLD)
        
    print(best_roc, best_accuracy)
    result[layer] = best_roc

with open('./result_sciq.pkl', 'wb') as f:
    pickle.dump(result, f)