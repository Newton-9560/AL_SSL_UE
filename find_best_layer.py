import argparse
from tqdm import tqdm
import os
import pickle

from llm_models.models import LLMs, model_path_dict
from hidden_state.generate import generate_dataset
from hidden_state.utils import split_dataset
from trainer import Trainer
from utils import parse_args, fix_seed

best_layer_dict = {
    'llama3': 17,
    'opt': 21,
    'qwen': 22
}

def single_layer_train():
    config = parse_args()
    fix_seed(config.seed)
    llm = LLMs(config.model)
    layer = best_layer_dict[config.model]
    results = []
    print(f"Training on layer {layer}")
    dataset, file_name = generate_dataset(llm, config.dataset, layer_id=layer, save=False)
    train_dataset, validation_dataset = split_dataset(dataset, int(config.split_portion*len(dataset)))
    trainer = Trainer(config.model_name, dim=4096 if 'qwen' not in config.model else 3584)
    results.append(trainer.train_supervised(train_dataset, validation_dataset, config))
    print(results)
    
def evaluate_model():
    config = parse_args()
    fix_seed(config.seed)
    llm = LLMs(config.model)
    layer = best_layer_dict[config.model]
    results = []
    print(f"Training on layer {layer}")
    dataset, file_name = generate_dataset(llm, config.dataset, layer_id=layer, save=False)
    train_dataset, validation_dataset = split_dataset(dataset, int(config.split_portion*len(dataset)))
    trainer = Trainer(config.model_name, dim=4096 if 'qwen' not in config.model else 3584)
    trainer.load_checkpoint(os.path.join('./checkpoints', 'opt' + '_' + 'trivia_qa' +'.pth'))
    trainer.set_criterion()
    predictions, labels, total_loss = trainer.validate(validation_dataset, config)
    metrics = trainer.cal_metrics(predictions, labels, total_loss)
    print(metrics)
    
def main():
    config = parse_args()
    fix_seed(config.seed)
    llm = LLMs(config.model)
    results = []
    for layer in range(llm.num_layers):
        print(f"Training on layer {layer}")
        dataset, file_name = generate_dataset(llm, config.dataset, layer_id=layer, save=False)
        train_dataset, validation_dataset = split_dataset(dataset, int(config.split_portion*len(dataset)))
        trainer = Trainer(config.model_name, dim=4096 if 'qwen' not in config.model else 3584)
        results.append(trainer.train_supervised(train_dataset, validation_dataset, config))

    result_path = os.path.join('./results', file_name+'.pkl') 
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
        print(f"Results saved to {result_path}")
    print(results)

if __name__ == "__main__":
    # main()
    # single_layer_train()
    evaluate_model()