import argparse
from tqdm import tqdm
import os
import pickle

from llm_models.models import LLMs, model_path_dict
from hidden_state.generate import generate_dataset
from hidden_state.utils import split_dataset
from trainer import Trainer, fix_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset with model outputs')
    parser.add_argument('--model', type=str, default='llama2-13b',
                        choices=['llama3', 'opt', 'qwen', 'llama2-13b'],
                        help='Model name to use for generation')
    parser.add_argument('--dataset', type=str, default='coqa',
                        choices=['coqa', 'trivia_qa', 'sciq'],
                        help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--model_name', type=str, default='mlp',
                        help='Model name to use for training')
    parser.add_argument('--align_threshold', type=float, default=0.7,
                        help='Align threshold')
    # TODO:: add uncertainty type
    parser.add_argument('--uncertainty_type', type=str, default='sar',
                        choices=['sar', 'ue'],
                        help='Uncertainty type')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['adam', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--split_size', type=int, default=3000,
                        help='Split size')
    args = parser.parse_args()
    
    print("Working on model: ", model_path_dict[args.model], " Dataset: ", args.dataset)

    return args


def main():
    config = parse_args()
    llm = LLMs(config.model)
    dim = 4096 if 'qwen' not in config.model else 3584
    results = []
    for layer in range(llm.num_layers):
        print(f"Training on layer {layer}")
        dataset, file_name = generate_dataset(llm, config.dataset, layer_id=layer, save=False)
        train_dataset, validation_dataset = split_dataset(dataset, config.split_size)
        trainer = Trainer(config.model_name, dim=dim)
        results.append(trainer.train_supervised(train_dataset, validation_dataset, config))

    
    result_path = os.path.join('./results', file_name+'.pkl') 
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
        print(f"Results saved to {result_path}")
    print(results)

if __name__ == "__main__":
    fix_seed(42)
    main()