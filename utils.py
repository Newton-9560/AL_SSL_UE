import numpy as np
import torch
import argparse

from llm_models.models import model_path_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset with model outputs')
    parser.add_argument('--model', type=str, default='opt',
                        choices=['llama3', 'opt', 'qwen', 'llama2-13b'],
                        help='Model name to use for generation')
    parser.add_argument('--dataset', type=str, default='truthful_qa',
                        choices=['coqa', 'trivia_qa', 'sciq', 'truthful_qa', 'tydiqa'],
                        help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--model_name', type=str, default='mlp',
                        help='Model name to use for training')
    parser.add_argument('--align_threshold', type=float, default=0.5,
                        help='Align threshold')
    # TODO: add uncertainty type
    parser.add_argument('--uncertainty_type', type=str, default='sar',
                        choices=['sar', 'semanticentropy', 'maximumsequenceprobability'],
                        help='Uncertainty type')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['adam', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--split_portion', type=float, default=0.8,
                        help='Split portion')
    parser.add_argument('--active_learning_rounds', type=int, default=1,
                        help='Number of active learning rounds')
    parser.add_argument('--active_learning_budget', type=int, default=75,
                        help='Number of active learning budget')
    parser.add_argument('--CONFIDENCE_THRESHOLD', type=float, default=0.9,
                        help='Confidence threshold for augmentation')
    parser.add_argument('--LAMBDA_U', type=float, default=0.8,
                        help='Lambda for unsupervised loss')
    parser.add_argument('--initial_labeled_size', type=int, default=200,
                        help='Initial labeled size')
    parser.add_argument('--pseudo_label_threshold', type=float, default=0.1,
                        help='Align threshold')
    
    args = parser.parse_args()
    
    print("Working on model: ", model_path_dict[args.model], " Dataset: ", args.dataset)

    return args

def fix_seed(seed=42):
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