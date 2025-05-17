import numpy as np
import torch
import argparse
import math
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset with model outputs')
    parser.add_argument('--model', type=str, default='mistral',
                        choices=['llama3', 'opt', 'qwen', 'mistral', 'llama3.1'],
                        help='Model name to use for generation')
    parser.add_argument('--dataset', type=str, default='trivia_qa',
                        choices=['coqa', 'trivia_qa', 'sciq', 'truthful_qa', 'tydiqa', 'squad', 'simple_qa', 'ambig_qa'],
                        help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--model_name', type=str, default='mlp',
                        help='Model name to use for training')
    parser.add_argument('--align_threshold', type=float, default=0.5,
                        help='Align threshold')
    # TODO: add uncertainty type
    parser.add_argument('--uncertainty_type', type=str, default='sar',
                        choices=['sar', 'semanticentropy', 'maximumsequenceprobability', 'lexicalsimilarity', 'montecarlosequenceentropy'],
                        help='Uncertainty type')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['adam', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--split_portion', type=float, default=0.8,
                        help='Split portion')
    parser.add_argument('--active_learning_rounds', type=int, default=0,
                        help='Number of active learning rounds')
    parser.add_argument('--active_learning_budget', type=int, default=0,
                        help='Number of active learning budget')
    parser.add_argument('--CONFIDENCE_THRESHOLD', type=float, default=0.9,
                        help='Confidence threshold for augmentation')
    parser.add_argument('--LAMBDA_U', type=float, default=1,
                        help='Lambda for unsupervised loss')
    parser.add_argument('--initial_labeled_size', type=int, default=96,
                        help='Initial labeled size')
    parser.add_argument('--pseudo_label_threshold', type=float, default=0.1,
                        help='Align threshold')
    parser.add_argument('--model_size', type=str, default='7',
                        help='Model size')
    
    args = parser.parse_args()
    
    print("Working on model: ", args.model, " Dataset: ", args.dataset)

    return args

def fix_seed(seed=40):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def calculate_auroc(result, us_metric, threshold=0.5):
    scores = []
    labels = []
    for data in result:
        value = data.get(us_metric)
        if value is not None and not math.isnan(value):
            scores.append(-value)
            labels.append(data['align'] > threshold)
    if len(set(labels)) < 2:
        print(f'{us_metric} skipped: only one class present in y_true.')
    else:
        auroc = roc_auc_score(labels, scores)
    return auroc

def sample_uniform_by_uncertainty(
    data,
    n_bins: int = 32,
    samples_per_bin: int = 1,
    value_key: str = "uncertainty_value",
):

    values = np.array([d[value_key] for d in data])
    min_v, max_v = np.min(values), np.max(values)
    bins = np.linspace(min_v, max_v, n_bins + 1)

    bin_to_items = defaultdict(list)
    for d in data:
        v = d[value_key]
        bin_idx = np.digitize(v, bins) - 1
        bin_idx = min(bin_idx, n_bins - 1)
        bin_to_items[bin_idx].append(d)

    selected = []
    for items in bin_to_items.values():
        selected.extend(random.sample(items, min(len(items), samples_per_bin)))

    selected_ids = set(id(item) for item in selected)
    remaining = [d for d in data if id(d) not in selected_ids]

    return selected, remaining

def format_delta_cell(value, delta, color="customcoral", up_arrow="\\uparrow", down_arrow="\\downarrow"):

    if delta > 0:
        arrow = up_arrow
        color = "upcolor"
    elif delta < 0:
        arrow = down_arrow
        delta = abs(delta)
        color = "downcolor"
    else:
        return str(value)
    
    # Fixed version with properly escaped braces
    return f"{value}{{\\textcolor{{{color}}}{{\\scriptsize{{${arrow}${delta}}}}}}}"