import torch
import numpy as np

def calculate_alignment_score(unlabeled_dataset, classifier_output, uncertainty_type='sar', type='ranking'):
    #! preprocess the uncertainty score
    uncertainty_score = [-d[uncertainty_type] for d in unlabeled_dataset]
    uncertainty_score = normalize_data(uncertainty_score)
    
    #! binary cross entropy
    if type == 'bce':
        alignment_score_multiply = [uncertainty_score[i] * classifier_output[i] for i in range(len(uncertainty_score))]
        alignment_score_numtiply_bce = binary_cross_entropy(alignment_score_multiply)
        alignment_result = [{'id': unlabeled_dataset[i]['id'], 'alignment_score': alignment_score_numtiply_bce[i]} for i in range(len(unlabeled_dataset))]
    
    #! ranking difference
    elif type == 'ranking':
        sorted_ue_index = np.argsort(uncertainty_score).tolist()
        alignment_score = [((classifier_output[i] >= 0.8 and sorted_ue_index.index(i) >= 0.8*len(sorted_ue_index)) or 
                        (classifier_output[i] <= 0.2 and sorted_ue_index.index(i) <= 0.2*len(sorted_ue_index)))
                        for i in range(len(uncertainty_score))]
        alignment_result = [{'id': unlabeled_dataset[i]['id'], 'alignment_score': 0 if alignment_score[i] else 1} for i in range(len(unlabeled_dataset))]
    elif type == 'entropy':
        classifier_output_bce = binary_cross_entropy(classifier_output)
        alignment_result = [{'id': unlabeled_dataset[i]['id'], 'alignment_score': classifier_output_bce[i]} for i in range(len(unlabeled_dataset))]
    elif type == 'ranking_difference':
        uncertainty_score = [d[uncertainty_type] for d in unlabeled_dataset]
        sorted_ue_index = np.argsort(uncertainty_score)
        sorted_classifier_output_index = np.argsort(classifier_output)
        alignment_score = [abs(sorted_ue_index[i] - sorted_classifier_output_index[i]) for i in range(len(uncertainty_score))]
        alignment_result = [{'id': unlabeled_dataset[i]['id'], 'alignment_score': alignment_score[i]} for i in range(len(unlabeled_dataset))]
    else:
        raise ValueError(f'Invalid type: {type}')
    return alignment_result


def binary_cross_entropy(x_list):
    """Calculate BCE for each value in the list."""
    x_array = np.array(x_list)
    
    # Avoid log(0) issues by adding a small epsilon
    epsilon = 1e-10
    x_array = np.clip(x_array, epsilon, 1 - epsilon)

    bce_values = - (x_array * np.log(x_array) + (1 - x_array) * np.log(1 - x_array))
    
    return bce_values

def normalize_data(data):
    """Normalize a list of numbers to the range [0,1]."""
    if not data:
        return []  # Handle empty list case
    
    min_val = min(data)
    max_val = max(data)
    
    if min_val == max_val:
        return [0.5] * len(data)  # If all values are the same, return 0.5 for all
    
    return [(x - min_val) / (max_val - min_val) for x in data]
    

# Weak augmentation
def weak_augmentation(x, noise_std=0.01, dropout_prob=0.01):
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise
    
    mask = torch.bernoulli((1 - dropout_prob) * torch.ones_like(x))
    x_aug = x_aug * mask
    
    return x_aug

# Strong augmentation
def strong_augmentation(x, crop_size=1024, shuffle_prob=0.5):
    batch_size, vector_length = x.shape
    
    start_idx = torch.randint(0, vector_length - crop_size + 1, (1,))
    x[:, start_idx:start_idx + crop_size] = 0
    
    if torch.rand(1).item() < shuffle_prob:
        idx = torch.randperm(vector_length)
        x = x[:, idx]
    
    return x

def delete_items(dataset, idx_list):
    return [item for item in dataset if item['id'] not in idx_list]

if __name__ == "__main__":
    x = [0.9]
    print(binary_cross_entropy(x))