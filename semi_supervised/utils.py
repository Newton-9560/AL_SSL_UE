import torch
import numpy as np

def calculate_alignment_score(unlabeled_dataset, classifier_output, uncertainty_type='sar'):
    # classifier_output_bce = binary_cross_entropy(classifier_output)
    
    # uncertainty_score = [-d[uncertainty_type] for d in unlabeled_dataset]
    # uncertainty_score = normalize_data(uncertainty_score)
    # uncertainty_score_bce = binary_cross_entropy(uncertainty_score)
    # alignment_score = uncertainty_score_bce + classifier_output_bce
    # alighment_result = [{'id': d['id'], 'alignment_score': alignment_score[i], 'uncertainty_score': uncertainty_score[i], 
    #                      'uncertainty_score_bce': uncertainty_score_bce[i], 'classifier_output_bce': classifier_output_bce[i],
    #                      'classifier_output': classifier_output[i]} for i, d in enumerate(unlabeled_dataset)]
    # return alighment_result
    uncertainty_score = [-d[uncertainty_type] for d in unlabeled_dataset]
    uncertainty_score = normalize_data(uncertainty_score)
    
    alignment_score_multiply = [uncertainty_score[i] * classifier_output[i] for i in range(len(uncertainty_score))]
    # alignment_score_multiply = [(uncertainty_score[i] + classifier_output[i])/2 for i in range(len(uncertainty_score))]
    alignment_score_numtiply_bce = binary_cross_entropy(alignment_score_multiply)
    
    alignment_result = [{'id': unlabeled_dataset[i]['id'], 'alignment_score': alignment_score_numtiply_bce[i]} for i in range(len(unlabeled_dataset))]
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