import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


from utils import parse_args, fix_seed, calculate_auroc
from find_best_layer import best_layer_dict
from llm_models.models import LLMs
from hidden_state.generate import generate_dataset
from hidden_state.utils import split_dataset
from trainer import Trainer
import numpy as np

from initial_data_selection.k_means import split_by_kmeans_multiple_without_pca, split_by_kmeans_with_pca_multiple

def main(model, model_size, dataset, uncertainty_type):
    fix_seed(40)
    
    # Find the best layer
    best_layer = best_layer_dict[model]
    print(f'The best layer is {best_layer}')
    
    # Generate dataset
    llm = LLMs(model, model_size)
    dataset, file_name = generate_dataset(llm, dataset, layer_id=best_layer, save=True)
    train_dataset, validation_dataset = split_dataset(dataset, int(0.8*len(dataset)), balance=False)
    print(f"The AUROC of the baseline method {uncertainty_type} is {calculate_auroc(train_dataset, uncertainty_type)}")
    labeled_dataset, unlabeled_dataset = split_dataset(train_dataset, 96, balance=False)

    portion = np.mean([i['align']>0.5 for i in labeled_dataset])
    
    labels = np.array([i['align']>0.5 for i in unlabeled_dataset])
    uncertainty1 = np.array([-i[uncertainty_type] for i in unlabeled_dataset])
    # assign_labels_by_uncertainty_2(uncertainty1, uncertainty2, labels, high_threshold=0.05, low_threshold=0.05)
    return assign_labels_by_uncertainty(uncertainty1, labels, high_threshold=1-portion, low_threshold=1-portion)[1]
    
    
def assign_labels_by_uncertainty(uncertainty_values, true_labels, high_threshold=0.4, low_threshold=0.4):
    # Convert to numpy array
    uncertainty_values = np.array(uncertainty_values)
    true_labels = np.array(true_labels)

    # Determine threshold indices
    num_high = int(len(uncertainty_values) * high_threshold)
    num_low = int(len(uncertainty_values) * low_threshold)

    # Get sorted indices in ascending order
    sorted_indices = np.argsort(uncertainty_values)

    # Assign labels
    assigned_labels = np.full_like(true_labels, -1)  # Initialize with -1 (unassigned)
    assigned_labels[sorted_indices[:num_low]] = 0   # Lowest 40% assigned False (0)
    assigned_labels[sorted_indices[-num_high:]] = 1 # Highest 30% assigned True (1)

    # Compute accuracy (only on assigned labels)
    mask = assigned_labels != -1  # Consider only assigned labels
    accuracy = np.mean(assigned_labels[mask] == true_labels[mask])

    return assigned_labels, accuracy
    
if __name__ == '__main__':
    import pandas as pd
    from utils import parse_args, fix_seed, calculate_auroc
    from find_best_layer import best_layer_dict
    from llm_models.models import LLMs
    from hidden_state.generate import generate_dataset
    from hidden_state.utils import split_dataset
    
    # Create a list to store results
    uncertainty_score_only_list = []
    
    # Create a list to store detailed results for the DataFrame
    results_data = []
    
    def run_and_store_result(model, model_size, dataset, uncertainty_type):
        try:
            accuracy = main(model, model_size, dataset, uncertainty_type)
            # Store detailed result for DataFrame
            results_data.append({
                'model': model,
                'dataset': dataset,
                'uncertainty_type': uncertainty_type,
                'accuracy': accuracy
            })
            return accuracy
        except Exception as e:
            print(f"Error with {model}, {dataset}, {uncertainty_type}: {e}")
            # Store error in results
            results_data.append({
                'model': model,
                'dataset': dataset,
                'uncertainty_type': uncertainty_type,
                'accuracy': None,
                'error': str(e)
            })
            return None
    model_list = {'llama3': '8', 'qwen': '7'}
    dataset_list = ['trivia_qa', 'truthful_qa', 'coqa', 'simple_qa']
    uncertainty_type_list = ['sar', 'semanticentropy', 'maximumsequenceprobability', 'lexicalsimilarity', 'montecarlosequenceentropy']
    for model in model_list:
        for dataset in dataset_list:
            for uncertainty_type in uncertainty_type_list:
                run_and_store_result(model, model_list[model], dataset, uncertainty_type)
    # Convert results_data to DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Save DataFrame to CSV
    results_df.to_csv('./uncertainty_score_only_results.csv', index=False)
    print(uncertainty_score_only_list)