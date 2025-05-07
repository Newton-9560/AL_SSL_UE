import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils import parse_args, fix_seed, calculate_auroc
from find_best_layer import best_layer_dict
from llm_models.models import LLMs
from hidden_state.generate import generate_dataset
from hidden_state.utils import split_dataset
from trainer import Trainer
import numpy as np

from initial_data_selection.k_means import split_by_kmeans_multiple_without_pca, split_by_kmeans_with_pca_multiple

def main(config, unlabeled_size=None):
    fix_seed(config.seed)
    
    # Find the best layer
    best_layer = best_layer_dict[config.model]
    print(f'The best layer is {best_layer}')
    
    # Generate dataset
    llm = LLMs(config.model, config.model_size)
    dataset, file_name = generate_dataset(llm, config.dataset, layer_id=best_layer, save=True)
    train_dataset, validation_dataset = split_dataset(dataset, int(config.split_portion*len(dataset)), balance=False)
    print(f"The AUROC of the baseline method {config.uncertainty_type} is {calculate_auroc(train_dataset, config.uncertainty_type)}")
    # labeled_dataset, unlabeled_dataset = split_by_kmeans_multiple_without_pca(train_dataset, n_clusters=48, n_select=config.initial_labeled_size)
    # labeled_dataset, unlabeled_dataset = split_by_kmeans_with_pca_multiple(train_dataset, n_clusters=config.initial_labeled_size, n_select=config.initial_labeled_size, pca_dim=256)
    labeled_dataset, unlabeled_dataset = split_dataset(train_dataset, config.initial_labeled_size, balance=False)
    if unlabeled_size is not None:
        unlabeled_dataset = unlabeled_dataset[:unlabeled_size]
    print('The disturbution of the align score of the labeled dataset is', np.mean([i['align']>config.align_threshold for i in labeled_dataset]))
    config.theta = np.mean([i['align']>config.align_threshold for i in labeled_dataset])
    # labeled_dataset, unlabeled_dataset = sample_uniform_by_uncertainty(train_dataset, n_bins=16, samples_per_bin=8, value_key=config.uncertainty_type)
    print(f'The number of labeled data is {len(labeled_dataset)}')
    print(f'The number of unlabeled data is {len(unlabeled_dataset)}')
    
    # Initialize Trainer
    trainer = Trainer(config.model_name, dim=4096 if 'qwen' not in config.model else 3584)
    trainer.init_dataset(labeled_dataset, unlabeled_dataset, validation_dataset)
    # Train the model
    results = trainer.train_semi_supervised(config)
    llm.delete_model()
    print('The accuracy of the pseudo-labeled dataset is', np.mean(trainer.accuracy_list_total[:2500]))
    result = trainer.train_supervised(trainer.labeled_dataset_cache, validation_dataset, config)
    return result    
    
def multiple_runs():
    models_name = ['llama3']
    datasets_name = ['truthful_qa']
    uncertainty_types = ['lexicalsimilarity', 'semanticentropy', 'maximumsequenceprobability', 'montecarlosequenceentropy', 'sar']
    for model_name in models_name:
        for dataset_name in datasets_name:
            for uncertainty_type in uncertainty_types:
                config = parse_args()
                config.model = model_name
                config.dataset = dataset_name
                config.uncertainty_type = uncertainty_type
                try:
                    results = main(config)
                    print('#'*50)
                    print(f'The auroc of {model_name} on {dataset_name} with {uncertainty_type} is {results["auroc"]}')
                except Exception as e:
                    print(f'Error: {e}')
                    print(f'The model {model_name} on {dataset_name} with {uncertainty_type} is not working')
                    
            
def auroc_with_initial_labeled_size(config):
    config.epochs = 50
    labeled_sizes = {'16':8, '32':16, '64':32, '96':32, '128':32, '256':32}
    models_name = ['qwen']
    uncertainty_types = ['sar']
    datasets_name = ['trivia_qa', 'truthful_qa', 'coqa', 'simple_qa']
    results = []
    for model_name in models_name:
        for uncertainty_type in uncertainty_types:
            for dataset_name in datasets_name:
                for labeled_size in labeled_sizes.keys():
                    config.model = model_name
                    config.uncertainty_type = uncertainty_type
                    config.dataset = dataset_name
                    config.initial_labeled_size = int(labeled_size)
                    config.batch_size = labeled_sizes[labeled_size]
                    try:
                        auroc = main(config)['auroc']
                    except Exception as e:
                        print(f'Error: {e}')
                        print(f'The model {model_name} on {dataset_name} with {uncertainty_type} is not working')
                        auroc = None
                    results.append(
                        {
                            'model': model_name,
                            'uncertainty_type': uncertainty_type,
                            'dataset': dataset_name,
                            'initial_labeled_size': labeled_size,
                            'auroc': auroc
                        }
                    )
                    print(results[-1])
    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('results_with_num_qwen.csv', index=False)
        
def auroc_with_unlabeled_size(config):
    unlabeled_sizes = range(100, 4000, 100)
    models_name = ['llama3']
    uncertainty_types = ['sar', 'semanticentropy', 'maximumsequenceprobability', 'lexicalsimilarity', 'montecarlosequenceentropy']
    datasets_name = ['trivia_qa', 'coqa']
    results = []
    for model_name in models_name:
        for uncertainty_type in uncertainty_types:
            for dataset_name in datasets_name:
                for unlabeled_size in unlabeled_sizes:
                    config.model = model_name
                    config.uncertainty_type = uncertainty_type
                    config.dataset = dataset_name
                    try:
                        auroc = main(config, unlabeled_size)['auroc']
                    except Exception as e:
                        print(f'Error: {e}')
                        print(f'The model {model_name} on {dataset_name} with {uncertainty_type} is not working')
                        auroc = None
                    results.append(
                        {
                            'model': model_name,
                            'uncertainty_type': uncertainty_type,
                            'dataset': dataset_name,
                            'unlabeled_size': unlabeled_size,
                            'auroc': auroc
                        }
                    )
                    print(results[-1])
    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('results_with_num_unlabeled_llama3.csv', index=False)

if __name__ == "__main__":
    config = parse_args()
    auroc_with_unlabeled_size(config)
    # print(main(config))