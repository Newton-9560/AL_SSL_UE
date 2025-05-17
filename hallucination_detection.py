from utils import parse_args, fix_seed, calculate_auroc, format_delta_cell
from find_best_layer import best_layer_dict
from llm_models.models import LLMs
from hidden_state.generate import generate_dataset
from hidden_state.utils import split_dataset
from trainer import Trainer
import numpy as np



model_path_dict = {
    'meta-llama/Meta-Llama-3-8B-Instruct': 4096,
    'Qwen/Qwen2.5-7B-Instruct-1M': 3584,
    'Qwen/Qwen2.5-14B-Instruct': 5120,
    'mistralai/Mistral-7B-Instruct-v0.3': 4096,
}

def main(config, unlabeled_size=None):
    fix_seed(config.seed)
    
    # Find the best layer
    best_layer = best_layer_dict[config.model]
    # best_layer = 31
    print(f'The best layer is {best_layer}')
    
    # Generate dataset
    llm = LLMs(config.model, config.model_size, load_model=False)
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
    dim = model_path_dict[llm.model_path]
    trainer = Trainer(config.model_name, dim=dim)
    trainer.init_dataset(labeled_dataset, unlabeled_dataset, validation_dataset)
    # Train the model
    results = trainer.train_semi_supervised(config)
    llm.delete_model()
    print('The accuracy of the pseudo-labeled dataset is', np.mean(trainer.accuracy_list_total[:2500]))
    result = trainer.train_supervised(trainer.labeled_dataset_cache, validation_dataset, config)
    return result, calculate_auroc(train_dataset, config.uncertainty_type)
    
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
    results_df.to_csv('results_with_num_mistral.csv', index=False)
        
def auroc_with_unlabeled_size(config):
    unlabeled_sizes = range(100, 3200, 100)
    models_name = ['qwen']
    uncertainty_types = ['sar', 'semanticentropy', 'maximumsequenceprobability', 'lexicalsimilarity', 'montecarlosequenceentropy']
    datasets_name = ['trivia_qa']
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
    results_df.to_csv('results_with_num_unlabeled_qwen.csv', index=False)
    
    
def structured_output_auroc(config):
    models_name = [['mistral', '7'],['qwen', '14']]
    uncertainty_type = 'lexicalsimilarity'
    datasets_name = ['trivia_qa', 'coqa', 'truthful_qa', 'simple_qa']
    results_1 = '& '
    results_2 = '& '
    for model_name in models_name:
        for dataset_name in datasets_name:
            config.model = model_name[0]
            config.model_size = model_name[1]
            config.uncertainty_type = uncertainty_type
            config.dataset = dataset_name
                        
            try:
                result, base_auroc = main(config)
                result_auroc = result['auroc']
                base_auroc = round(base_auroc*100, 2)
                result_auroc = round(result_auroc*100, 2)
                diff = round(result_auroc - base_auroc, 2)
                results_1 += f'{base_auroc} & '
                results_2 += f'{format_delta_cell(result_auroc, diff)} & '
            except Exception as e:
                print(f'Error: {e}')
                print(f'The model {model_name} on {dataset_name} with {uncertainty_type} is not working')
                results_1 += f'wrong & '
                results_2 += f'wrong & '
    
    print(results_1)
    print('-'*50)
    print(results_2)

if __name__ == "__main__":
    config = parse_args()
    # auroc_with_unlabeled_size(config)
    result = main(config)
    print(result)
    print(result[0]['auroc']-result[1])
    # structured_output_auroc(config)