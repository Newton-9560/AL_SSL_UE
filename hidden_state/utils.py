import os
import numpy as np

ue_name_list = ['montecarlosequenceentropy', 'lexicalsimilarity', 'semanticentropy', 'maximumsequenceprobability', 'sar']
model_name_list = ['opt', 'llama3', 'qwen']
dataset_name_list = ['coqa', 'trivia_qa', 'sciq']

def load_generated_result(model_name, dataset_name, root_path='/home/hanwenli/work/2025/AL_SSL/data'):
    folder_path = os.path.join(root_path, model_name, dataset_name)
    files = os.listdir(folder_path)
    for file in files:
        data = np.array([])
        if file.endswith('.npy'):
            print("Reading file: ", file)
            data = np.load(os.path.join(folder_path, file), allow_pickle=True)
            data = np.array([
                d for d in data 
                if not any(value is None for value in d.values())
            ])
            
            # data = np.concatenate([data, i, allow_pickle=True)], axis=0)
    return data

def split_dataset(dataset: list[dict], split_num: int, balance=False):
    if balance:
        align_above = [d for d in dataset if d['align'] >= 0.5]
        align_below = [d for d in dataset if d['align'] < 0.5]

        samples_per_group = split_num // 2

        above_count = min(samples_per_group, len(align_above))
        below_count = min(samples_per_group, len(align_below))

        remaining = split_num - (above_count + below_count)
        if remaining > 0:
            if len(align_above) > above_count:
                above_count += remaining
            elif len(align_below) > below_count:
                below_count += remaining

        balanced_dataset = (
            np.random.choice(align_above, above_count, replace=False).tolist() +
            np.random.choice(align_below, below_count, replace=False).tolist()
        )

        remaining_samples = [d for d in dataset if d not in balanced_dataset]
        return balanced_dataset, remaining_samples
    else:
        np.random.shuffle(dataset)
        return dataset[:split_num], dataset[split_num:]

def calculate_auroc(dataset, ue_name, threshold=0.3):
    from sklearn.metrics import roc_auc_score
    ue_score = np.array([data[ue_name] for data in dataset])
    label = np.array([data['align'] <= threshold for data in dataset])
    return roc_auc_score(label, ue_score)

if __name__ == '__main__':
    for model_name in model_name_list:
        for dataset_name in dataset_name_list:
            data = load_generated_result(model_name, dataset_name)
            for ue_name in ue_name_list:
                auroc = calculate_auroc(data, ue_name)
                print(f'{model_name} {dataset_name} {ue_name} auroc: {auroc}')
