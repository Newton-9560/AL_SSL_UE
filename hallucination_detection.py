from utils import parse_args, fix_seed
from find_best_layer import best_layer_dict
from llm_models.models import LLMs
from hidden_state.generate import generate_dataset
from hidden_state.utils import split_dataset
from trainer import Trainer

def main():
    config = parse_args()
    fix_seed(config.seed)
    
    # Find the best layer
    best_layer = best_layer_dict[config.model]
    print(f'The best layer is {best_layer}')
    
    # Generate dataset
    llm = LLMs(config.model)
    dataset, file_name = generate_dataset(llm, config.dataset, layer_id=best_layer, save=True)
    train_dataset, validation_dataset = split_dataset(dataset, int(config.split_portion*len(dataset)))
    labeled_dataset, unlabeled_dataset = split_dataset(train_dataset, config.initial_labeled_size)
    print(f'The number of labeled data is {len(labeled_dataset)}')
    print(f'The number of unlabeled data is {len(unlabeled_dataset)}')
    print(f'The number of validation data is {len(validation_dataset)}')
    
    # Initialize Trainer
    trainer = Trainer(config.model_name, dim=4096 if 'qwen' not in config.model else 3584)
    trainer.init_dataset(labeled_dataset, unlabeled_dataset, validation_dataset)
    # Train the model
    results = trainer.train_semi_supervised(config)
    print(results)

if __name__ == "__main__":
    main()