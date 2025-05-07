import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

from hidden_state.dataset import HiddenStateDataset
from semi_supervised.utils import calculate_alignment_score, weak_augmentation, strong_augmentation, delete_items

class Trainer:
    def __init__(self, model_name='mlp', dim=4096):
        self.dim = dim
        self.init_model(model_name)
        
        self.init_accuracy_list()
        
    def init_dataset(self, labeled_dataset, unlabeled_dataset, validation_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.validation_dataset = validation_dataset
        
    def init_model(self, model_name='mlp'):
        if model_name == 'mlp':
            self.model = MLP(dim=self.dim).cuda()
        else:
            raise ValueError(f'Model {model_name} not supported')
        
    def init_accuracy_list(self):
        self.accuracy_list_total = []
    
    '''
    Training the model using active learning and semi-supervised learning
    '''
    def train_semi_supervised(self, config):
        auroc_list = []
        # 1. Active learning round
        for round in range(config.active_learning_rounds+1):
            self.init_model()
            self.set_optimizer(config)
            self.set_criterion()
            self.init_accuracy_list()
            best = {'auroc': 0}
            
            # print(f'The active learning round {round} starts')
            # print(f'The labeled dataset has {len(self.labeled_dataset)} samples')
            # print(f'The unlabeled dataset has {len(self.unlabeled_dataset)} samples')
            # Print label distribution in labeled dataset
            # align_scores = [data['align'] for data in self.labeled_dataset]
            # positive_samples = sum(1 for score in align_scores if score >= config.align_threshold)
            # negative_samples = len(align_scores) - positive_samples
            
            # print(f'Label distribution in labeled dataset:')
            # print(f'Positive samples (align >= {config.align_threshold}): {positive_samples}')
            # print(f'Negative samples (align < {config.align_threshold}): {negative_samples}')
            # print(f'Positive ratio: {positive_samples/len(align_scores):.2%}')
            labeled_dataset = self.labeled_dataset.copy()
            unlabeled_dataset = self.unlabeled_dataset.copy()
            for epoch in range(config.epochs):
                
                # 3. Train one epoch using function train_one_epoch_semi_supervised
                self.train_one_epoch_semi_supervised(labeled_dataset, unlabeled_dataset, config)
                # 4. Evaluate the auroc of the model
                predictions, labels, total_loss = self.validate(self.validation_dataset, config)
                metrics = self.cal_metrics(predictions, labels, total_loss)
                if metrics['auroc'] > best['auroc']:
                    # print(f'The best auroc is {metrics["auroc"]}')
                    best = metrics
                # 5. Assign Peusdo label to unlabeled samples (copy the current dataset, do not change the original data)
                labeled_dataset, unlabeled_dataset = self.assign_pseudo_label(labeled_dataset, unlabeled_dataset, config)
                # 6. Save the best model
                if epoch == 50:
                    self.save_checkpoint(os.path.join('./checkpoints', config.model + '_' + config.dataset + '.pth'))
                # print(np.mean([i['align'] for i in labeled_dataset]))
                self.labeled_dataset_cache = labeled_dataset
                if len(labeled_dataset) > 3000:
                    return None
            print(f'The best auroc is {best["auroc"]}')
            auroc_list.append(best['auroc'])
            
            if round < config.active_learning_rounds:
                self.activa_learning_update(config)
        return best, auroc_list
        
    def assign_pseudo_label(self, labeled_dataset, unlabeled_dataset, config):
        
        outputs_list = self.get_model_output(unlabeled_dataset, config.align_threshold, config.uncertainty_type)
        
        alignment_result = calculate_alignment_score(unlabeled_dataset, outputs_list, config.uncertainty_type, type='ranking')
        
        # Create mapping for O(1) lookup
        id_to_index = {item['id']: i for i, item in enumerate(unlabeled_dataset)}
        
        # First pass to identify all candidates that meet the threshold
        label0_candidates = []
        label1_candidates = []
        
        for item in alignment_result:
            if item['alignment_score'] <= config.pseudo_label_threshold:
                item_id = item['id']
                index = id_to_index[item_id]
                data_item = unlabeled_dataset[index]
                
                # Get the model output
                mlp_output = outputs_list[index]
                
                # Assign pseudo label
                pseudo_label = 1 if mlp_output >= 0.9 else 0
                
                # Track correctness for evaluation
                is_correct = pseudo_label == (data_item['align'] >= config.align_threshold)
                
                # Store relevant data
                candidate_info = {
                    'id': item_id,
                    'index': index,
                    'mlp_output': mlp_output,
                    'pseudo_label': pseudo_label,
                    'is_correct': is_correct
                }
                
                # Separate by label
                if pseudo_label == 0:
                    label0_candidates.append(candidate_info)
                else:
                    label1_candidates.append(candidate_info)
        
        # Sort candidates by confidence
        # For label 0, smaller mlp_output means higher confidence
        label0_candidates.sort(key=lambda x: x['mlp_output'])  
        # For label 1, larger mlp_output means higher confidence
        label1_candidates.sort(key=lambda x: x['mlp_output'], reverse=True)
        
        # Calculate total number of candidates
        total_candidates = len(label0_candidates) + len(label1_candidates)
        if total_candidates == 0:
            return labeled_dataset, unlabeled_dataset
            
        # Determine how many of each label to select based on theta
        # Check which ratio is limiting (label0 or label1)
        ratio_label1 = config.theta
        ratio_label0 = 1 - ratio_label1
        
        # If we have more label1 than the desired ratio would allow:
        if len(label1_candidates) / total_candidates > ratio_label1 and ratio_label0 > 0:
            # We're limited by the number of label0 candidates
            num_label0 = len(label0_candidates)
            num_label1 = int(num_label0 * (ratio_label1 / ratio_label0))
            num_label1 = min(num_label1, len(label1_candidates))
            num_label1 = max(num_label1, 20) 
        # If we have more label0 than the desired ratio would allow:
        elif len(label0_candidates) / total_candidates > ratio_label0 and ratio_label1 > 0:
            # We're limited by the number of label1 candidates
            num_label1 = len(label1_candidates)
            num_label0 = int(num_label1 * (ratio_label0 / ratio_label1))
            num_label0 = min(num_label0, len(label0_candidates))
        # If the current distribution is already within the desired ratio:
        else:
            num_label0 = len(label0_candidates)
            num_label1 = len(label1_candidates)
        
        # Select the most confident samples
        selected_label0_candidates = label0_candidates[:num_label0]
        selected_label1_candidates = label1_candidates[:num_label1]
        
        # Prepare the selected data and track which indices to delete
        selected_data = []
        pseudo_labels_correctness = []
        indices_to_delete = []
        
        # Process label 0 candidates
        for candidate in selected_label0_candidates:
            index = candidate['index']
            data_item = unlabeled_dataset[index].copy()
            data_item['align'] = candidate['pseudo_label']
            selected_data.append(data_item)
            pseudo_labels_correctness.append(candidate['is_correct'])
            indices_to_delete.append(index)
        
        # Process selected label 1 candidates
        for candidate in selected_label1_candidates:
            index = candidate['index']
            data_item = unlabeled_dataset[index].copy()
            data_item['align'] = candidate['pseudo_label']
            selected_data.append(data_item)
            pseudo_labels_correctness.append(candidate['is_correct'])
            indices_to_delete.append(index)
        
        # Update accuracy tracking
        self.accuracy_list_total.extend(pseudo_labels_correctness)
        
        # Update labeled dataset
        labeled_dataset.extend(selected_data)
        
        # Filter unlabeled dataset if necessary
        if len(unlabeled_dataset) > 32:
            # Sort indices in descending order to avoid index shifting issues
            indices_to_delete.sort(reverse=True)
            for index in indices_to_delete:
                unlabeled_dataset.pop(index)
        
        return labeled_dataset, unlabeled_dataset
                
    
    def activa_learning_update(self, config):
        self.load_checkpoint(os.path.join('./checkpoints', config.model + '_' + config.dataset + '.pth'))
        
        outputs_list = self.get_model_output(self.unlabeled_dataset, config.align_threshold, config.uncertainty_type)
        alignment_score = calculate_alignment_score(self.unlabeled_dataset, outputs_list, config.uncertainty_type, type='ranking_difference')
        budget = int(config.active_learning_budget/config.active_learning_rounds)
        
        alignment_score_sorted = np.argsort([i['alignment_score'] for i in alignment_score])
        selected_idx = [alignment_score[i]['id'] for i in alignment_score_sorted[-budget:]]
        
        selected_data = [i for i in self.unlabeled_dataset if i['id'] in selected_idx]
        
        self.labeled_dataset.extend(selected_data)
        self.unlabeled_dataset = delete_items(self.unlabeled_dataset, selected_idx)
    
    def train_one_epoch_semi_supervised(self, labeled_dataset, unlabeled_dataset, config):
        # print(f'The labeled dataset has {len(labeled_dataset)} samples')
        # print(f'The unlabeled dataset has {len(unlabeled_dataset)} samples')
        
        labeled_dataset = HiddenStateDataset(labeled_dataset, threshold=config.align_threshold, uncertainty_type=config.uncertainty_type)
        unlabeled_dataset = HiddenStateDataset(unlabeled_dataset, threshold=config.align_threshold, uncertainty_type=config.uncertainty_type)
        
        labeled_loader = DataLoader(labeled_dataset, batch_size=config.batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.batch_size, shuffle=True)
        
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        total_loss = 0.0
        total_supervised_loss = 0.0
        total_unsupervised_loss = 0.0
        
        l = max(len(unlabeled_loader), len(labeled_loader))
        # l = len(labeled_loader)
        for batch_idx in range(l):
            try:
                labeled_data = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_data = next(labeled_iter)
            
            try:
                unlabeled_data = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_data = next(unlabeled_iter)
            
            # Labeled data
            sample_id, labeled_hs, labeled_labels, ue = labeled_data
            labeled_hs, labeled_labels = labeled_hs.cuda().float(), labeled_labels.cuda().float()
            
            # Unlabeled data
            sample_id, unlabeled_hs, _, ue = unlabeled_data
            unlabeled_hs = unlabeled_hs.cuda().float()
            
            # Weak augmentation for unlabeled data
            weak_augmented = weak_augmentation(unlabeled_hs)
            
            # Forward pass for weakly augmented unlabeled data
            with torch.no_grad():
                weak_logits = self.model(weak_augmented).reshape(-1)
                weak_probs = torch.sigmoid(weak_logits)
                pseudo_labels = (weak_probs > config.CONFIDENCE_THRESHOLD).float()
                # TODO: maybe should delete this? 
                pseudo_labels[weak_probs < (1 - config.CONFIDENCE_THRESHOLD)] = 0

                mask = ((weak_probs > config.CONFIDENCE_THRESHOLD) | (weak_probs < (1 - config.CONFIDENCE_THRESHOLD))).float()
                # pseudo_labels = (weak_probs > CONFIDENCE_THRESHOLD).float()
                # mask = (weak_probs > CONFIDENCE_THRESHOLD).float()

            # data_consistency, data_inconsistency, label_consistency, label_inconsistency = strong_augmentation_llm(input_text, output_text, pseudo_labels, mask)
            strong_augmented = strong_augmentation(unlabeled_hs)
            
            labeled_logits =self.model(labeled_hs).reshape(-1)
            strong_logits = self.model(strong_augmented).reshape(-1)
            
            supervised_loss = self.criterion(labeled_logits, labeled_labels)
            
            unsupervised_loss = (self.criterion(strong_logits, pseudo_labels) * mask).mean()
            
            # Total loss
            loss = supervised_loss + config.LAMBDA_U * unsupervised_loss
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def get_model_output(self, dataset, threshold, uncertainty_type):
        dataset = HiddenStateDataset(dataset, threshold=threshold, uncertainty_type=uncertainty_type)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        outputs_list = []
        with torch.no_grad():
            for sample_id, hs, labels, ue in data_loader:
                hs = hs.cuda().float()
                outputs = self.model(hs).reshape(-1)
                outputs_list.append(torch.sigmoid(outputs).cpu().item())
        return outputs_list
    
    
    '''
    Training the model on all the dataset
    '''
    def train_supervised(self, train_dataset, validation_dataset, config):
        self.init_model()

        print(f'The train dataset has {len(train_dataset)} samples')
        print(f'The validation dataset has {len(validation_dataset)} samples')
        
        self.set_optimizer(config)
        self.set_criterion()

        
        train_dataset = HiddenStateDataset(train_dataset, threshold=config.align_threshold, uncertainty_type=config.uncertainty_type)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        best = {'auroc': 0}
        for epoch in range(config.epochs):
            self.train_one_epoch(train_loader)
            predictions, labels, total_loss = self.validate(validation_dataset, config)
            metrics = self.cal_metrics(predictions, labels, total_loss)

            if metrics['auroc'] > best['auroc']:
                best = metrics
                self.save_checkpoint(os.path.join('./checkpoints', config.model + '_' + config.dataset +'.pth'))
                # print('Save the best model to: ', os.path.join('./checkpoints', config.model + '_' + config.dataset +'.pth'))
                
        return best
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch_idx, (sample_id, hs, label, ue) in enumerate(train_loader):
            self.optimizer.zero_grad()
            hs = hs.cuda().float()
            label = label.cuda().float()
            
            outputs = self.model(hs).reshape(-1)
            loss = self.criterion(outputs, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
        
    def validate(self, validation_dataset, config):
        validation_dataset = HiddenStateDataset(validation_dataset, threshold=config.align_threshold, uncertainty_type=config.uncertainty_type)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
            
        self.model.eval()
        total_loss = 0.0
        predictions = []
        labels = []
        with torch.no_grad():
            for batch_idx, (sample_id, hs, label, ue) in enumerate(validation_loader):
                hs = hs.cuda().float()
                label = label.cuda().float()
                
                outputs = self.model(hs).reshape(-1)
                loss = self.criterion(outputs, label)
                total_loss += loss.item()
                
                pred = torch.sigmoid(outputs)
                predictions.extend(pred.cpu().flatten().tolist())
                labels.extend(label.cpu().flatten().tolist())
                
        return predictions, labels, total_loss
              
                
    def set_optimizer(self, config):
        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError(f"Optimizer {config.optimizer} not supported")

    def set_criterion(self):
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
    
    # TODO: select the best threshold for the model
    def cal_metrics(self, predictions, labels, total_loss, threshold=0.5):
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        accuracy_positive = np.mean([score >= threshold for score, label in zip(predictions, labels) if label == 1])
        accuracy_negative = np.mean([score < threshold for score, label in zip(predictions, labels) if label == 0])
        accuracy = np.mean((predictions >= threshold) == labels)

        auroc = roc_auc_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'accuracy_positive': accuracy_positive,
            'accuracy_negative': accuracy_negative,
            'total_loss': total_loss,
            'auroc': auroc
        }
    

class MLP(torch.nn.Module):
    def __init__(self, dim=4096):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(dim, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Linear(512, 1)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                x = self.layer1(x)
                x = self.layer2(x)
                e = x
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            e = x
        out = self.layer3(e)
        if last:
            return out, e
        else:
            return out
        
        
def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True