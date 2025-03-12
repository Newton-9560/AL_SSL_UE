import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from hidden_state.dataset import HiddenStateDataset
from semi_supervised.utils import calculate_alignment_score, weak_augmentation, strong_augmentation

class Trainer:
    def __init__(self, model_name='mlp', dim=4096):
        if model_name == 'mlp':
            self.model = MLP(dim=dim).cuda()
        else:
            raise ValueError(f'Model {model_name} not supported')
        
    def train_supervised(self, train_dataset, validation_dataset, config):
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
                
        return best
    
    def train_semi_supervised(self, labeled_dataset, unlabeled_dataset, validation_dataset, config):
        best = {'auroc': 0}
        # 1. Active learning round
        for _ in range(config.rounds):
            # 2. Load the best checkpoint from previous step, or the optimizer
            # 3. Add new labeled samples according to ACTIVE LEARNING FUNCTION
            for _ in range(config.epochs):
                # 3. Assign Peusdo label to unlabeled samples (copy the current dataset, do not change the original data)
                self.assign_pseudo_label(unlabeled_dataset, config)
                # 4. Train one epoch using function train_one_epoch_semi_supervised
                self.train_one_epoch_semi_supervised(labeled_dataset, unlabeled_dataset, config)
                # 5. Evaluate the auroc of the model
                predictions, labels, total_loss = self.validate(validation_dataset, config)
                metrics = self.cal_metrics(predictions, labels, total_loss)
                if metrics['auroc'] > best['auroc']:
                    best = metrics
                # 6. Save the best model
                
        return best
    
    def assign_pseudo_label(self, unlabeled_dataset, config):
        # TODO: refresh every epoch or every active learning round?
        outputs_list = self.get_model_output(unlabeled_dataset, config.align_threshold, config.uncertainty_type)
        
        alignment_score = calculate_alignment_score(unlabeled_dataset, outputs_list)
        # TODO: assign pseudo label to the unlabeled dataset according to alignment_score
    
    def train_one_epoch_semi_supervised(self, labeled_dataset, unlabeled_dataset, config):
        print(f'The labeled dataset has {len(labeled_dataset)} samples')
        print(f'The unlabeled dataset has {len(unlabeled_dataset)} samples')
        
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
        
    def activa_learning_update(self, labeled_dataset, unlabeled_dataset, config):
        outputs_list = self.get_model_output(unlabeled_dataset, config.align_threshold, config.uncertainty_type)
        alignment_score = calculate_alignment_score(unlabeled_dataset, outputs_list)
        # TODO: assign True label to the unlabeled dataset according to alignment_score
    
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
    
    def load_checkpoint():
        pass
    
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