import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from hidden_state.dataset import HiddenStateDataset

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
        
    def train_semi_supervised(self, labeled_dataset, unlabeled_dataset, validation_dataset, config):
        pass
        
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