from torch.utils.data import Dataset


class HiddenStateDataset(Dataset):
    def __init__(self, dataset: list[dict], threshold: float = 0.7, uncertainty_type: str = 'sar'):
        self.dataset = dataset
        self.threshold = threshold
        self.uncertainty_type = uncertainty_type
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        data_point = self.dataset[idx]
        label = data_point['align'] >= self.threshold
        hidden_state = data_point['hidden_state']
        id = data_point['id']
        uncertainty = data_point[self.uncertainty_type]
        
        return id, hidden_state, label, uncertainty
        
    def add_data_point(self, data_point: list[dict]):
        self.dataset.extend(data_point)
        
    def remove_data_point(self, idx: list[int]):
        self.dataset = [item for item in self.dataset if item['id'] not in idx]
        
    def get_data_point(self, idx: int):
        return next(item for item in self.dataset if item['id'] == idx)
        
        