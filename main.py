import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(4096, 2048),
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
    
    def get_embedding_dim(self):
        return 512

def validation_loop(model, val_loader, threshold):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    predicted_label_list = []
    predicted_score_list = []
    label_list = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.cuda().float()
            labels = labels.cuda().float()  # Change to float for binary cross entropy
            outputs = model(inputs).reshape(-1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            pred_scores = torch.sigmoid(outputs)
            predicted = (pred_scores >= threshold).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_label_list += predicted.cpu().flatten().tolist()
            predicted_score_list += outputs.cpu().flatten().tolist()
            label_list += (labels >= threshold).long().cpu().flatten().tolist()
    accuracy = correct / total
    
    if label_list:
        accuracy_pos = np.mean([score >= threshold for score, label in zip(predicted_score_list, label_list) if label == 1])
        accuracy_neg = np.mean([score < threshold for score, label in zip(predicted_score_list, label_list) if label == 0])
    
    return predicted_score_list, label_list, accuracy_pos, accuracy_neg, accuracy

def train_loop(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.cuda().float()
        labels = labels.cuda().float()
        optimizer.zero_grad()
        outputs = model(inputs).reshape(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # import matplotlib.pyplot as plt

    # # Collect predictions for the entire training dataset
    # all_predictions = []
    # model.eval()
    # with torch.no_grad():
    #     for inputs, labels, _, _, _ in train_loader:
    #         inputs = inputs.cuda().float()
    #         outputs = model(inputs).reshape(-1)
    #         pred_scores = torch.sigmoid(outputs)
    #         all_predictions += pred_scores.cpu().flatten().tolist()

    # # Plot the distribution of predictions
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_predictions, bins=50, alpha=0.75, color='blue', edgecolor='black')
    # plt.title('Distribution of Training Sample Predictions')
    # plt.xlabel('Prediction Score')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig('/home/hanwenli/work/UQ_SSL/training_predictions_distribution.png')
    # plt.close()
    return running_loss / len(train_loader)

def run(model, optimizer, criterion, train_dataset, val_dataset, batch_size=32, epoch=50, threshold=0.7):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    best_roc = 0
    best_accuracy = 0
    best_accuracy_neg = 0
    best_accuracy_pos = 0
    for e in tqdm(range(epoch)):
        train_loss = train_loop(model, optimizer, criterion, train_loader)
        predicted_score_list, label_list, accuracy_pos, accuracy_neg, accuracy = validation_loop(model, val_loader, threshold)
        test_auroc = roc_auc_score(label_list, predicted_score_list)
        if test_auroc > best_roc:
            best_roc = test_auroc
            # torch.save(model.state_dict(), '/home/hanwenli/work/llm-early-exit/SSL/output/original_auroc.pth')
        if accuracy > best_accuracy:
            # torch.save(model.state_dict(), '/home/hanwenli/work/llm-early-exit/SSL/output/original_acc.pth')
            # print('Model saved!')
            best_accuracy = accuracy
            best_accuracy_pos = accuracy_pos
            best_accuracy_neg = accuracy_neg
        # if accuracy_pos > best_accuracy_pos:
        #     torch.save(model.state_dict(), '/home/hanwenli/work/llm-early-exit/SSL/output/balanced_acc_p.pth')
        #     best_accuracy_pos = accuracy_pos
        # if accuracy_neg > best_accuracy_neg:
        #     torch.save(model.state_dict(), '/home/hanwenli/work/llm-early-exit/SSL/output/balanced_acc_n.pth')
        #     best_accuracy_neg = accuracy_neg
        # print(f'[EPOCH:{epoch}]Test AUROC: {test_auroc}')
        
    print(f'The best Auroc is {best_roc:4f}')
    print(f'Best predicted accuracy: {best_accuracy}')
    print(f'Best predicted accuracy for label 1: {best_accuracy_pos}')
    print(f'Best predicted accuracy for label 0: {best_accuracy_neg}')
    return best_roc, best_accuracy


def get_split(idx, size):
    split1 = np.random.choice(idx, size=size, replace=False)
    split2 = np.setdiff1d(idx, split1)
    return split1, split2