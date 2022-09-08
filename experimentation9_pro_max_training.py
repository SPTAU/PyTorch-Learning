"""
Filename: experimentation9_pro_max_training.py
Author: SPTAU
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


def reset_weight(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


class OGPCCDataset(Dataset):
    def __init__(self, filepath):
        xy = pd.read_csv(filepath, sep=",", dtype="float32")
        self.len = xy.shape[0]
        self.x_data = torch.tensor(xy.iloc[:, :93].values)
        self.y_data = torch.tensor(xy.iloc[:, 93:].values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(93, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, 9)

    def forward(self, x):
        x = x.view(-1, 93)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)


if __name__ == "__main__":

    batch_size = 32
    num_epochs = 40
    k_folds = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    TRAIN_PATH = "./dataset/otto-group-product-classification-challenge/processed_train.csv"
    TEST_PATH = "./dataset/otto-group-product-classification-challenge/test.csv"
    train_dataset = OGPCCDataset(TRAIN_PATH)
    test_dataset = OGPCCDataset(TEST_PATH)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}

    print("--------------------------------")

    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):

        print("FOLD %d" % fold)
        print("--------------------------------")

        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4)
        test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=4)

        model = Net()
        model.apply(reset_weight)
        model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        for epoch in range(0, num_epochs):

            current_loss = 0.0

            correct, total = 0, 0
            epoch_loss_list = []
            epoch_acc_list = []

            for batch_idx, data in enumerate(train_loader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                _, predicted = torch.max(outputs.data, dim=1)
                _, labels = torch.max(targets.data, dim=1)
                total += targets.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = current_loss / batch_idx
            epoch_acc = 100 * correct / total
            epoch_loss_list.append(epoch_loss)
            epoch_acc_list.append(epoch_acc)

            print("[Epoch %3d] loss: %.3f acc: %.3f" % (epoch + 1, epoch_loss, epoch_acc))

        print("Training process has finished. Saving trained model.")

        save_path = f"./OGPCC Weight/model-fold-{fold}.pth"
        torch.save(model.state_dict(), save_path)

        print("Starting testing")

        correct, total = 0, 0
        with torch.no_grad():

            for batch_idx, data in enumerate(test_loader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(targets.data, dim=1)
                total += targets.size(0)
                correct += (predicted == labels).sum().item()

            print("Accuracy for fold %d: %.3f %%" % (fold, 100.0 * correct / total))
            print("--------------------------------")
            results[fold] = 100.0 * (correct / total)

    print("K-FOLD CROSS VALIDATION RESULTS FOR %d FOLDS" % k_folds)
    print("--------------------------------")
    sum = 0.0
    for key, value in results.items():
        print("Fold %d: %.3f %%" % (key, value))
        sum += value
    print("Average: %.3f %%" % (sum / len(results.items())))
