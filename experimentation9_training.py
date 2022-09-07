"""
Filename: experimentation9_training.py
Author: SPTAU
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 派生 CSV_Dataset 类
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


TRAIN_PATH = "./dataset/otto-group-product-classification-challenge/processed_train.csv"
TEST_PATH = "./dataset/otto-group-product-classification-challenge/test.csv"
batch_size = 64
epoch = 100

# 装载数据集
train_dataset = OGPCCDataset(TRAIN_PATH)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = OGPCCDataset(TEST_PATH)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    epoch_loss_list = []
    epoch_acc_list = []
    for i in range(epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, targets = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            _, labels = torch.max(targets.data, dim=1)
            total += targets.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / batch_idx
        epoch_acc = 100 * correct / total
        epoch_loss_list.append(epoch_loss)
        epoch_acc_list.append(epoch_acc)
        print("[Epoch %3d] loss: %.3f acc: %.3f" % (i + 1, epoch_loss, epoch_acc))
    fig, ax = plt.subplots()
    ax.plot(np.arange(epoch), epoch_loss_list)
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(np.arange(epoch), epoch_acc_list)
    plt.show()


if __name__ == "__main__":
    train(epoch)
