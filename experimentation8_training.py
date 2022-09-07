"""
Filename: experimentation8_training.py
Author: SPTAU
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# 派生 CSV_Dataset 类
class TitanicDataset(Dataset):
    def __init__(self, filepath):
        xy = pd.read_csv(filepath, sep=",", dtype="float32")
        self.len = xy.shape[0]
        self.x_data = torch.tensor(xy.iloc[:, 1:].values)
        self.y_data = torch.tensor(xy.iloc[:, 0].values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 装载数据集
dataset = TitanicDataset("./dataset/titanic/processed_train.csv")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


# 构建模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Linear1 = nn.Linear(7, 22)
        self.Linear2 = nn.Linear(22, 11)
        self.Linear3 = nn.Linear(11, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.Linear1(x))
        x = self.relu(self.Linear2(x))
        x = self.sigmoid(self.Linear3(x))
        x = x.squeeze(-1)
        return x


model = Model()

criterion = nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 开始训练
acc_list = []

for epoch in range(300):

    acc_index = 0

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data

        label_pred = model(inputs)
        loss = criterion(label_pred, labels)

        for j in range(len(label_pred)):
            if round(label_pred[i - 1].item()) == labels[i - 1].item():
                acc_index += 1

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    acc = acc_index / len(dataset)
    acc_list.append(acc)
    print("epoch:", epoch, " acc:", acc)

fig, ax = plt.subplots()
ax.plot(np.arange(300), acc_list)
plt.show()
