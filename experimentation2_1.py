"""
Filename: experimentation2_1.py
Author: SPTAU
"""
import matplotlib.pyplot as plt
import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 0
b = 0


def forward(x):
    """forward

    Returns: x * w + b
    """

    return x * w + b


def loss(x, y):
    """loss

    Returns: (y_pred - y) * (y_pred - y)
    """
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []
mse_list = []
MSE = []
for w in np.arange(-4.0, 4.1, 0.1):
    mse_list = []
    b_list = []
    for b in np.arange(-4.0, 4.1, 0.1):
        # print("w=", weight)
        # print("b=", bias)
        loss_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            loss_sum += loss_val
            print("\t", x_val, y_val, y_pred_val, loss_val)
        mse = loss_sum / len(x_data)
        # print("MSE=", mse)
        b_list.append(b)
        mse_list.append(mse)
    w_list.append(w)
    MSE.append(mse_list)

MSE = np.array(MSE)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
W, B = np.meshgrid(w_list, b_list)
ax.plot_surface(W, B, MSE)
ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("MSE")
plt.show()
