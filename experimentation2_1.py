"""
Filename: experimentation2_1.py
Author: SPTAU
"""
import matplotlib.pyplot as plt
import numpy as np

input_data = [1.0, 2.0, 3.0]
output_data = [2.0, 4.0, 6.0]
weight = 0
bias = 0


def forward(input):
    """forward

    Returns: input * w + b
    """

    return input * weight + bias


def loss(input, output):
    """loss

    Returns: (output_pred - output) * (output_pred - output)
    """
    output_pred = forward(input)
    return (output_pred - output) * (output_pred - output)


weight_list = []
mse_list = []
MSE = []
for weight in np.arange(-4.0, 4.1, 0.1):
    mse_list = []
    bias_list = []
    for bias in np.arange(-4.0, 4.1, 0.1):
        # print("w=", weight)
        # print("b=", bias)
        loss_sum = 0
        for input_val, output_val in zip(input_data, output_data):
            output_pred_val = forward(input_val)
            loss_val = loss(input_val, output_val)
            loss_sum += loss_val
            print("\t", input_val, output_val, output_pred_val, loss_val)
        mse = loss_sum / len(input_data)
        # print("MSE=", mse)
        bias_list.append(bias)
        mse_list.append(mse)
    weight_list.append(weight)
    MSE.append(mse_list)

MSE = np.array(MSE)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
Weight, Bias = np.meshgrid(weight_list, bias_list)
ax.plot_surface(Weight, Bias, MSE)
ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("MSE")
plt.show()
