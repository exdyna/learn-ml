# this piece of code is discarded

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# TODO - understand batch size
# If you have 10 samples or examples in a batch, 
# then the batch size is 10. Maximum batch_size is limited by 
# the memory that your system has -- main memory in case of 
# CPU and GPU memory if you are using the GPU.
# batch size = the number of training examples in one forward/backward pass
# Batch size is a term used in machine learning and 
# refers to the number of training examples utilized in one 
# iteration. The batch size can be one of three options: ... 
# Usually, a number that can be divided into the total dataset size. 
# stochastic mode: where the batch size is equal to one.

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# TODO - what is learning rate?
learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2