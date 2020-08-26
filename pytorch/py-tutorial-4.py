# Regression with PyTorch
# https://carpentries-incubator.github.io/deep-learning_intro/03-regression/index.html

# increase the hidden layers

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# ==============================================================================
# define the neural network
# 
# multi-layered neural network will have one hidden layer (it is still not deep learning!) so in total 3 layers:
# one input layer
# one hidden layer
# one output layer

# use Linear model for each layer and we take a ReLU (Rectified Linear Unit)
# TODO: understand ReLU 

import torch
class Model(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        
        # H is number of nerons in the hidden layer

        super().__init__()

        # input layer
        self.l1 = torch.nn.Linear(D_in, H)

        # hidden layer
        self.relu_1 = torch.nn.ReLU()

        self.relu_2 = torch.nn.ReLU()

        # output layer
        self.l2=torch.nn.Linear(H, D_out)

    def forward(self, X):
        
        # return self.l2(self.relu_2(self.relu_1(self.l1(X))))
        return self.l2(self.relu_1(self.l1(X)))

# ==============================================================================
# 64 samples
N = 640
# 1000 neurons in the input layer
D_in = 1000
# and 10 neurons in the output layer
D_out = 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# in hidden layer, define H neurons
H = 100

model = Model(D_in, H, D_out)

# loss_fn = torch.nn.MSELoss(reduction = 'sum')
loss_fn = torch.nn.MSELoss()
# TODO: understand 'reduction

# an Optimizer will update the weights of the model
# Here we will use Adam. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.

# learning rate influences the accuracy
learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(5000):

        
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # forward pass: compute predicted y by pssing x to the model
    y_pred = model(x)

    # compute and print loss
    loss = loss_fn(y_pred,y)
    print(t, loss.item())

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    if loss < 1e-5:
        break

y_last = model(x)
print(y_last - y)


# matplotlib inline
plt.figure(figsize=(5,5))
plt.plot(y.detach().numpy(), y_last.detach().numpy(), 'o')
plt.xlabel('Targeted y', fontsize=16)
plt.ylabel('Modeled y', fontsize=16)
plt.savefig('./model-validation-tutorial-4-1.jpg',dpi=300)

