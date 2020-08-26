# https://carpentries-incubator.github.io/deep-learning_intro/02-pytorch/index.html

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# ex-1
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

df = pd.read_csv('./iris.data', header=None)
print(type(df))

print(df.describe())
df.sort_values(by=1)


# ex-2
# create a perceptron with one single linear neuron and one single output
class Perceptron(torch.nn.Module):
    def __init__(self):
	    # We initialize using the base 
		# class initialization
		# super() lets you avoid referring to the base class explicitly.
        super().__init__()
		# we define a layer using Linear model
		# as we will apply a linear transformation to our inputs
		# the first parameter is the number of neurons in the input layer
		# and the second parameter is the number of outputs (here 1).
        self.fl = torch.nn.Linear(1,1)
	
    def forward(self, x):
        x = self.fl(x)
        return x

neural_network = Perceptron()
print(neural_network)

print(list(neural_network.parameters()))

input = torch.randn(1, requires_grad=True)
print(input)

out = neural_network(input)
print(out)

# TODO - understand the loss function
cost_function = torch.nn.MSELoss()
# torch.nn.MSELoss() creates a criterion that measures the mean squared error 
# (squared L2 norm) between each element in the modeled and targeted y.

# TODO - understand the optim algorithm
perceptron_optimizer = torch.optim.SGD(neural_network.parameters(), lr=0.01)
# SGD stands for Stochastic Gradient Descent and is the de-facto gradient descent method in PyTorch

# training the neural network
# giving inputs: 6 pairs
inputs = [(1.,3.), (2.,6.), (3.,9.), (4.,12.), (5.,15.), (6.,18.)]

for epoch in range(500):
    
    for i, data in enumerate(inputs):

        #(1) get the current set of input and output
        X, Y = iter(data)

        X = torch.tensor([X], requires_grad=True)
		# output does not need to have requires_grad=True

        Y = torch.tensor([Y], requires_grad=False)
		
        #(2) Initialize optimizer
        # for each input-output pair, the gradient need to be initialized
        perceptron_optimizer.zero_grad()
        
        #(3) calcualt the predicted output
        outputs = neural_network(X)
        
        #(4) calculate the cost
        cost = cost_function(outputs, Y)
        cost.backward()
        # computes dcost/dx for every parameter x which has requires_grad=True. 
        # These are accumulated into x.grad for every parameter x.
        # x.grad += dloss/dx

        # All optimizers implement a step() method, that updates the parameters
        perceptron_optimizer.step()
        # optimizer.step updates the value of x using the gradient x.grad. 
        # For example, the SGD optimizer performs:
        # x += -lr * x.grad

        if (i % 10 == 0):
            print("Epoch {} - loss: {}".format(epoch, cost.item()))



