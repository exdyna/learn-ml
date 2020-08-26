# Regression with PyTorch
# https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
# matplotlib inline

import numpy as np
import imageio


# this is one way to define a network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer

        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):

        # TODO: understand activation function
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output

        return x

# sets the random seed from pytorch random number generators
torch.manual_seed(1)    # reproducible

# TODO: understand torch.usnqueeze()
# x = torch.tensor([1,2,3])
# print(x,x.size(),x.dim())

# y = torch.unsqueeze(x,0)
# print(y,y.size(),y.dim())

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  
# x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())
# noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)
# TODO: understand Variable


# view data
plt.figure(figsize=(10,4))
plt.scatter(x.data.numpy(), y.data.numpy(), color = "orange")
plt.title('Regression Analysis')
plt.xlabel('Independent varible')
plt.ylabel('Dependent varible')
plt.savefig('./tutorial-5-fig-1.jpg',dpi=300)

# ? why in feature 1, hidden feature 10, and out feature 1?
net = Net(n_feature=1, n_hidden=20, n_output=1)     # define the network
print(net)  # net architecture

# TODO: understand the optim.SGD
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

# define loss function
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

my_images = []
fig, ax = plt.subplots(figsize=(12,7))

# train the network
for t in range(200):

    # input x and predict based on x
    prediction = net(x)     

    # must be (1. nn output, 2. target)
    loss = loss_func(prediction, y)     
    print(t, loss.item())

    # clear gradients for next train
    optimizer.zero_grad()

    # backpropagation, compute gradients
    loss.backward()

    # apply gradients
    optimizer.step()
    
    
    # plot and show learning process
    plt.cla()
    ax.set_title('Regression Analysis', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    ax.set_xlim(-1.05, 1.5)
    ax.set_ylim(-0.25, 1.25)
    
    ax.scatter(x.data.numpy(), y.data.numpy(), color = "orange")
    ax.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=3)
    
    ax.text(1.0, 0.1, 'Step = %d' % t, fontdict={'size': 24, 'color':  'red'})
    ax.text(1.0, 0, 'Loss = %.4f' % loss.data.numpy(),
            fontdict={'size': 24, 'color':  'red'})

    # Used to return the plot as an image array 
    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)

    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    my_images.append(image)
    

# save images as a gif    
#imageio.mimsave('./curve_1.gif', my_images, fps=10)

