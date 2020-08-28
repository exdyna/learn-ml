# Split traning data to train and test sets
# https://carpentries-incubator.github.io/deep-learning_intro/04-ann/index.html

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import imageio

# 
torch.manual_seed(1)    # reproducible

# x data (tensor), shape=(100, 1)
# x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  
x_size = 64
x = torch.rand(100,x_size)*10-5
x,ind = torch.sort(x)
print(x.size())

# noisy y data (tensor), shape=(100, 1)
y = torch.sin(x) + 0.2*torch.rand(x.size())
print(y.size())

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

plt.figure(figsize=(10,4))
plt.scatter(x.data.numpy(), y.data.numpy(), color = "blue",s=1)
plt.title('Regression Analysis')
plt.xlabel('Independent varible')
plt.ylabel('Dependent varible')
plt.savefig('tutorial-6-sample.jpg')


# Step-1 define model
# another way to define a network
# no need to use class

# net = torch.nn.Sequential(
#         torch.nn.Linear(x_size, 50),
#         torch.nn.ELU(),
#         torch.nn.Linear(50, 100),
#         torch.nn.ELU(),
#         torch.nn.Linear(100, 100),
#         torch.nn.ELU(),
#         torch.nn.Linear(100, 50),
#         torch.nn.ELU(),
#         torch.nn.Linear(50, x_size),
    # )

# net = torch.nn.Sequential(
#         torch.nn.Linear(10, 20),
#         torch.nn.LeakyReLU(),
#         torch.nn.Linear(20, 100),
#         torch.nn.LeakyReLU(),        
#         torch.nn.Linear(100, 20),
#         torch.nn.LeakyReLU(),
#         torch.nn.Linear(20, 10),
#     )

# net = torch.nn.Sequential(
#         torch.nn.Linear(x_size, 10),
#         torch.nn.LeakyReLU(),
#         torch.nn.Linear(10, 20),
#         torch.nn.LeakyReLU(),        
#         torch.nn.Linear(20, x_size),
#     )

net = torch.nn.Sequential(
        torch.nn.Linear(x_size, 50),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50,50),        
        torch.nn.LeakyReLU(),   
        torch.nn.Linear(50, x_size)
    )


# Step-2: define a optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-2,weight_decay=0.005)

# optimizer = torch.optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')

# Step-3 define a loss function
# this is for regression mean squared loss
loss_func = torch.nn.MSELoss()  

# Step-4: define input and out put
BATCH_SIZE = 10
EPOCH = 200
training_convergence = 1e-2

torch_dataset = Data.TensorDataset(x, y)
# Dataset wrapping tensors.
# Each sample will be retrieved by indexing tensors along the first dimension.


# TODO: understand loader
loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,)
# this loader will load the data with the number of batch_size from dataset randomly
# in this case the loader will pick up 64 data each time for 15 times, and the rest 40 data  

# num_workers is not related to batch_size. 
# Say you set batch_size to 20 and the training size is 2000, 
# then each epoch would contain 100 iterations, i.e. for each iteration, 
# the data loader returns a batch of 20 instances. num_workers > 0 
# is used to preprocess batches of data so that the next batch is ready 
# for use when the current batch has been finished. More num_workers 
# would consume more memory usage but is helpful to speed up the I/O process. 


my_images = []
fig, ax = plt.subplots(figsize=(16,10))

# start training
for epoch in range(EPOCH):

    for step, (batch_x, batch_y) in enumerate(loader): 
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        # each time pick up 64 data, and the last time pick up the rest 40 data
        
        # calculate the prediction
        prediction = net(b_x)

        # calculate the loss function
        loss = loss_func(prediction, b_y)
                
        # clear gradients for next train
        optimizer.zero_grad()   
        
        # backpropagation, compute gradients
        loss.backward()

        # apply gradients
        optimizer.step()

        if step == 1:

            print(epoch, b_x.size(), loss.item())            

            # # plot and show learning process
            # plt.cla()
            # ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
            # ax.set_xlabel('Independent variable', fontsize=24)
            # ax.set_ylabel('Dependent variable', fontsize=24)
            # ax.set_xlim(-11.0, 13.0)
            # ax.set_ylim(-1.1, 1.2)
            # ax.scatter(b_x.data.numpy(), b_y.data.numpy(), color = "blue", alpha=0.2)
            # ax.scatter(b_x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
            # ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
            #         fontdict={'size': 24, 'color':  'red'})
            # ax.text(8.8, -0.95, 'Loss = %.4f' % loss.data.numpy(),
            #         fontdict={'size': 24, 'color':  'red'})

            # # Used to return the plot as an image array 
            # # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
            # fig.canvas.draw()       # draw the canvas, cache the renderer
            # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            # image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # my_images.append(image)
    
    if loss < training_convergence:
        break

torch.save(net.state_dict(), 'model.pth')
torch.save(optimizer.state_dict(), 'optimiser.pth')

    
# save images as a gif    
# imageio.mimsave('./tutorial-6-training.gif', my_images, fps=12)


fig, ax = plt.subplots(figsize=(16,10))
plt.cla()
ax.set_title('Regression Analysis - model 3, Batches', fontsize=35)
ax.set_xlabel('Independent variable', fontsize=24)
ax.set_ylabel('Dependent variable', fontsize=24)
ax.set_xlim(-11.0, 13.0)
ax.set_ylim(-1.1, 1.2)

ax.scatter(x.data.numpy(), y.data.numpy(), color = "blue", alpha=0.8)
prediction = net(x) 
ax.scatter(x.data.numpy(), prediction.data.numpy(), color='red', alpha=0.5)

# ! if the model is trained using randomly distributed data, the prediciton only works for the randomly distributed data
x = torch.rand(1,x_size)*10-5
x.ind = torch.sort(x)
prediction = net(x) 
ax.scatter(x.data.numpy(), prediction.data.numpy(), color='black')

x = torch.linspace(-5,5,x_size)
prediction = net(x) 
ax.scatter(x.data.numpy(), prediction.data.numpy(), color='yellow')

plt.savefig('tutorial-6-result.jpg')
# plt.show()

