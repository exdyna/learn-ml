# try to find the polynomial coefficients
# y = a2*x^2 + a1*x^1 + a0
# given data set of (y,a)
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
        
def setup(ax):
    ax.grid(False)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)
        

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,4), useOffset=True, useLocale=None, useMathText=True)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,4), useOffset=True, useLocale=None, useMathText=True)



def training_loop(n_epochs, optimizer, model, loss_fn, 
    train_loader, test_loader,coverge_tol):

    loss_history_train = np.empty(1)
    loss_history_test = np.empty(1)
    
    for epoch in range(1, n_epochs + 1):
        
        # ======================================================================
        # train        
        total_train_corrects = 0
        total_train_loss = 0

        for step, (batch_train_in, batch_train_out) in enumerate(train_loader): 

            train_in = Variable(batch_train_in)
            train_out = Variable(batch_train_out)
            
            # calculate the prediction
            prediction = model(train_in)

            # calculate the loss function
            train_loss = loss_fn(prediction, train_out)
            
            total_train_loss += train_loss
            
            # clear gradients for next train
            optimizer.zero_grad()   
            
            # backpropagation, compute gradients
            train_loss.backward()

            # apply gradients
            optimizer.step()

        # ======================================================================
        # test
        with torch.no_grad():
            
            total_test_loss = 0
            total_test_correct = 0

            for step, (batch_test_in, batch_test_out) in enumerate(test_loader): 
                
                test_in = Variable(batch_test_in)
                test_out = Variable(batch_test_out)

                test_prediction = model(test_in)
                test_loss = loss_fn(test_prediction, test_out)

                total_test_loss += test_loss

            assert test_loss.requires_grad == False

            scheduler.step(total_test_loss) 

            print('Epoch {}, Train loss:{:.4f}, Test loss: {:.4f}'.format((epoch+1),
                total_train_loss, total_test_loss))
        
        loss_history_train = np.append(loss_history_train, total_train_loss.item())
        loss_history_test = np.append(loss_history_test, total_test_loss.item())

        # print(epoch,train_loss.item(), test_loss.item())

        # ======================================================================
        if (total_test_loss.item() < coverge_tol) or (epoch == n_epochs):
            
            fig, ax = plt.subplots(figsize=cm2inch((8,8)))
            plt.cla()
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
            plt.rcParams['font.size'] = font_size

            ax.set_xlabel('Input', fontsize=font_size)
            ax.set_ylabel('Prediction', fontsize=font_size)

            # ax.scatter(train_in.data.numpy(), prediction.data.numpy() , color='red', alpha=0.5)
            ax.scatter(test_out.data.numpy(), test_prediction.data.numpy(), color='blue')
            bnd = np.max([np.max(np.abs(test_out.data.numpy())),np.max(np.abs(test_prediction.data.numpy()))])

            ax.plot([-bnd,bnd],[-bnd,bnd],color='red')
            setup(ax)

            fig.subplots_adjust(top=0.95,bottom=0.2,left=0.2)

            ax.set_aspect('equal', 'box')


            plt.savefig('fig-test-result.jpg',dpi=300)
            plt.close()


            torch.save(model.state_dict(), 'model_tutorial_7.pth')
            torch.save(optimizer.state_dict(), 'optimiser_tutorial_7.pth')

            return loss_history_train, loss_history_test

            break

        # if epoch == n_epochs:
            
        #     fig, ax = plt.subplots(figsize=(10,10))
        #     plt.cla()
        #     ax.set_xlabel('Input', fontsize=24)
        #     ax.set_ylabel('Prediction', fontsize=24)

        #     # ax.scatter(train_in.data.numpy(), prediction.data.numpy() , color='red', alpha=0.5)
        #     ax.scatter(test_out.data.numpy(), test_prediction.data.numpy(), color='blue', alpha=0.8)

        #     ax.set_aspect('equal', 'box')

        #     plt.savefig('tutorial-7-result.jpg')            

        #     torch.save(model.state_dict(), 'model_tutorial_7.pth')
        #     torch.save(optimizer.state_dict(), 'optimiser_tutorial_7.pth')
            

# ==============================================================================
font_size = 10

torch.manual_seed(1) 
np.random.seed(1)

# preparing input and output data
# number of  samples
n_samples = 50
# number of features for each sample
n_in = 5

# number of ouptput: n_out = n + 1 for n-th order polynomial
n_out = 2

# lower and upper bound for x in polynomial function y=f(x)
x_lb = -2
x_ub = 2

# bound of polynomial coefficients
p_lb = -1
p_ub = 1

x_poly = np.random.randn(n_samples,n_in)*(x_ub - x_lb) + x_lb

p = np.random.randn(n_samples,n_out)

# iniitalize polyfunction vaue
y = np.zeros((n_samples,n_in),dtype=float)

for count, pi in enumerate(p):
    y[count,:] = np.polyval(pi,x_poly[count,:])

y_tensor = torch.tensor(y) 
p_tensor = torch.tensor(p)

y_tensor = y_tensor.float()
p_tensor = p_tensor.float()

# ==============================================================================
# split the data set into train and testidation sets
batch_size_train = int(n_samples*0.8)
batch_size_test = n_samples - batch_size_train

shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-batch_size_test]
test_indices = shuffled_indices[-batch_size_test:]

y_train = y_tensor[train_indices]
y_test = y_tensor[test_indices]

p_train = p_tensor[train_indices]
p_test = p_tensor[test_indices]

train_dataset = Data.TensorDataset(y_train,p_train)
test_dataset = Data.TensorDataset(y_test,p_test)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_train,num_workers=1)
test_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_test,num_workers=1)

# dataset = Data.TensorDataset(y_tensor, p_tensor)
# train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size_train,num_workers=1)
# test_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size_test,num_workers=1)

print(train_loader.dataset)
# ==============================================================================
# define the neural network
model = torch.nn.Sequential(
        torch.nn.Linear(n_in, 10),
        torch.nn.LeakyReLU(),        
        torch.nn.Linear(10, n_out)
    )

learning_rate = 1e-2
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')

# start training
loss_train, loss_test = training_loop(n_epochs=10000, 
    optimizer = optimizer,
    loss_fn = loss_fn,
    model = model, 
    train_loader = train_loader, 
    test_loader = test_loader,
    coverge_tol = 1e-2)

torch.save(model.state_dict(), 'model_tutorial_7.pth')
torch.save(optimizer.state_dict(), 'optimiser_tutorial_7.pth')

plt.clf()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = font_size
fig, ax = plt.subplots(figsize=cm2inch((8,8)))
ax.set_xlabel('Number of iteration', fontsize=font_size)
ax.set_ylabel('Loss', fontsize=font_size)
ax.plot(loss_train[1:], color='red',label='train error')
ax.plot(loss_test[1:], color='blue',label='test error')

ax.legend(fancybox=True,facecolor ='white',edgecolor='black',ncol=1,fontsize=font_size)
setup(ax)
fig.subplots_adjust(top=0.95,bottom=0.2,left=0.2)
plt.savefig('fig-loss-history.jpg',dpi=300)
plt.close()
