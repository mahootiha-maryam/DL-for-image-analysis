# -*- coding: utf-8 -*-
"""
We try to load MNIST dataset and make a model to predict the classes of this dataset.
We use two convolution layers and three fully connected layers, use relu activation
function in each layer and use max pooling after two conv layers. we use data 
batching with small batches. we train our model with cross entropy and adam optimization
we evaluate our dataset with confusion matrix, precision and recall.
we can track the missed predictions. And at the end we predict a class for an image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,recall_score, precision_score
import matplotlib.pyplot as plt


transform = transforms.ToTensor()
#load the data
train_data = datasets.MNIST(root='./Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./Data', train=False, download=True, transform=transform)

#Create loaders:When working with images, we want relatively small batches;
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

'''
make a class for convolution Network
we set up the convolutional layers with torch.nn.Conv2d()
Conv2d(number of input channels, number of kernels, size of kernels, stride)
The first layer has one input channel (the grayscale color channel).
We'll assign 6 output channels for feature extraction. We'll set our kernel
size to 3 to make a 3x3 filter, and set the step size to 1.
The second layer will take our 6 input channels and deliver 16 output channels.
The input size of (5x5x16) is determined by the effect of our kernels on the 
input image size. A 3x3 filter applied to a 28x28 image leaves a 1-pixel edge 
on all four sides. In one layer the size changes from 28x28 to 26x26. We could
address this with zero-padding, but since an MNIST image is mostly black at the
edges, we should be safe ignoring these pixels. We'll apply the kernel twice, 
and apply pooling layers twice, so our resulting output will be (((28−2)/2)−2)/2=5.5 
which rounds down to 5 pixels per side.
Activations can be applied to the convolutions in one line using F.relu() and 
pooling is done using F.max_pool2d()
F.max_pool2d(x,size of filters, stride)
We flatten the data for the fully connected layers by X = X.view(-1, 5*5*16)
'''

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)#120,84 is an arbitrary number
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
    
torch.manual_seed(42)
model = ConvolutionalNetwork()

#counting the number of parameters that are calculated for our model
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')
    
count_parameters(model)

#loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training the model
def training():
    import time
    start_time = time.time()

    epochs = 2
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
    
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
        
            # Apply the model
            y_pred = model(X_train)  # we don't flatten X-train here
            loss = criterion(y_pred, y_train)
 
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
        
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Print interim results
            if b%600 == 0:
                print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/60000]  loss: {loss.item():10.8f}  \
                      accuracy: {trn_corr.item()*100/(10*b):7.3f}%')
        
        train_losses.append(loss.item())
        train_correct.append(trn_corr.item())
        
        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):

                # Apply the model
                y_val = model(X_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()
            
        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)
        print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
    
    return  train_losses, test_losses,train_correct,test_correct

train_losses, test_losses, train_correct, test_correct= training()
tr1,tr2,tr3=np.array(train_losses), np.array(test_losses), np.array(train_correct)
print(tr1.shape, tr2.shape,tr3.shape)

'''
#Plot the loss and accuracy comparisons
plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend()
'''

'''
train correct and test correct contain 5 numbers each number show the correct 
predicted for understanding the accuracy we devide them by the number of data 
that is 60000 for train and 10000 for test and then multiply them to 100 
to understand the percentage
'''
'''
plt.plot([t/600 for t in  train_correct], label='training accuracy')
plt.plot([t/100 for t in train_losses], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
'''
'''
Evaluate Test Data
'''
# Extract the data all at once, not in batches
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)  # we don't flatten the data this time
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()
print(f'Test accuracy: {correct.item()}/{len(test_data)} = {correct.item()*100/(len(test_data)):7.3f}%')

'''
display confusion matrix, precision and recall
'''
# print a row of values for reference
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))
print(np.arange(10).reshape(1,10))
print()

# print the confusion matrix
print(confusion_matrix(y_test.view(-1),predicted.view(-1)))
print(recall_score(y_test.view(-1),predicted.view(-1),average='weighted'))
print(precision_score(y_test.view(-1),predicted.view(-1),average='weighted'))


'''
We can track the index positions of "missed" predictions, and extract the
corresponding image and label. We'll do this in batches to save screen space.
'''
'''
misses = np.array([])
for i in range(len(predicted.view(-1))):
    if predicted[i] != y_test[i]:
        misses = np.append(misses,i).astype('int64')
        
# Display the number of misses
print(len(misses))
# Set up an iterator to feed batched rows
r = 12   # row size
row = iter(np.array_split(misses,len(misses)//r+1))

nextrow = next(row)
print("Index:", nextrow)
print("Label:", y_test.index_select(0,torch.tensor(nextrow)).numpy())
print("Guess:", predicted.index_select(0,torch.tensor(nextrow)).numpy())

images = X_test.index_select(0,torch.tensor(nextrow))
im = make_grid(images, nrow=r)
plt.figure(figsize=(10,4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
'''
'''
Run a new image through the model:
We can also pass a single image through the model to obtain a prediction.
Pick a number from 0 to 9999, assign it to "x", and we'll use that value to 
select a number from the MNIST test set.
'''
x = 5809
plt.figure(figsize=(1,1))
plt.imshow(test_data[x][0].reshape((28,28)), cmap="gist_yarg")

#evaluate our model
with torch.no_grad():
    new_pred = model(test_data[x][0].view(1,1,28,28)).argmax()
print("Predicted value:",new_pred.item())

