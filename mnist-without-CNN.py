# -*- coding: utf-8 -*-
'''
In this excercise we model the MNIST dataset using only linear layers.
In this exercise we'll use the same logic laid out in the ANN notebook.
We'll reshape the MNIST data from a 28x28 image to a flattened 1x784 vector
to mimic a single row of 784 features.
We used nn.module and nn.linear, batching, relu activation function, cross entropy loss,
adam optimizer.
Then we evaluate our model based on accuracy, confusion matrix,
precision and recall.
finally we plot 12 images of the wrong predicted  
''' 
import torch
import torch.nn as nn
import torch.nn.functional as F          # adds some efficiency
from torch.utils.data import DataLoader  # lets us load data in batches
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import confusion_matrix,recall_score, precision_score  # for evaluating results
import matplotlib.pyplot as plt

'''
we can apply multiple transformations (reshape, convert to tensor, normalize,
etc.) to the incoming data.For this exercise we only need to convert images
to tensors. 
'''
transform = transforms.ToTensor()

#get the mnist data
#root is the folder where the dataset is stored in
#when download is true , if data folder is empty then, it downloads the data to root

train_data = datasets.MNIST(root='./Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./Data1', train=False, download=True, transform=transform)

#the dataset contains image pixels and the label of image
image, label = train_data[0]
print('Shape:', image.shape, '\nLabel:', label)
print(train_data[0])
'''
#show the image with plot
plt.imshow(train_data[0][0].reshape((28,28)), cmap="gray")
plt.imshow(train_data[0][0].reshape((28,28)), cmap="gist_yarg")
'''
#batching
torch.manual_seed(101)  # for consistent results
#we use shuffle to randomly choose data for input
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

'''
Once we've defined a DataLoader, we can create a grid of images using 
torchvision.utils.make_grid
'''


from torchvision.utils import make_grid
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}')) # to widen the printed array

# Grab the first batch of images
for images,labels in train_loader: 
    break

# Print the first 12 labels
print('Labels: ', labels[:12].numpy())
# Print the first 12 images
im = make_grid(images[:12], nrow=12)  # the default nrow is 8
plt.figure(figsize=(10,4))


'''
We need to transpose the images from CWH(channel,width, height) to WHC, because
imshow just organize the pictures as WHC and the picture comes from make_grid is CWH
'''
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

'''
defining the model:
For this exercise we'll use fully connected layers to develop a multilayer 
perceptron.Our input size is 784 once we flatten the incoming 28x28 tensors.
Our output size represents the 10 possible digits.
'''

class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[120,84] #the first layer has 120 neurons 
                 #the second layer has 84 neurons
                 ):
        super().__init__()
        self.fc1 = nn.Linear(in_sz,layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc3 = nn.Linear(layers[1],out_sz)
    
    def forward(self,X):
        #first we give it to fully connected layer, then we give it to activation function
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        #we don't need activation function for the final layer
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(101)
model = MultilayerPerceptron()

'''
This optional step shows that the number of trainable parameters in our model
matches the equation above.
'''
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

count_parameters(model)

'''
Define loss function & optimizer
'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#we can change the dimension from 100,28,28 to 100, 784 by images.view(100,-1)
'''
train our model
'''
def training():
    #for detecting the duration of training
    import time
    start_time = time.time()

    epochs = 10
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
    
        # Run the training batches
        # train loader contains images and their labels
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
        
            # Apply the model
            y_pred = model(X_train.view(100, -1))  # Here we flatten X_train
            loss = criterion(y_pred, y_train)
 
            # Tally the number of correct predictions
            #see where the probability is max then compares with label
            predicted = torch.max(y_pred.data, 1)[1]
            #count the situations where predicted=ytrain 
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
        
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Print interim results
            if b%200 == 0:
                print(f'epoch: {i:2}  batch: {b:4} [{100*b:6}/60000]  loss: {loss.item():10.8f}  \
                      accuracy: {trn_corr.item()*100/(100*b):7.3f}%')
    
        # Update train loss & accuracy for the epoch
        train_losses.append(loss.item())
        train_correct.append(trn_corr.item())
        
        # Run the testing batches
        #no_grad is for the time we don't want to update weight and biases
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):

                # Apply the model
                y_val = model(X_test.view(500, -1))  # Here we flatten X_test

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()
    
        # Update test loss & accuracy for the epoch
        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)
        
        print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed  
    return train_losses, test_losses

x,y=training()
plt.plot(x, label='training loss')
plt.plot(y, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend()

'''
we'd like to compare the predicted values to the ground truth (the y_test labels),
so we'll run the test set through the trained model all at once.
'''

# Extract the data all at once, not in batches
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test.view(len(X_test), -1))  # pass in a flattened view of X_test
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()
        
print(f'Test accuracy: {correct.item()}/{len(test_data)}= {correct.item()*100/(len(test_data)):7.3f}%')

print(confusion_matrix(y_test.view(-1), predicted.view(-1)))
print(recall_score(y_test,predicted,average='weighted'))
print(precision_score(y_test,predicted,average='weighted'))


'''
We can track the index positions of "missed" predictions, and extract the
corresponding image and label. We'll do this in batches to save screen space.
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