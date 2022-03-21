# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

data = pd.read_csv('bike_sharing.csv', index_col=0) 
'''
plt.figure(figsize=(8, 6))

#x is yr y is cnt and coloring is based on the spring
sns.barplot('yr', 'cnt', hue = 'season', data = data, ci=None)

plt.legend(loc = 'upper right', bbox_to_anchor=(1.2,0.5))

plt.xlabel('Year')
plt.ylabel('Total number of bikes rented')

plt.title('Number of bikes rented per season')
'''
'''
plt.figure(figsize=(8, 6))
sns.barplot('mnth', 'cnt', hue = 'workingday', data = data, ci=None)

plt.legend(loc = 'upper right', bbox_to_anchor=(1.2,0.5))

plt.xlabel('Year')
plt.ylabel('Total number of bikes rented')

plt.title('Number of bikes rented per month')
'''
#get the seazon field and change each attribute to one column
data = pd.get_dummies(data, columns= ['season'])
#need just these columns
columns = ['registered', 'holiday', 'weekday', 
           'weathersit', 'temp', 'atemp',
           'season_fall', 'season_spring', 
           'season_summer', 'season_winter']

#features are the input(xtrain) of neural network and target is the ytrain
features=data[columns]
target=data[['cnt']]

#use sklearn for dividing our data to train and test
from sklearn.model_selection import train_test_split
#80 percent of data is for training
X_train, x_test, Y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.2)
#change to tensors
X_train_tensor = torch.tensor(X_train.values, dtype = torch.float)
x_test_tensor = torch.tensor(x_test.values, dtype = torch.float)

Y_train_tensor = torch.tensor(Y_train.values, dtype = torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype = torch.float)

'''
batch the data
'''
#use data utils for batching
import torch.utils.data as data_utils 

#tensordataset and loader both used to load multiple samples in parallel
train_data = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = data_utils.DataLoader(train_data, batch_size=100, shuffle=True)
features_batch, target_batch = iter(train_loader).next()


inp = X_train_tensor.shape[1]
out = 1

hid = 10

loss_fn = torch.nn.MSELoss()

#making the neural network model
model = torch.nn.Sequential(torch.nn.Linear(inp, hid),
                            torch.nn.ReLU(),
                            #dropout is good for overfitting the p is the 
                            #probability of deleting the neuron
                            torch.nn.Dropout(p=0.2),
                            torch.nn.Linear(hid, out))

#defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
##############################################
##make epochs based on the train loader size##
##############################################

total_step = len(train_loader)

num_epochs = 10000
#train model based on every batch data 
for epoch in range(num_epochs + 1):
    for i, (features, target) in enumerate(train_loader):
        
        output = model(features)
        loss = loss_fn(output, target)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        if epoch % 2000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#evaluate our model

model.eval()

#get all predicted y value for the all x test
with torch.no_grad():
    y_pred_tensor = model(x_test_tensor)
    
y_pred = y_pred_tensor.detach().numpy()

#make a table for comparing between actual and predicted
compare_df = pd.DataFrame({'actual': np.squeeze(y_test.values), 'predicted': np.squeeze(y_pred)})
#show ten random samples of data frame
print(compare_df.sample(10))

print(sklearn.metrics.r2_score(y_test, y_pred))

'''
Pytorch allows our model to be saved. The parameters to the torch.save() 
method are the model to be saved followed by the directory path where it 
should be saved
'''
torch.save(model, 'my_model')
#We can load a saved model using the torch.load() method
saved_model = torch.load('my_model')


'''
#It is now used exactly how we used the model before it was saved
y_pred_tensor = saved_model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()
'''

#comparing the predicted and actual values based with plot
plt.figure(figsize=(12, 8))
plt.plot(y_pred, label='Predicted count')
plt.plot(y_test.values, label='Actual count')
plt.legend()
plt.show()