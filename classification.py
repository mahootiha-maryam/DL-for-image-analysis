# -*- coding: utf-8 -*-
'''
we have tabular data of some students who took gre and toefl exam .
We want to predict the admission probability based on the other features.
we used a class for making our model and a function for training our model.
we used activation function and dropout
'''

import torch
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('admission_predict.csv')
#you can see the data fields by data.head()
#you can see the data dimension by data.shape
#you can understand the mean of data in columns, std and these kinds of
#things by data.describe()

data = data.rename(index=str, columns={'Chance of Admit ': 'Admit_Probability'
                                       })

data = data[['GRE Score', 'TOEFL Score', 
             'University Rating', 'SOP', 
             'LOR ', 'CGPA', 'Research', 
             'Admit_Probability']]
'''
plt.figure(figsize=(8, 8))
fig = sns.regplot( x ="GRE Score", y = "TOEFL Score", data = data)
plt.title("GRE Score vs TOEFL Score")

plt.show()

plt.figure(figsize=(8, 8))
fig = sns.scatterplot(x = 'Admit_Probability', y = 'CGPA', data = data,
                      hue = 'Research')

plt.title("CGPA vs Admit Probability")

plt.xlabel('Admit_Probability')
plt.ylabel('CGPA')
'''
from sklearn import preprocessing

#subtract the mean and devide by standard deviation
#we standardize our data before passing it to neural networks by scale
data[['GRE Score', 'TOEFL Score', 'SOP', 'LOR ', 'CGPA' ]] = \
                preprocessing.scale(data[['GRE Score', 'TOEFL Score','SOP', 'LOR ', 'CGPA']])
                
col = ['GRE Score','TOEFL Score', 'SOP', 'LOR ', 'CGPA']

features = data[col]
target = data[['Admit_Probability']]
listoftarget=[]

#we change the admit probability column to 3 classes based on the probabalities
#intervals we need these classes for classification
for i,j in enumerate(target['Admit_Probability']):
    if float(j)>= 0.80:
        listoftarget.append(2)
    if 0.60<=float(j)<0.80:
        listoftarget.append(1)
    if float(j)<0.60:
        listoftarget.append(0)
        
#change the list to a table(dataframe)        
ytarget=pd.DataFrame(data=listoftarget,columns=['Admit_Probability'] )

from sklearn.model_selection import train_test_split
#split our data to train and test the test is 20 percent of train
X_train, x_test, Y_train, y_test = train_test_split(features,
                                                    ytarget,
                                                    test_size=0.2)

#converting data to the tensors

Xtrain = torch.from_numpy(X_train.values).float()
Xtest = torch.from_numpy(x_test.values).float()

'''
view: with view we reshape the tensor view with -1
If there is any situation that you don't know how many columns you want but 
are sure of the number of rows then you can mention it as -1, or visa-versa 
(You can extend this to tensors with more dimensions. Only one of the axis 
 value can be -1)
'''
Ytrain = torch.from_numpy(Y_train.values).view(1, -1)[0].long()

Ytest = torch.from_numpy(y_test.values).view(1, -1)[0].long()

import torch.nn as nn

#gives granular control over design
import torch.nn.functional as F

input_size = Xtrain.shape[1]
#gives the classes we have in this column
output_size = len(ytarget['Admit_Probability'].unique())


'''
Define a neural network class from which to create our model We create a class
named Net which inherits nn.Module(Base class for all neural network modules.)
we want to train the neural network such that we pass in the hidden size,choose
our activation function and choose wether the model will have dropout layer or
not super : This is calling the init() method of the parent class(nn.Module)
fc1 to fc3 : Applies a linear transformation to the incoming data: y=Wx+b 
Dropout : During training, randomly zeroes some of the elements of the input 
tensor with probability p using samples from a Bernoulli distribution.
Parameters : in_features – size of each input sample out_features – size of 
each output sample bias – If set to False, the layer will not learn an additive
bias. Default: True
Sigmoid : Applies the element-wise function Sigmoid(x)= 1 / (1+exp(−x)) 
Tanh: Applies the element-wise function Tanh(x) = (ex - e-x)/(ex + e-x)
ReLu : Applies the rectified linear unit function element-wise ReLu(x)= max(0, x)
log_softmax : Softmax applies the Softmax() function to an n-dimensional 
input Tensor rescaling them so that the elements of the n-dimensional output 
Tensor lie in the range (0,1) and sum to 1 While mathematically equivalent to
log(softmax(x)), doing these two operations separately is slower, and numerically
unstable. This function uses an alternative formulation to compute the output 
and gradient correctly. Parameters: dim(int) – A dimension along which Softmax 
will be computed (so every slice along dim will sum to 1)
'''
class Net(nn.Module):
    
    def __init__(self,hidden_size, activation_fn, apply_dropout=False): 
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        
        self.dropout = None
        if apply_dropout:
            self.dropout = nn.Dropout(0.2)

    
    def forward(self, x):
        
        activation_fn = None
        if  self.activation_fn == 'sigmoid':
                activation_fn = F.torch.sigmoid

        elif self.activation_fn == 'tanh':
                activation_fn = F.torch.tanh

        elif self.activation_fn == 'relu':
                 activation_fn = F.relu
                 

                 
        x = activation_fn(self.fc1(x))
        x = activation_fn(self.fc2(x))

        if self.dropout != None:
            x = self.dropout(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim = -1)
    
import torch.optim as optim

#train the model
def train_and_evaluate_model(model, learn_rate=0.001):
    epoch_data = []
    epochs = 1001
    
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    
    #we use logsoftmax for loss function
    loss_fn = nn.NLLLoss()
    
    test_accuracy = 0.0
    for epoch in range(1, epochs):

        optimizer.zero_grad()

        Ypred = model(Xtrain)

        loss = loss_fn(Ypred , Ytrain)
        loss.backward()

        optimizer.step()

        Ypred_test = model(Xtest)
        loss_test = loss_fn(Ypred_test, Ytest)

        _, pred = Ypred_test.data.max(1)

        test_accuracy = pred.eq(Ytest.data).sum().item() / y_test.values.size
        
        epoch_data.append([epoch, loss.data.item(), loss_test.data.item(), test_accuracy])

        if epoch % 100 == 0:
            print ('epoch - %d (%d%%) train loss - %.2f test loss - %.2f Test accuracy - %.4f'\
                   % (epoch, epoch/150 * 10 , loss.data.item(), loss_test.data.item(), test_accuracy))
            

    return {'model' : model,
            'epoch_data' : epoch_data, 
            'num_epochs' : epochs, 
            'optimizer' : optimizer, 
            'loss_fn' : loss_fn,
            'test_accuracy' : test_accuracy,
            '_, pred' : Ypred_test.data.max(1),
            'actual_test_label' : Ytest,
            }

net = Net(hidden_size=50, activation_fn='relu', apply_dropout=True)
result = train_and_evaluate_model(net)


#Converting all our data in dataframes to plot it
df_epochs_data = pd.DataFrame(result['epoch_data'], 
                              columns=["epoch", "train_loss", "test_loss", "accuracy"])


# plot the accuracy , train loss and test loss in two diagrams
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

df_epochs_data[["train_loss", "test_loss"]].plot(ax=ax1)
df_epochs_data[["accuracy"]].plot(ax=ax2)
plt.ylim(bottom = 0.5)
plt.show()

#plot a confusion matrix for the result
from sklearn.metrics import confusion_matrix, recall_score, precision_score
_, pred = result['_, pred'] 
y_pred = pred.detach().numpy()
Ytest = result['actual_test_label'].detach().numpy()
results = confusion_matrix(Ytest, y_pred)
recall_s=recall_score(Ytest, y_pred,average = 'weighted')
precision_s=precision_score(Ytest, y_pred,average = 'weighted')
