'''
_______________________________________________________________________________
In this code we try to predict survival probabilities from 3d images
_______________________________________________________________________________
We must have two kinds of data:1-3d images 2-events + time duration(events are 0 or 1 
                                                           1 shows that s.th happened
                                                           for example death,heart stroke.
                                                           0 shows that this is a
                                                           censored data and we do not
                                                           idea what happened for that patients
                                                           maybe they leave the research)

_______________________________________________________________________________
We use DeepSurv for our survival model. this is one neural network model for 
estimating survival probabilities

_______________________________________________________________________________
We use PyCox one useful library in python for making survival models and 
evaluating them
_______________________________________________________________________________
Because our inputs to the survival model are images we use CNN + FC for making 
neural networks. And as we have 3d images we use 3d CNN
_______________________________________________________________________________ 
The point here is DatasetSingle class that takes images, events and time duration 
and makes a dataset that contains all of them 
We give this dataset to the survival model(contains the neural network) 
_______________________________________________________________________________
'''

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import torchtuples as tt
from pycox.models import LogisticHazard, CoxPH
from pycox.utils import kaplan_meier
from pycox.evaluation import EvalSurv




import os
from glob import glob
import shutil
from tqdm import tqdm
import numpy as np

from monai.data import Dataset
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)


#the path to the images of patients
data_dir = 'C:/Users/Asus/Documents/py-torch/kits21'

path_train_volumes = sorted(glob(os.path.join(data_dir, "imagestr", "*.nii.gz")))
path_test_volumes = sorted(glob(os.path.join(data_dir, "valim", "*.nii.gz")))

train_files = [{"vol": image_name} for image_name in path_train_volumes]
test_files = [{"vol": image_name} for image_name in path_test_volumes]

#making some transforms on the train and test dataset(preprocessing)
pixdim=(1.5, 1.5, 1.0)
a_min=-200
a_max=200
spatial_size=[128,128,64]

train_transforms = Compose(
     [
        LoadImaged(keys=["vol"]),
        AddChanneld(keys=["vol"]),
        #Spacingd(keys=["vol"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=["vol"], source_key="vol"),
        Resized(keys=["vol"], spatial_size=spatial_size),   
        ToTensord(keys=["vol"]),

     ]
 )

test_transforms = Compose(
     [
        LoadImaged(keys=["vol"]),
        AddChanneld(keys=["vol"]),
        #Spacingd(keys=["vol"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=['vol'], source_key='vol'),
        Resized(keys=["vol"], spatial_size=spatial_size),   
        ToTensord(keys=["vol"]),

         
     ]
 )

train_ds = Dataset(data=train_files, transform=train_transforms)
#train_loader = DataLoader(train_ds, batch_size=1)

#train_loader = DataLoader(train_ds, batch_size=len(train_ds),collate_fn=pad_list_data_collate)
#train_dataset_array = next(iter(train_loader))[0].numpy()

test_ds = Dataset(data=test_files, transform=test_transforms)
#test_loader = DataLoader(test_ds, batch_size=1)

# =============================================================================
# 
# def sim_event_times(kidneys, max_time=700):
#     
#     betas = 365 * np.exp(-0.6 * kidneys) / np.log(1.2)
#     event_times = np.random.exponential(betas)
#     censored = event_times > max_time
#     event_times[censored] = max_time
#     return tt.tuplefy(event_times, ~censored)
# 
# sim_train = sim_event_times(train_dataset_array)
# #sim_test = sim_event_times(test_ds)
# =============================================================================

'''
We should have two tuples one for time durations and one for event
We make a nested tuple and change this nested tuple to array
'''
duration=(521,700,315,15,110,700)
event=(1,0,1,1,1,0)
surv_tup= ((duration, ) + (event, ))
surv_tup=np.asarray(surv_tup)

duration1=(5,20,10)
event1=(1,1,1)
surv_tup1= ((duration1, ) + (event1, ))
surv_tup1=np.asarray(surv_tup1)

labtrans = LogisticHazard.label_transform(5)
target_train = labtrans.fit_transform(*surv_tup)
target_test = labtrans.transform(*surv_tup1)

'''
This class takes the dataset, time duration and events 
Then return all of them as a dataset with inheritance from Dataset class
'''

class DatasetSingle(Dataset):

    def __init__(self, kidney_dataset, time, event):
        self.kidney_dataset = kidney_dataset
        self.time, self.event = tt.tuplefy(time, event).to_tensor()

    def __len__(self):
        return len(self.kidney_dataset)

    def __getitem__(self, index):
        if type(index) is not int:
            raise ValueError(f"Need `index` to be `int`. Got {type(index)}.")
        #extract the image pixels from the dictionary of dataset
        img = self.kidney_dataset[index]['vol']
        return img, (self.time[index], self.event[index])
    
    
dataset_train = DatasetSingle(train_ds, *target_train)
dataset_test = DatasetSingle(test_ds, *target_test)

samp = tt.tuplefy(dataset_train[1])
print(samp.shapes())

def collate_fn(batch):
    """Stacks the entries of a nested tuple"""
    return tt.tuplefy(batch).stack()

#making the dataloader from the dataset
batch_size = 1
dl_train = DataLoader(dataset_train, batch_size, shuffle=True, collate_fn=collate_fn)
dl_test = DataLoader(dataset_test, batch_size, shuffle=False, collate_fn=collate_fn)

batch = next(iter(dl_train))
print(batch.shapes())

'''
Make a Neural Network that contains 1 3d cnn with 16 filters 5*5, 1 3d maxpooling
2*2,  1 3d cnn with 16 filters 5*5, 1 AdaptiveAvgPool3d, 1 FC layer with 16 input
and 16 output and the last is a FC layer with 16 input and the output is 
**the time intervals we have**
For example we can have 20 time intervals
'''
class Net(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 5, 1)
        self.max_pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 16, 5, 1)
        self.glob_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.glob_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
net = Net(5)

#model = LogisticHazard(net, tt.optim.Adam(0.01))
'''
make survival analysis model with CoxPH 
Fit the model for training with early stopping
'''

model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

pred = model.predict(batch[0])
print(pred.shape)
callbacks = [tt.cb.EarlyStopping(patience=5)]
epochs = 5
verbose = True
#log = model.fit_dataloader(dl_train, epochs, callbacks, verbose, val_dataloader=dl_test)

#_ = log.plot()

'''
Predict the model
To predict, we need a data loader that only gives the images and not the targets.
We therefore need to create a new Dataset for this purpose.
'''
class imInput(Dataset):
    def __init__(self, im_dataset):
        self.im_dataset = im_dataset

    def __len__(self):
        return len(self.im_dataset)

    def __getitem__(self, index):
        img = self.im_dataset[index]['vol']
        return img
    
dataset_test_x = imInput(test_ds)
dl_test_x = DataLoader(dataset_test_x, batch_size, shuffle=False)
print(next(iter(dl_test_x)).shape)


surv = model.interpolate(10).predict_surv_df(dl_test_x)
ev = EvalSurv(surv, *surv_tup1, 'km')

print(ev.concordance_td())
time_grid = np.linspace(0, surv_tup1.max())
print(ev.integrated_brier_score(time_grid))

# =============================================================================
# for i in range(2): 
#     surv.iloc[:, i].plot()
# _ = plt.legend()
# =============================================================================

surv.iloc[:, 0].plot()
surv.iloc[:, 1].plot()