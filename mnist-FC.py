# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F          # adds some efficiency
from torch.utils.data import DataLoader  # lets us load data in batches
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix  # for evaluating results
import matplotlib.pyplot as plt

'''
we can apply multiple transformations (reshape, convert to tensor, normalize,
etc.) to the incoming data.For this exercise we only need to convert images
to tensors. 
'''
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)

