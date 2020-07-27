import os
import torch
import torchvision
import numpy as np

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from ImgRecModels import NumRecModel, fit, accuracy, eval

#read in dataset
dataset_nums = MNIST(root='data/', download = True, transform=ToTensor())

#Number Recognition Model setup
train_set_size = 50000
val_set_size = len(dataset_nums) - train_set_size

train_data, val_data = random_split(dataset_nums, [train_set_size, val_set_size])

dat_batch_size = 100

train_data_loader = DataLoader(train_data, dat_batch_size, shuffle = True, pin_memory = True)
val_data_loader = DataLoader(train_data, dat_batch_size*3, shuffle = True, pin_memory = True)

input_size = 784
hidden_layer_size = 64
output_size = 10

model = NumRecModel(input_size, hidden_layer_size, output_size)

# train Number Recognition Model
fit(10, .5, model, train_data_loader, val_data_loader)
