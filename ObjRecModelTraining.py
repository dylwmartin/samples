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

from ImgRecModels import ObjRecModel, fit, accuracy, eval

# read in dataset
dataset_objects = ImageFolder('./data/cifar10', transform=ToTensor())

# Object Recognition Model setup
train_set_size = 45000
val_set_size = len(dataset_objects) - train_set_size

train_data, val_data = random_split(dataset_objects, [train_set_size, val_set_size])

dat_batch_size = 50

train_data_loader = DataLoader(train_data, dat_batch_size, shuffle = True, pin_memory = True)
val_data_loader = DataLoader(train_data, dat_batch_size*3, shuffle = True, pin_memory = True)

model = ObjRecModel()

# train Object Recognition Model
fit(5, .1, model, train_data_loader, val_data_loader)