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

# define mutual attributes for calculating model error and accuracy
class ImageRecognitionBase(nn.Module):
    def train_step(self, dat_batch):
        X, y = dat_batch
        out = self(X)
        error = F.cross_entropy(out, y)
        return error

    def val_step(self, dat_batch):
        X, y = dat_batch
        out = self(X)
        error = F.cross_entropy(out, y)
        acc = accuracy(out, y)
        return {'validation_error': error, 'validation_acc': acc}

    def val_rep_end(self, outputs):
        dat_batch_ers = [x['validation_error'] for x in outputs]
        rep_er = torch.stack(dat_batch_ers).mean()
        dat_batch_accs = [x['validation_acc'] for x in outputs]
        rep_acc = torch.stack(dat_batch_accs).mean()
        return {'validation_error': rep_er.item(), 'validation_acc': rep_acc.item()}

    def rep_end(self, rep, result):
        print("Rep [{x_1}], validation_error: {x_2:.3f}, validation_acc: {x_3:.3f}".format(x_1=rep,
                                                                                           x_2=result[
                                                                                               'validation_error'],
                                                                                           x_3=result[
                                                                                               'validation_acc']))


# define Number Recognition Model with one hidden layer:
class NumRecModel(ImageRecognitionBase):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x_b):
        x_b = x_b.view(x_b.size(0), -1)
        # transformation 1
        out = F.relu(self.linear1(x_b))
        # transformation 2
        out = self.linear2(out)
        return out


# define Object Recognition Module (Convolutional Neural Network)
class ObjRecModel(ImageRecognitionBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x_b):
        return self.network(x_b)


#TRAINING FUNCTIONS
def eval(model, validation_loader):
    outs = [model.val_step(dat_batch) for dat_batch in validation_loader]
    return model.val_rep_end(outs)

def fit(reps, learning_rate, model, train_data_loader, val_data_loader, optimizer_func=torch.optim.SGD):
    optim = optimizer_func(model.parameters(), learning_rate)
    performance_history = []
    for rep in range(reps):
        for dat_batch in train_data_loader:
            error = model.train_step(dat_batch)
            error.backward()
            optim.step()
            optim.zero_grad()
        result = eval(model, val_data_loader)
        performance_history.append(result)
        model.rep_end(rep, result)
    return performance_history

def accuracy(outputs, labels):
    _, predictions = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))