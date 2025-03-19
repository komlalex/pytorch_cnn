"""CONVOLUTIONAL NEURAL NETWORKS""" 
import torch 
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import CIFAR10  
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

import matplotlib.pyplot as plt 
import numpy as np 

"""Write device-agnostic code"""
def get_default_device(): 
    if torch.cuda.is_available(): 
        return torch.device("cuda") 
    else: 
        return torch.device("cpu") 

def to_device(data, device): 
    if isinstance(data, (list, tuple)): 
        return [to_device(x, device) for x in data] 
    return data.to(device, non_blocking=True) 

device = get_default_device() 



"""Download data"""

datasets = CIFAR10(root="/data", 
                   download=True, 
                   transform=ToTensor())
test_ds = CIFAR10(root="/data", 
                        download=True, 
                        transform=ToTensor(), 
                        train=False) 
TRAIN_LENGTH = 40_000
VAL_LENGTH = len(datasets) - TRAIN_LENGTH
"""Explore data"""
"""Split datasets into training and validation sets """
train_ds, val_ds = random_split(datasets, [TRAIN_LENGTH, VAL_LENGTH])   

"""Create device data loader"""
class DeviceDataLoader():
    """Takes a data loader object and transfers it to the available device"""
    def __init__(self, dl, device):
        self.dl = dl 
        self.device = device 

    def __iter__(self): 
        for batch in self.dl:
            yield to_device(batch, self.device) 

    def __len__(self):
        return len(self.dl) 

""" Create dataloaders """
BATCH_SIZE = 128
train_dl = DeviceDataLoader(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle= True, pin_memory=True), device)
val_dl = DeviceDataLoader(DataLoader(val_ds, batch_size=BATCH_SIZE*2, pin_memory=True), device)
test_dl = DeviceDataLoader(DataLoader(test_ds, BATCH_SIZE*2, pin_memory=True), device)


"""Define accuracy function""" 
def accuracy(outputs, y_true): 
    "Computes the accuracy of the model"
    y_preds = torch.argmax(outputs, dim=1) 
    return torch.tensor(torch.sum(y_preds == y_true).item() / len(y_true))

class CFar10Model(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, xb): 
        pass 

    def training_step(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels) 
        return loss 
    
    def validation_step(self, batch):
        images, labels = batch 
        outputs = self(images) 
        loss = F.cross_entropy(outputs, labels) 
        acc = accuracy(outputs, labels)  
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, results):
        epoch_losses = [x["val_loss"] for x in results]
        loss = torch.stack(epoch_losses).mean() 
        epoch_accs = [x["val_acc"] for x in results]  
        acc = torch.stack(epoch_accs).mean() 
        return {"val_loss": loss, "val_acc": acc} 
    
    def epoch_end(self, epoch, result): 
        print(f"\33[32m Epoch: {epoch+1} | Loss: {result["val_loss"]} | Acc: {result["val_acc"]}")  


def evaluate(model, val_dl): 
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs) 
    

def fit(epochs: int, lr: float, model: CFar10Model, train_dl: DataLoader, val_dl: DataLoader, opt_func=torch.optim.SGD): 
    optimizer = opt_func(model.parameters(), lr) 
    history = []
    for epoch in range(epochs): 
        # Training phase 
        for batch in train_dl: 
            loss = model.training_step(batch)
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad() 

        # Evaluation Phase 
        results = evaluate(model, val_dl) 
        model.epoch_end(epoch, results) 
        history.append(results) 
    return history


model = CFar10Model()
to_device(model, device)

# history = fit(5, 0.05, model)


