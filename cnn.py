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
class_names = datasets.classes 

class_to_idx = datasets.class_to_idx


"""Split datasets into training and validation sets """
torch.manual_seed(42) 
torch.cuda.manual_seed(42)
VAL_SIZE = 5000
TRAIN_SIZE = len(datasets) - VAL_SIZE
train_ds, val_ds = random_split(datasets, [TRAIN_SIZE, VAL_SIZE])   


"""View sample images""" 
def show_example(img, label):  
    plt.figure(figsize=(12, 6))
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Label: {class_names[label]}") 
    plt.axis(False) 


show_example(*datasets[0])
show_example(*datasets[1099])
show_example(*datasets[3000])

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
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle= True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, pin_memory=True) 
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE*2, pin_memory=True)

def show_batch(images): 
    plt.figure(figsize=(12, 6))
    plt.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    plt.axis(False)
    plt.title("Sample Batch")
   

for images, labels in train_dl:
    show_batch(images) 
    break 

plt.show()

"""Transfer Data loaders to device"""
train_dl = DeviceDataLoader(train_dl, device) 
val_dl = DeviceDataLoader(val_dl, device) 
test_dl = DeviceDataLoader(test_dl, device)

"""Define accuracy function""" 
def accuracy(outputs, y_true): 
    "Computes the accuracy of the model"
    y_preds = torch.argmax(outputs, dim=1) 
    return torch.tensor(torch.sum(y_preds == y_true).item() / len(y_true))

class CFar10Model(nn.Module):
    def __init__(self):
        super().__init__() 
        self.linear1 = nn.Linear(in_features=3072, out_features=64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 10)

    def forward(self, xb): 
        xb = xb.reshape(-1, 3072) 
        out = self.linear1(xb) 
        out = F.relu(out)
        out = self.linear2(out) 
        out = F.relu(out) 
        out = self.linear3(out)
        return out 

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
        batch_losses = [x["val_loss"] for x in results]
        epoch_loss = torch.stack(batch_losses).mean().item()
        batch_accs = [x["val_acc"] for x in results]  
        epoch_acc = torch.stack(batch_accs).mean().item()
        return {"val_loss": epoch_loss, "val_acc": epoch_acc} 
    
    def epoch_end(self, epoch, result): 
        print(f"\33[33m Epoch: {epoch+1} | Loss: {result["val_loss"]:.4f} | Acc: {result["val_acc"]:.4f}")  



def evaluate(model, val_dl): 
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs) 
    

def fit(epochs: int, lr: float, model: CFar10Model, train_dl: DataLoader, val_dl: DataLoader, opt_func=torch.optim.Adam): 
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

#fit(5, 0.5, model, train_dl, val_dl)

#results = evaluate(model=model, val_dl=val_dl) 
#print(results)


