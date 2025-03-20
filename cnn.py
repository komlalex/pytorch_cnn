"""CONVOLUTIONAL NEURAL NETWORKS""" 
import torch 
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from torchvision.datasets import CIFAR10  
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

import matplotlib.pyplot as plt 
import numpy as np 


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



"""Define accuracy function""" 
def accuracy(outputs, y_true): 
    "Computes the accuracy of the model"
    y_preds = torch.argmax(outputs, dim=1) 
    return torch.tensor(torch.sum(y_preds == y_true).item() / len(y_true))
"""Defining the Model (Convolutional Neural Networks) 
In previous examples, we defined a deep neural network with fully-connected layers using 
nn.Linear. Here, we will use a convolutional neural network, using the nn.Con2d class from PyTorch.

The 2D convolution is a fairly simple operation at heart: you start with a kernel which is simply a small matrix of weights. 
The kernel "slides" over the 2D input data, performing an element-wise multiplication 
with the part of the input it is curently on, then summing up the results into a single output pixel"""

"""Before we define the entire model, let's look at how a single convolutiona layer followed by a max-pooling 
layer operates on the data"""
conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1) 
pool = nn.MaxPool2d(2, 2)
for images, labels in train_dl:
    #print(images.shape)
    out = conv(images)
    #print(out.shape) 
    out = pool(out) 
    #print(out.shape)
    break

simple_model  = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), 
    nn.MaxPool2d(2, 2)
)

for images, labels in train_dl:
    #print(images.shape)
    out = simple_model(images)
    #print(out.shape)
    break



class ImageClassificationBase(nn.Module):
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
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, results):
        batch_losses = [x["val_loss"] for x in results]
        epoch_loss = torch.stack(batch_losses).mean().item()
        batch_accs = [x["val_acc"] for x in results]  
        epoch_acc = torch.stack(batch_accs).mean().item()
        return {"val_loss": epoch_loss, "val_acc": epoch_acc} 
    
    def epoch_end(self, epoch, result): 
        print(f"\33[33m Epoch: {epoch+1} | train_loss: {result["train_loss"]:.4f} | val_loss: {result["val_loss"]:.4f} | val_acc: {result["val_acc"]:.4f}")  

"""We'll use the nn.Sequential to chain the layers and activation functions into a single
architecture"""
class Cifar10CnnModel(ImageClassificationBase): 
    def __init__(self):
        super().__init__() 
        self.network = nn.Sequential(
            # input: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # Output: 32 x 32 x 32
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: 64 x 32 x 32
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),# Ouputs: 64 x 16 x 16 

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: 128 x 16 x 16
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # Output: 128 x 16 x 16
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), # Output 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # Output: 256 x 8 x 8
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # Output: 256 x 8 x 8
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), # Output 256 x 4 x 4 

            nn.Flatten(),
            nn.Linear(256*4*4, 1024), 
            nn.ReLU(), 
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Linear(512, 10)

        ) 

    def forward(self, x) -> torch.Tensor:
        return self.network(x)

model = Cifar10CnnModel() 
#print(summary(model))

"""Let's verify that the model produces the expected output on a batch of training data. The 10 outputs for each
image can be interpreted as the proberbility for the 10 target classes (after applying softmax), and the class
with the highest probability is chosen as the label predicted by the model for th input image."""

for images, labels in train_dl: 
    #print("images shape: ", images.shape)
    out = model(images)
    #print("output shape: ", out.shape) 
    #print("out[0]: ", out[0])
    break 

"""To seamlessly use the GPU, if one is available, we define a couple of helper functions 
(get_default_device & to_device) and a helper class DeviceDataLoader to movev out model and data 
to the GPU as required."""

def get_default_device(): 
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available(): 
        return torch.device("cuda") 
    else: 
        return torch.device("cpu") 

def to_device(data: torch.Tensor, device: torch.device): 
    """Move tensor to chosen device"""
    if isinstance(data, (list, tuple)): 
        return [to_device(x, device) for x in data] 
    return data.to(device, non_blocking=True) 


class DeviceDataLoader():
    """Wraps a data loader  move data to device"""
    def __init__(self, dl, device):
        self.dl = dl 
        self.device = device 

    def __iter__(self): 
        for batch in self.dl:
            yield to_device(batch, self.device) 

    def __len__(self):
        return len(self.dl) 


device = get_default_device() 

"""We can now wrap our training and validation data loaders using DeviceDataLoader to automatically 
transfer batches of data to the GPU (if available), and use to_device to move our model to the GPU (if available)
"""
train_dl = DeviceDataLoader(train_dl, device) 
val_dl = DeviceDataLoader(val_dl, device) 
to_device(model, device)

"""Training the Model
We'll define two functions: fit and evaluate to train the model using gradient descent and evaluate 
its performance on the validation set.
"""

def evaluate(model: nn.Module, val_dl): 
    model.eval()
    with torch.inference_mode():
        outputs = [model.validation_step(batch) for batch in val_dl]
        return model.validation_epoch_end(outputs) 
    

def fit(epochs: int, lr: float, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, opt_func=torch.optim.Adam): 
    optimizer = opt_func(model.parameters(), lr) 
    history = []
    for epoch in range(epochs): 
        # Training phase 
        model.train() 
        train_losses = []
        for batch in train_dl: 
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward() 
            optimizer.step()  
            optimizer.zero_grad() 

        # Evaluation Phase 
        result = evaluate(model, val_dl) 
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result) 
        history.append(result) 
    return history


"""Before we begin training, let's instantiate the model once again and see how 
it performs on the validation set with the inial set of parameters"""
result = evaluate(model, val_dl) 
print(result)

"""The initial accuracy is around 10%, which is what one might expect from a randomly 
initialized model (since it has a 1 in 10 chance of guessing a label right by guessing randomly). 
We'll use the following hyperparameters (learning rate, no. of epochs, batch_size etc) to train our 
model."""
num_epochs = 10 
opt_func = torch.optim.Adam 
lr = 0.001 

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func) 

"""We can also plot the validation set accuracies to study how the model improves
over time""" 

def plot_accuracies(history): 
    accuracies = [x["val_acc"] for x in history] 
    plt.figure(figsize=(16, 10))
    plt.plot(accuracies, "-x") 
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs No. of epochs") 
    plt.show()

#plot_accuracies(history) 

"""Our model reaches an accuracy of around 75%, and by looking at the graph, it seems unlikely
that the model will achieve an accruacy higher than 80% even after training for a long time. This 
suggests that we might need to use a more powerful model to capture the relationship 
between the images and the labels. This can be done by adding more convolutional layers to our model, or increasing the no. of channels in each 
convolutional layer, or by using regularization techniques.

We can also plot the training and validation losses to study the trend""" 
def plot_losses(history): 
    train_losses = [x["train_loss"] for x in history]
    val_losses = [x["val_loss"] for x in history] 
    plt.figure()
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx") 
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"]) 
    plt.title("Loss vs No. of epochs")  
    plt.show() 

plot_losses(history)