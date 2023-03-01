import torch
from torch.optim import Adam
import torch.nn as nn
import torchmetrics
from torchmetrics.functional import dice_score

from model import UNet
from data import train_loader,val_loader

# Initialise UNet model
model = UNet()

# Set learning rate
lr = 0.0001
# Set number of epochs
epochs = 2
# Define loss function
loss = dice_score()
# Define optimizer
optimizer = Adam(model.parameters(),lr=lr)
# Define metrics
metrics = [torchmetrics.IoU(threshold=0.5)]