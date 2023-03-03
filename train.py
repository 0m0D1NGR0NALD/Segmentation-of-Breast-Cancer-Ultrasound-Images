import torch
from torch.optim import Adam
import torch.nn as nn

from model import UNet
from data import train_loader,val_loader,train_dataset,val_dataset,batch_size

# Initialise UNet model
model = UNet()
# Set learning rate
lr = 0.0001
# Set number of epochs
epochs = 2
# Define loss function
loss = nn.BCEWithLogitsLoss()
# Define optimizer
optimizer = Adam(model.parameters(),lr=lr)
# Calculate steps per epoch for training, validation and test set
train_steps = len(train_dataset)//batch_size
val_steps = len(val_dataset)//batch_size
# Initialize a dictionary store training history
history = {"train_loss":[],"val_loss":[]}

