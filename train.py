import torch
from torch.optim import Adam
import torch.nn as nn


from model import UNet
from data import train_loader,val_loader

model = UNet()

lr = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(),lr=lr)