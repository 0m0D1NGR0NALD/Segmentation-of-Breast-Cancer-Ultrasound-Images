import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels,out_channels,3)

    def forward(self,x):
        return self.conv2(self.relu(self.conv1(x)))
