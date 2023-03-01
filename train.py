from model import UNet
from data import train_loader,val_loader

model = UNet()

criterion = nn.CrossEntropyLoss()