import torch
import torch.nn as nn
import torchvision

class Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels,out_channels,3)

    def forward(self,x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self,channels=(3,64,128,256,512,1024)):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([Block(channels[i],channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features
    
class Decoder(nn.Module):
    def __init__(self,channels=(1024,512,256,128,64)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i],channels[i+1],2,2) for i in range(len(channels)-1)])
        self.decoder_blocks = nn.ModuleList([Block(channels[i],channels[i+1]) for i in range(len(channels)-1)])
    
    def forward(self,x,encoder_features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            encoder_features = self.crop(encoder_features[i],x)
            x = torch.cat([x,encoder_features],dim=1)
            x = self.decoder_blocks[i](x)

    def crop(self,encoder_features,x):
        _,_,H,W = x.shape
        encoder_features = torchvision.transforms.CenterCrop([H,W](encoder_features))
        return encoder_features