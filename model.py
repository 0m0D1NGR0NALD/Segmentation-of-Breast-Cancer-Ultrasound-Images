import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        # Store convolution and relu layers
        self.conv1 = nn.Conv2d(in_channels,out_channels,3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,3)

    def forward(self,x):
        # Apply convolution>>relu>>convolution to inputs and return output
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self,channels=(3,64,128,256,512,1024)):
        super().__init__()
        # Store encoder blocks and maxpooling layer
        self.encoder_blocks = nn.ModuleList([Block(channels[i],channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        # Initialize empty list to store intermediate outputs
        features = []
        # Loop through the encoder blocks
        for block in self.encoder_blocks:
            # Pass the inputs through the current encoder block 
            x = block(x)
            # Append and store outputs to list
            features.append(x)
            # Apply maxpooling on the output
            x = self.pool(x)
        # Return list with intermediate outputs
        return features
    
class Decoder(nn.Module):
    def __init__(self,channels=(1024,512,256,128,64)):
        super().__init__()
        # Initialize the number of channels, upconvolution blocks and decoder blocks
        self.channels = channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i],channels[i+1],2,2) for i in range(len(channels)-1)])
        self.decoder_blocks = nn.ModuleList([Block(channels[i],channels[i+1]) for i in range(len(channels)-1)])
    
    def forward(self,x,encoder_features):
        # Loop through the number of channels
        for i in range(len(self.channels)-1):
            # Pass the inputs through the upconvolution blocks
            x = self.upconvs[i](x)
            # Crop the current features from the encoder blocks
            enc_features = self.crop(encoder_features[i],x)
            # Concatenate the features with the current upconvolution features
            x = torch.cat([x,enc_features],dim=1)
            # Pass the concatenated output through the current decoder block
            x = self.decoder_blocks[i](x)
        # Return the final decoder output
        return x

    def crop(self,encoder_features,x):
        # Capture the dimensions of the inputs 
        (_,_,H,W) = x.shape
        # Crop the encoder features to match the dimensions
        encoder_features = torchvision.transforms.CenterCrop([H,W])(encoder_features)
        # Return the cropped features
        return encoder_features
    
class UNet(nn.Module):
    def __init__(self,encoder_channels=(3,64,128,256,512,1024),decoder_channels=(1024,512,256,128,64),n_classes=1,retain_dim=False,out_size=(572,572)):
        super().__init__()
        # Initialize the encoder and decoder
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        # Initailizethe regression head and store the class variables
        self.head  = nn.Conv2d(decoder_channels[-1],n_classes,1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self,x):
        # Capture the features fromthe encoder
        encoder_features = self.encoder(x)
        # Pass the encoder features through the decoder
        # Ensuring the dimensions are suited for concatenation
        out = self.decoder(encoder_features[::-1][0],encoder_features[::-1][1:])
        # Pass the decoder features through the regression head to obtain segmentation mask
        out = self.head(out)
        # Chneck to see if we are retraining the original output
        # Then, resize the output to match them
        if self.retain_dim:
            out = F.interpolate(out,self.out_size)
        # Return the segmentation map
        return out

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))   

class unet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
             
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
