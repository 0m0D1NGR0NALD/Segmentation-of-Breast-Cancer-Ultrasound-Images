import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

# Set up batch size
batch_size = 4
# Root directory to dataset
root_dir = 'Dataset/'
# Instantiate root directory of dataset to path 
path = Path(root_dir)
# Creating list of both image and mask paths
image_paths = list(path.glob('*/*.png'))
# Creating list of image paths
images = [str(image_path) for image_path in image_paths if '_mask' not in str(image_path)]
# Creating list of mask paths
masks = [str(image_path) for image_path in image_paths if '_mask' in str(image_path)]

class CustomDataset(Dataset):
    def __init__(self,images:list,masks:list):
        # Store image and mask paths as well as transforms
        self.images = images
        self.masks = masks
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((224,224))])
        
    def __getitem__(self,index):
        # Capture image and mask path from the current index
        image_path = self.images[index]
        mask_path = self.masks[index]
        # Load image
        image = cv2.imread(image_path)
        # Swap the channels
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # Read the respective mask
        mask = cv2.imread(mask_path,0)
        # Check to see if we are applying any transforms
        if self.transforms is not None:
            # Performing transforms to image and mask
            images = self.transforms(image)
            masks = self.transforms(mask)
        # Return tuple of images and their respective masks
        return (images,masks)
    
    def __len__(self):
        # Return the number of total images contained in the dataset
        return len(self.images)
    
# Splitting images and their respective masks into train, test and validation sets
train_data,test_data,train_data_masks,test_data_masks = train_test_split(images,masks,test_size=0.15,shuffle=True,random_state=12)
train_data,val_data,train_data_masks,val_data_masks = train_test_split(train_data,train_data_masks,test_size=0.15,shuffle=True,random_state=12)

# Creating train, val and test Datasets
train_dataset = CustomDataset(images=train_data,masks=train_data_masks)
val_dataset = CustomDataset(images=val_data,masks=val_data_masks)
test_dataset = CustomDataset(images=test_data,masks=test_data_masks)

# Creating train, val and test DataLoaders
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
