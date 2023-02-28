from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import os

root_dir = 'Dataset/'
path = Path(root_dir)
image_paths = list(path.glob('*/*.png'))

images = [str(image_path) for image_path in image_paths if '_mask' not in str(image_path)]
masks = [str(image_path) for image_path in image_paths if '_mask' in str(image_path)]
labels = [os.path.split(os.path.split(name)[0])[1] for name in images]

classes = list(set(labels))
labels_dict = {label: i for i,label in enumerate(classes)}
labels = [labels_dict[label_key] for label_key in labels]

class CustomDataset(Dataset):
    def __init__(self,images:list,masks:list,labels:list):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
        
    def __getitem__(self,index):
        image = Image.open(self.images[index])
        images = self.transforms(image)
        mask = Image.open(self.masks[index])
        masks = self.transforms(mask)
        labels = self.labels
        return images,masks,labels
    
    def __len__(self):
        return len(self.images)
    
# Splitting data into train, test and validation set
train_data,test_data,train_data_masks,test_data_masks,train_labels,test_labels = train_test_split(images,masks,labels,test_size=0.15,shuffle=True,random_state=12)
train_data,val_data,train_data_masks,val_data_masks,train_labels,val_labels = train_test_split(train_data,train_data_masks,train_labels,test_size=0.15,shuffle=True,random_state=12)

# Setting train, val and test DataLoaders
train_dataset = CustomDataset(images=train_data,masks=train_data_masks,labels=train_labels)
val_dataset = CustomDataset(images=val_data,masks=val_data_masks,labels=val_labels)
test_dataset = CustomDataset(images=test_data,masks=test_data_masks,labels=test_labels)



