from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from pathlib import Path
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
    def __init__(self,image_path,target_path):
        self.image_path = image_path
        self.target_path = target_path
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
    