from torch.utils.data import Dataset

from pathlib import Path
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
