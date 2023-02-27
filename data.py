import torch    
from torch.utils.data import Dataset

import glob
from pathlib import Path
import os

root_dir = 'Dataset/'
path = Path(root_dir)
image_paths = list(path.glob('*/*.png'))

images = [str(image_path) for image_path in image_paths if '_mask' not in str(image_path)]