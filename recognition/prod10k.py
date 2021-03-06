# Code to demonstrate use
# of __getitem__() in python
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Scale


COLUMN_NAMES = ['image_name', 'classe', 'group']
  
class Prod10k(Dataset):
    def __init__(self,csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file, names=COLUMN_NAMES)
        self.root_dir = root_dir
        self.transform = transform
        groupby = list(self.df.groupby(['image_name']))
        
        self.targets = [
            {
            "classe": classe.values,
            "group": group.values
            }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.df[idx])
        image = Image.open(img_name)
        target = self.targets[idx]
        
        if(self.transform is not None):
            image, target = self.transform(image, target)
        else:
            image = ToTensor()(image)
  
            image = Resize(300,2)(image)

            image = RandomResizedCrop(300)(image)
    
        return image



