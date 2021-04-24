# Code to demonstrate use
# of __getitem__() in python
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
from PIL import Image
from PIL import ImageFile
import cv2
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Scale
import albumentations as A
import BBtransform
from BBtransform import BBtrans

ImageFile.LOAD_TRUNCATED_IMAGES = True
COLUMN_NAMES = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width',
                'image_height']
  
class Sku(Dataset):
      
    # This function prints the type
    # of the object passed as well
    # as the object item
    def __init__(self,csv_file, root_dir, transform):
        self.df = pd.read_csv(csv_file, names=COLUMN_NAMES)
        self.root_dir = root_dir
        self.transform = transform
                            
        groupby = list(self.df.groupby(['image_name',
                                   'image_width',
                                   'image_height']))
        self.images = [
            image_name
            for (image_name, _, _), group
            in groupby
        ]
        
        self.size = [
            {
            "size": group[['image_width', 'image_height']].values
            }
            for (image_name, width, height), group
            in groupby]
            
        self.targets = [
            {
            "boxes": group[['x1', 'y1', 'x2', 'y2']].values,
            "labels": np.array([0]*len(group))
            }
            for (image_name, width, height), group
            in groupby]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = cv2.imread(img_name)
        target = self.targets[idx]
        size = self.size[idx]
        #import pdb; pdb.set_trace()
        if(self.transform is not None):
            target = BBtrans()(target, size)
            transformed = self.transform(image=image, bboxes=target['boxes'], class_labels=target['labels'])
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'])
            target['labels'] = torch.tensor(transformed['class_labels'])
            pil_image=Image.fromarray(image)
            image = ToTensor()(pil_image)
    
        return image, target
        
# Driver code
#train_set = Sku(csv_file = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/annotations/annotations_train.csv',root_dir = '../../../datasets/sku110k/images')

#train_set[0]
#train_set[1000]
#train_set[100000]


