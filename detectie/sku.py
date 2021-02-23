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


  



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=256):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def take(self, sample):
        image, boundingbox = sample['image'], sample['boundingbox']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        boundingbox = boundingbox * [new_w / w, new_h / h]

        return {'image': img, 'boundingbox': boundingbox}




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['boundingbox']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.double()
        image = image/255
        return {'image': torch.from_numpy(image),
                'boundingbox': torch.from_numpy(boundingbox)}
  
  
class Sku(Dataset):
      
    # This function prints the type
    # of the object passed as well
    # as the object item
    def __init__(self,csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        #print(img_name)
        image = io.imread(img_name)
        
        output_size = 256

        h, w = image.shape[:2]
        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_h, new_w))


    # h and w are swapped for landmarks because for images,
    # x and y axes are axis 1 and 0 respectively
        
        #print(type(image))
        boundingbox = self.annotations.iloc[idx,1:5]
        #print(boundingbox[1])
        boundingbox = np.array([boundingbox])
        boundingbox = boundingbox.astype('float').reshape(-1, 2)
        boundingbox = boundingbox * [new_w / w, new_h / h]
        sample = {'image': img, 'boundingbox': boundingbox}
        
        
        
        image, boundingbox = sample['image'], sample['boundingbox']

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.float()
        image = image/255
        print(image)
        #print(image.type)
        image = image[0:3, 0:256, 0:256]
        #print(image.shape)
        sample =  {'image': image, 'boundingbox': torch.from_numpy(boundingbox)}

        #print(sample['image'][0])
        return (sample)
        

        
# Driver code
train_set = Sku(csv_file = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/annotations/annotations_train.csv',root_dir = '../../../datasets/sku110k/images')

train_set[0]
train_set[1000]
train_set[100000]


