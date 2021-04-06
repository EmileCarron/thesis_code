import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
from PIL import Image
import cv2

class BBtrans(Dataset):
      
    def __call__(self, target, size):
        size1 = size["size"][0]
        w = size["size"][0][0]
        h = size["size"][0][1]
        counter = 0
        #import pdb; pdb.set_trace()
        b = target['boxes']
        l = target['labels']
        for x in target['boxes']:
            x1 = x[0]
            y1 = x[1]
            x2 = x[2]
            y2 = x[3]
            
            if x2 >= w:
                b = np.delete(b, counter, 0)
                l = np.delete(l, counter, 0)
                counter = counter - 1

            if y2 >= h:
                b = np.delete(b, counter, 0)
                l = np.delete(l, counter, 0)
                counter = counter - 1
                
            if x1 >= x2:
                b = np.delete(b, counter, 0)
                l = np.delete(l, counter, 0)
                counter = counter - 1
            
            if y1 >= y2:
                b = np.delete(b, counter, 0)
                l = np.delete(l, counter, 0)
                counter = counter - 1
                
            counter = counter + 1
        target['boxes'] = b
        target['labels'] = l
        return target

