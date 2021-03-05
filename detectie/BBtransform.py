import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
from PIL import Image
import cv2

class BBtrans(Dataset):
      
    # This function prints the type
    # of the object passed as well
    # as the object item
    #def __init__(self):
   
    def __call__(self, target, size, image):
        h = image.size()[1]
        w = image.size()[2]
        #print(target)
        for x in target['boxes']:
            x1 = x[0]
            y1 = x[1]
            x2 = x[2]
            y2 = x[3]
            
            if h > w:
                new_h, new_w = size * h / w, size
            else:
                new_h, new_w = size, size * w / h
            
            newx1 = x1 * (new_w / w)
            newy1 = y1 * (new_h / h)
            newx2 = x2 * (new_w / w)
            newy2 = y2 * (new_h / h)
            
            if newx1 >= 300:
                newx1 = 299
            
            if newx2 >= 300:
                newx2 = 300
                        
            if newy1 >= 300:
                newy1 = 299
            
            if newy2 >= 300:
                newy2 = 300
                
            x[0] = newx1
            x[1] = newy1
            x[2] = newx2
            x[3] = newy2
        
        #print(target)
        return target
        
