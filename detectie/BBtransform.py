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
        colum0 = b[:,0]
        colum1 = b[:,1]
        colum2 = b[:,2]
        colum3 = b[:,3]
        
        colum2[colum2 > w] = w
        colum3[colum3 > h] = h
        
        testarray1 = colum2>colum0
        testarray2 = colum3>colum1
        
        newcolum0 = np.where(testarray1, colum0, colum2)
        newcolum2 = np.where(testarray1, colum2, colum0)
        
        newcolum1 = np.where(testarray2, colum1, colum3)
        newcolum3 = np.where(testarray2, colum3, colum1)
        
        b[:,0] = newcolum0
        b[:,1] = newcolum1
        b[:,2] = newcolum2
        b[:,3] = newcolum3
        
        return target

