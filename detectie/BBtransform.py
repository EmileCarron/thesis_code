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
    
    
#    def __call__(self, target, targettrans):
#
#        counter = 0
#        for x in target['boxes']:
#            #print(target)
#            #print(targettrans)
#            x[0] = targettrans['boxes'][counter][0]
#            x[1] = targettrans['boxes'][counter][1]
#            x[2] = targettrans['boxes'][counter][2]
#            x[3] = targettrans['boxes'][counter][3]
#            counter = counter + 1
#
#        #print(target)
#        #print(targettrans)
#
#        return target
   
    def __call__(self, target, size):
        size1 = size["size"][0]
        w = size["size"][0][0]
        h = size["size"][0][1]
        print(size1)
#        print(w, h)
#        print(target)
        counter = 0
        #import pdb; pdb.set_trace()
        b = target['boxes']
        l = target['labels']
        #y = t[:, np.r_[:1, 3:]]
        #t = torch.cat((y[:,:3], y[:,4:]))
        #y = np.delete(t, 2, 0)
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
        print(target['boxes'])
        target['boxes'] = b
        target['labels'] = l
        
        #print(target)
        return target

