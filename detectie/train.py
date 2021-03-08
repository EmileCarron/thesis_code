import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import sku
from sku import Sku
from retinanet import RetinaNetLightning
import sys

def main(arguments):

    train_set = Sku(csv_file = '../../../dataset/SKU110K_fixed/annotations/annotations_train.csv',root_dir = '../../../dataset/SKU110K_fixed/images')
    val_set = Sku(csv_file = '../../../dataset/SKU110K_fixed/annotations/annotations_val.csv',root_dir = '../../../dataset/SKU110K_fixed/images')
    
    train_set, train2 = torch.utils.data.random_split(train_set, [1000, 7219])
    
    train= DataLoader(train_set, batch_size=1, num_workers=int(arguments[1]))
    val = DataLoader(val_set, batch_size=1, num_workers=int(arguments[1]))
    
    model = RetinaNetLightning()
    
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=int(arguments[2]))
    trainer.fit(model, train, val)

if __name__ == '__main__':
    main(sys.argv)
