import torch
from torch import nn 
from torch import optim
import pytorch_lightning as pl 
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import sku
from sku import Sku


if __name__ == '__main':
    
    train_set = Sku(csv_file = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/annotations/annotations_train.csv',root_dir = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/images')
    val_set = Sku(csv_file = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef//Tutorial/5/SKU110K_fixed/annotations/annotations_val.csv',root_dir = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/images')
    
    train= DataLoader(train_set, batch_size=12, num_workers=4)
    val = DataLoader(val_set, batch_size=12, num_workers=4)
    
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False, progress=True, num_classes=1, pretrained_backbone=True, **kwargs
    model.eval()
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=10,)
    trainer.fit(model, train, val)
