import sys
import os
import torch
from torch import nn 
from torch import optim
import pytorch_lightning as pl 
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from sku import Sku
from torchvision import models


class RetinaNetLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = models.detection.retinanet_resnet50_fpn(pretrained=True)
        self.args = args
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        #import pdb; pdb.set_trace()
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        losses = self.model(x,y)
        tot = losses['classification'] + losses['bbox_regression']
        self.log("loss_training_class", losses['classification'], on_step=True, on_epoch=True)
        self.log("loss_training_bb", losses['bbox_regression'], on_step=True, on_epoch=True)
        self.log("loss_training", tot, on_step=True, on_epoch=True)
        return losses['classification'] + losses['bbox_regression']
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        detections = self.model(x,y)
        #print(losses[0]['scores'])
        #loss = torch.argmax(losses[0]['scores'])
        #self.log("valid_score", loss, on_step=True, on_epoch=True)
       
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.args.lr,
                                    weight_decay = self.args.weight_decay)
        return optimizer
