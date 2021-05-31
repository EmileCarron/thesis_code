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
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.retinanet import RetinaNetRegressionHead
from torch.nn import CosineEmbeddingLoss
import wandb
from collections import OrderedDict


model_urls = {
    'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
}

#Training a stock retinanet for comparing with our joint network
            
class RetinaNetLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = models.detection.retinanet_resnet50_fpn(pretrained=True)
        self.args = args
        self.save_hyperparameters()
        self.data_dir = args.data_dir
        
    def training_step(self, batch, batch_idx):
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
        scores = detections[0]['scores']
        length = scores.size()[0]
        totscore = torch.sum(scores)
        totscore = torch.div(totscore, length)
        self.log("Score", totscore, on_step=True, on_epoch=True)
        return detections
       
    def configure_optimizers(self):
        if self.args.optim == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.args.lr,
                             weight_decay=self.args.weight_decay)
                             
        #Stochastic Gradient Descent is usefull when you have a lot of redundancy in your data
        elif self.args.optim == 'SGD':
            optimizer = SGD(self.parameters(), 
                            lr=self.args.lr,
                            weight_decay=self.args.weight_decay,
                            momentum=self.args.momentum)
                            
        return optimizer
