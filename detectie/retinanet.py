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


class RetinaNetLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        #self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        losses = self.model(x,y)
        tot = losses['classification'] + losses['bbox_regression']
        self.log("loss_epoch_class", losses['classification'], on_step=True, on_epoch=True)
        self.log("loss_epoch_bb", losses['bbox_regression'], on_step=True, on_epoch=True)
        self.log("loss_epoch", tot, on_step=True, on_epoch=True)
        return losses['classification'] + losses['bbox_regression']
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        losses = self.model(x,y)
        #self.log("valid_loss", losses, on_step=True, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        losses = self.model(x,y)
        #self.log("test_loss", losses, on_step=True, on_epoch=True)
       

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
