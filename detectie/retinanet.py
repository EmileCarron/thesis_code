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
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        losses = self.model(x,y)
        return losses['classification'] + losses['bbox_regression']
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        losses = self.model(x,y)
        #self.log('valid_loss', losses)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        y_hat = torch.argmax(y_hat, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
