import torch
from torch import nn 
from torch import optim
import pytorch_lightning as pl 
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from collections import OrderedDict
from torchvision.models import resnet18

class RecognitionModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        

#    def forward(self, x):
#        return self.model(x)
#
#    def resnet18(pretrained: bool = False, progress: bool = True, **kwargs):
#        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        out = self.model(x)
        
        loss = torch.nn.CrossEntropyLoss()(out, labels)
        
        return {
            'loss': loss
          #  'log': {f'{self.hparams.loss}Loss/train': loss}
        }


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
