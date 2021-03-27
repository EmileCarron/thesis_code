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
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class RecognitionModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        

#    def forward(self, x):
#        return self.model(x)
#
#    def resnet18(pretrained: bool = False, progress: bool = True, **kwargs):
#        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)


    def training_step(self, batch, batch_idx):
        x, labels = batch
        out = self.model(x)
        
        loss = torch.nn.CrossEntropyLoss()(out, labels)
        accuracy = self.accuracy(out, labels)
        self.log("loss_training_class", loss, on_step=True, on_epoch=True)
        self.log("accuaracy_training_class", accuracy, on_step=True, on_epoch=True)
        
        return loss
          
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        embeddings = self.model(x)


        out = embeddings

        loss = torch.nn.CrossEntropyLoss()(out, labels)

        labels = labels.cpu().numpy()
        embeddings = embeddings.cpu().numpy()
        self.log("loss_validation_class", loss, on_step=True, on_epoch=True)
        return loss

        


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
