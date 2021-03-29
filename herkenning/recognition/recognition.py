from collections import OrderedDict
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
from torchvision.models import resnet, resnet18
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import losses




class RecognitionModel(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.args = args
        self.num_classes = self.args.num_classes

        if self.args.loss == 'CrossEntropy':
            self.loss = torch.nn.CrossEntropyLoss()
            
        elif self.args.loss == 'ArcFace':
            self.loss = losses.ArcFaceLoss(
                margin=0.5,
                num_classes=self.num_classes,
                embedding_size = 256
            )
            
        elif self.args.loss == 'CosFace':
            self.loss = losses.CosFaceLoss(
                num_classes = self.num_classes,
                embedding_size = 512,
                margin=0.35,
                scale=64
            )
 
        elif self.args.loss == 'TripletMargin':
            self.loss = losses.TripletMarginLoss(
                margin=0.5
                )
            
        elif self.args.loss == 'CircleLoss':
            self.loss = losses.CircleLoss(m=0.4, gamma=80)

        else:
            raise ValueError(f'Unsupported loss: {self.args.loss}')



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
        if self.args.optim == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.args.lr,
                             weight_decay=self.args.weight_decay)
        elif self.args.optim == 'SGD':
            optimizer = SGD(self.parameters(), lr=self.args.lr,
                            weight_decay=self.args.weight_decay,
                            momentum=self.args.momentum)

        return optimizer
