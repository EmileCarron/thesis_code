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
import wandb
#import pdb; pdb.set_trace()


class RecognitionModel(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        #ckpt = '../../../Masterproef/thesis_code/recognition/wandb/run-20210409_112741-28fdpx5s/files/thesis/28fdpx5s/checkpoints/epoch=299-step=18899.ckpt' 
        
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 195, True)
        self.args = args

        self.extractor = torch.nn.Sequential(
            OrderedDict(
                list(self.model.named_children())[:-1]
            ),
         
        )

        self.classifier = torch.nn.Sequential(
            OrderedDict(
                list(self.model.named_children())[-1:]
            )
        )

        if self.args.loss == 'CrossEntropy':
            self.loss = torch.nn.CrossEntropyLoss()
            self.loss_requires_classifier = True
            
        elif self.args.loss == 'ArcFace':
            self.loss = losses.ArcFaceLoss(
            margin=0.5,
            embedding_size=self.classifier.fc.in_features,
            num_classes=self.classifier.fc.out_features
            )
            self.loss_requires_classifier = False
            
        elif self.args.loss == 'ContrastiveLoss':
            self.loss = losses.ContrastiveLoss(
            pos_margin=0,
            neg_margin=1
            )
            self.loss_requires_classifier = False
        
        #sampler toevoegen!!!! 
        elif self.args.loss == 'TripletMargin':
            self.loss = losses.TripletMarginLoss(
            margin=0.1
            )
            self.loss_requires_classifier = False
 
        elif self.args.loss == 'CircleLoss':
            self.loss = losses.CircleLoss(m=0.4,
            gamma=80
            )
            self.loss_requires_classifier = False
            
        else:
            raise ValueError(f'Unsupported loss: {self.args.loss}')

    def forward(self, x):
        return self.extractor.forward(x)


    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)


    def training_step(self, batch, batch_idx):
        import pdb; pdb.set_trace()

        x, labels = batch
        out = torch.squeeze(self(x))
        
        if self.loss_requires_classifier:
            out = self.model(x)

        loss = self.loss(out, labels)
        accuracy = self.accuracy(out, labels)

        self.log("loss_training", loss, on_step=True, on_epoch=True)
        self.log("accuaracy_training", accuracy, on_step=False, on_epoch=True)
 
        return loss
          
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        
        if self.loss_requires_classifier:
            out = self.model(x)
        else:
            out = self.extractor(x)
        loss = self.loss(out, labels)

        labels = labels.cpu().numpy()
        
        self.log("loss_validation", loss, on_step=True, on_epoch=True)
        return loss

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
