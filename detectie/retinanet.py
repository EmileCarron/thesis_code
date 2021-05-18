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

#import tensorflow as tf

model_urls = {
    'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
}
            
class RetinaNetLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        #self.backbone.fc = nn.Linear(512, 2, True)
        #self.model = models.detection.RetinaNet(self.backbone, num_classes = 195)
        self.model = models.detection.retinanet_resnet50_fpn(pretrained=True)
        # state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
        #                                       progress=True)
        # self.model.load_state_dict(state_dict)
        # overwrite_eps(self.model, 0.0)
        #self.bbone = torchvision.models.resnet18(pretrained=True)

        #self.extractor = torch.nn.Sequential(
        #    OrderedDict(
        #        list(self.bbone.named_children())[:-1]
        #    ),
         
        #)
        #self.extractor.fc = nn.Linear(512, 195, True)

        # anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        # anchor_generator = AnchorGenerator(
        #     anchor_sizes, aspect_ratios
        #     )

        # self.backbone = self.backbone1(False)
        # # put the pieces together inside a RetinaNet model
        # self.model = models.detection.RetinaNet(self.backbone, num_classes = 2)

        self.args = args
        self.save_hyperparameters()
        #self.teacher_model = self.teacher(args)
        #self.tm = self.teacher_model.get_model()
        self.data_dir = args.data_dir
        #self.teacher_model.train(False)


        
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
        #import pdb; pdb.set_trace()
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]


        import pdb; pdb.set_trace()
        detections = self.model(x,y)
        scores = detections[0]['scores']
        length = scores.size()[0]
        totscore = torch.sum(scores)
        totscore = torch.div(totscore, length)
        self.log("Score", totscore, on_step=True, on_epoch=True)
        #print(detections)
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
