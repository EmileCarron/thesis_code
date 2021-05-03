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
from collections import OrderedDict
#import tensorflow as tf

model_urls = {
    'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
}


class RecognitionModel(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        
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

    def get_model(self):
        return self.extractor
            
class RetinaNetLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),)
        )
        self.backbone = self.backbone1(False)
        self.cosloss = torch.nn.CosineEmbeddingLoss()

        self.model = models.detection.RetinaNet(self.backbone, num_classes = 195)

        self.bbone = torchvision.models.resnet18(pretrained=True)

        self.extractor = torch.nn.Sequential(
            OrderedDict(
                list(self.bbone.named_children())[:-1]
            ),
         
        )

        self.args = args
        self.save_hyperparameters()
        self.teacher_model = self.teacher(args)
        self.tm = self.teacher_model.get_model()


    def teacher(self, args):
        teacher = RecognitionModel(args)
        teacher_model = teacher.load_from_checkpoint(checkpoint_path=args.checkpoint, args=args)
        return teacher_model
        
    def embeddings(self, train_set):
        #import pdb; pdb.set_trace()
        for train_idx in range(len(train_set)):
            x, y = train_set[train_idx]
            y = [{'boxes': b, 'labels': l, 'embedding': e}
            for b, l, e in zip(y['boxes'],y['labels'], y['embedding'])
            ]

            boxes = y[0]['boxes'].int()
            counter = 0
            for idx in range(len(boxes)):
                
                height = boxes[idx][3]-boxes[idx][1]
                width = boxes[idx][2]-boxes[idx][0]
                if height < 7:
                    #import pdb; pdb.set_trace()
                    height = 7
                if width < 7:
                    #import pdb; pdb.set_trace()
                    width = 7

                image = torchvision.transforms.functional.crop(x, boxes[idx][1], boxes[idx][0], height, width)
                self.tm.eval()
                embedding_path = self.data_dir + '/annotations/embeddings/embedding' + train_idx + '_' + counter
                predictions = self.tm(image)
                _, predicted = torch.max(predictions.data, 1)
                y[0]['labels'][counter] = predicted 
                torch.save(predictions, embedding_path)
                counter = counter + 1
       
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.args.lr,
                                    weight_decay = self.args.weight_decay)
        return optimizer
