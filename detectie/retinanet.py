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
import wandb
from collections import OrderedDict
#import tensorflow as tf

model_urls = {
    'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
}


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

    def get_model(self):
        return self.model
            
class RetinaNetLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),)
        )
        self.backbone = self.backbone1(False)
        #self.backbone.fc = nn.Linear(512, 2, True)
        self.model = models.detection.RetinaNet(self.backbone, num_classes = 195)
        #self.model = models.detection.retinanet_resnet50_fpn(pretrained=True)
        # state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
        #                                       progress=True)
        # self.model.load_state_dict(state_dict)
        # overwrite_eps(self.model, 0.0)
        #self.teacher_model = torchvision.models.resnet18(pretrained=True)

        self.args = args
        self.save_hyperparameters()
        self.teacher_model = self.teacher(args)
        self.tm = self.teacher_model.get_model()
        #self.teacher_model.train(False)

    def teacher(self, args):
        teacher = RecognitionModel(args)
        teacher_model = teacher.load_from_checkpoint(checkpoint_path=args.checkpoint, args=args)
        return teacher_model


    def backbone1(self, pretrained_backbone, pretrained=False, trainable_backbone_layers=None):
        trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

        if pretrained:
            # no need to download the backbone if pretrained is set
            pretrained_backbone = False
        # skip P2 because it generates too many anchors (according to their paper)
        backbone = resnet_fpn_backbone('resnet18', pretrained_backbone, returned_layers=[2, 3, 4],
                                       extra_blocks=LastLevelP6P7(256, 256), trainable_layers=trainable_backbone_layers)
        return backbone
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        
        boxes = y[0]['boxes'].int()
        counter = 0
        for idx in boxes:
            height = idx[3]-idx[1]
            width = idx[2]-idx[0]
            if height < 7:
                #import pdb; pdb.set_trace()
                height = 7
            if width < 7:
                #import pdb; pdb.set_trace()
                width = 7

            image = torchvision.transforms.functional.crop(x, idx[1], idx[0], height, width)
            self.tm.eval()
            predictions = self.tm(image)
            _, predicted = torch.max(predictions.data, 1)
            y[0]['labels'][counter] = predicted 
            counter = counter + 1

            #self.logger.experiment.log({"input image":[wandb.Image(x, caption="val_input_image")]})
            #self.logger.experiment.log({"bbx image":[wandb.Image(image, caption="val_input_image")]})

        

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
        detections = self.model(x,y)
        #if detections[0]['scores'].size() != torch.Size([0]):
            #self.log("valid_score", detections[0]['scores'][0], on_step=True, on_epoch=True)

        #self.log("valid_score", detections[0]['scores'][0], on_step=True, on_epoch=True)
        return detections
       
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.args.lr,
                                    weight_decay = self.args.weight_decay)
        return optimizer
