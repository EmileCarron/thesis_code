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

model_urls = {
    'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
}



class RetinaNetLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),)
        )
        self.backbone = self.backbone1(False)
        #self.backbone.fc = nn.Linear(512, 2, True)
        self.model = models.detection.RetinaNet(self.backbone, num_classes = 2)
        #self.model = models.detection.retinanet_resnet50_fpn(pretrained=True)
        # state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
        #                                       progress=True)
        # self.model.load_state_dict(state_dict)
        # overwrite_eps(self.model, 0.0)
        self.args = args
        self.save_hyperparameters()

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
        import pdb; pdb.set_trace()
        x, y = batch
        y = [{'boxes': b, 'labels': l}
        for b, l in zip(y['boxes'],y['labels'])
        ]
        detections = self.model(x,y)
        #print(losses[0]['scores'])
        #loss = torch.argmax(detections[0]['scores'])
        #self.log("valid_score", detections[0]['scores'][0], on_step=True, on_epoch=True)
        return detections
       
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.args.lr,
                                    weight_decay = self.args.weight_decay)
        return optimizer
