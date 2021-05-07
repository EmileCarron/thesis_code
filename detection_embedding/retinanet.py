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
from torch.nn import CosineSimilarity
import wandb
from collections import OrderedDict
#import tensorflow as tf

model_urls = {
    'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
}

class HeadJDE(RetinaNetHead):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__(in_channels, num_anchors, num_classes)
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)
        self.embedding_head = RetinaNetEmbeddingHead(in_channels, num_anchors, 512)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
            'embedding': self.embedding_head.compute_loss(targets, head_outputs, matched_idxs),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x),
            'embedding': self.embedding_head(x),
        }

class RetinaNetEmbeddingHead(RetinaNetClassificationHead):
    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01):
        super().__init__(in_channels, num_anchors, num_classes, prior_probability=0.01) 
        self.cos = CosineSimilarity()

    def compute_loss(self, targets, head_outputs, matched_idxs): 
        #import pdb; pdb.set_trace()
        losses = []

        cls_logits = head_outputs['embedding']

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            # gt_classes_target = torch.zeros_like(cls_logits_per_image)
            # gt_classes_target[
            #     foreground_idxs_per_image,
            #     targets_per_image['embedding'][matched_idxs_per_image[foreground_idxs_per_image]]
            # ] = 1.0

         
            self.fc = nn.Linear(cls_logits_per_image.size()[0], targets_per_image['embedding'].size()[0])
            cls_logits_per_image = torch.transpose(cls_logits_per_image, 0, 1)
            cls_logits_per_image = self.fc(cls_logits_per_image)
            cls_logits_per_image = torch.transpose(cls_logits_per_image, 0, 1)

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            loss = self.cos(cls_logits_per_image, targets_per_image['embedding']) 
            loss = torch.add(loss, 1)
            loss = torch.div(loss, 2)
            loss = torch.sum(loss)
            loss = torch.div(loss, cls_logits_per_image.size()[0])
            losses.append(loss)

        return _sum(losses) / len(targets) 
        


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

    def get_extractor(self):
        return self.extractor
            
class RetinaNetLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
            )
        self.anchor_generator = anchor_generator
        self.backbone = self.backbone1(False)
        self.head = HeadJDE(self.backbone.out_channels, self.anchor_generator.num_anchors_per_location()[0], 195)

        #self.backbone.fc = nn.Linear(512, 2, True)
        self.model = models.detection.RetinaNet(self.backbone, num_classes = 195, head= self.head)
        #self.model = models.detection.retinanet_resnet50_fpn(pretrained=True)
        # state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
        #                                       progress=True)
        # self.model.load_state_dict(state_dict)
        # overwrite_eps(self.model, 0.0)
        

        self.args = args
        self.save_hyperparameters()
        self.teacher_model = self.teacher(args)
        self.tm_full = self.teacher_model.get_model()
        self.tm_extractor = self.teacher_model.get_extractor()
        self.data_dir = args.data_dir
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
        #with torch.no_grad():
        x, y = batch
        y = [{'boxes': b, 'labels': l, 'embedding': e}
        for b, l, e in zip(y['boxes'],y['labels'], y['embedding'])
        ]
        
        boxes = y[0]['boxes'].int()
        counter = 0
        for idx in boxes:
            #import pdb; pdb.set_trace()
            height = idx[3]-idx[1]
            width = idx[2]-idx[0]
            if height < 7:
                #import pdb; pdb.set_trace()
                height = 7
            if width < 7:
                #import pdb; pdb.set_trace()
                width = 7

            image = torchvision.transforms.functional.crop(x, idx[1], idx[0], height, width)
            self.tm_full.eval()
            self.tm_extractor.eval()
            predictions = self.tm_full(image)
            predictions_embedding = self.tm_extractor(image)
            #embedding_path = self.data_dir + '/SKU110K/annotations/embeddings/embedding' + str(counter)+'.pt'
            #torch.save(predictions, embedding_path)
            _, predicted = torch.max(predictions.data, 1)
            predictions_embedding = torch.squeeze(predictions_embedding)
            predictions_embedding = predictions_embedding.clone().detach().numpy()
            predictions_embedding = torch.Tensor(predictions_embedding)
            y[0]['labels'][counter] = predicted 
            
                #test = torch.no_grad(predictions)
            y[0]['embedding'][counter] = predictions_embedding
            counter = counter + 1
            
            


            #self.logger.experiment.log({"input image":[wandb.Image(x, caption="val_input_image")]})
            #self.logger.experiment.log({"bbx image":[wandb.Image(image, caption="val_input_image")]})

        
        
        import pdb; pdb.set_trace()
        losses = self.model(x,y)
        tot = losses['classification'] + losses['bbox_regression']
        self.log("loss_training_class", losses['classification'], on_step=True, on_epoch=True)
        self.log("loss_training_bb", losses['bbox_regression'], on_step=True, on_epoch=True)
        self.log("loss_training", tot, on_step=True, on_epoch=True)
        return losses['classification'] + losses['bbox_regression']
        
    # def validation_step(self, batch, batch_idx):
    #     #import pdb; pdb.set_trace()
    #     x, y = batch
    #     y = [{'boxes': b, 'labels': l, 'embedding': e}
    #     for b, l, e in zip(y['boxes'],y['labels'], y['embedding'])
    #     ]

    #     boxes = y[0]['boxes'].int()
    #     counter = 0
    #     for idx in boxes:
    #         height = idx[3]-idx[1]
    #         width = idx[2]-idx[0]
    #         if height < 7:
    #             #import pdb; pdb.set_trace()
    #             height = 7
    #         if width < 7:
    #             #import pdb; pdb.set_trace()
    #             width = 7

    #         image = torchvision.transforms.functional.crop(x, idx[1], idx[0], height, width)
    #         self.tm.eval()
    #         predictions = self.tm(image)
    #         _, predicted = torch.max(predictions.data, 1)
    #         y[0]['labels'][counter] = predicted 
    #         y[0]['embedding'][counter] = predictions
                
    #         counter = counter + 1

    #     detections = self.model(x,y)
    #     #if detections[0]['scores'].size() != torch.Size([0]):
    #         #self.log("valid_score", detections[0]['scores'][0], on_step=True, on_epoch=True)

    #     #self.log("valid_score", detections[0]['scores'][0], on_step=True, on_epoch=True)
    #     return detections
       
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.args.lr,
                                    weight_decay = self.args.weight_decay)
        return optimizer
