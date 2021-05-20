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
from torchvision.models.detection import RetinaNet
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops import boxes as box_ops
from torch.nn import CosineEmbeddingLoss
from torch.nn import CosineSimilarity
import wandb
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from torch import nn, Tensor
#import tensorflow as tf

model_urls = {
    'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
}

def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

class HeadJDE(RetinaNetHead):
    def __init__(self, in_channels, num_anchors, num_classes, args):
        super().__init__(in_channels, num_anchors, num_classes)
        self.args = args
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)
        self.embedding_head = RetinaNetEmbeddingHead(in_channels, num_anchors, args.embedding_size, self.args)

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
    def __init__(self, in_channels, num_anchors, num_classes, args, prior_probability=0.01):
        super().__init__(in_channels, num_anchors, num_classes, prior_probability=0.01) 
        self.args = args
        self.cos = CosineSimilarity()

    def compute_loss(self, targets, head_outputs, matched_idxs):
        
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs['embedding']

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[foreground_idxs_per_image, targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]]] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(sigmoid_focal_loss(cls_logits_per_image[valid_idxs_per_image],gt_classes_target[valid_idxs_per_image],reduction='sum',) / max(1, num_foreground))

        return _sum(losses) / len(targets)

class RetinaNetEmbedding(RetinaNet):
    def __init__(self, backbone, num_classes,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # Anchor parameters
                 anchor_generator=None, head=None,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000):
        super().__init__(backbone, num_classes,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # Anchor parameters
                 anchor_generator=None, head=head,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000)
       

    def eager_outputs(self, losses, detections, embedding512, embedding195):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections, embedding512, embedding195

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
            # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs['cls_logits']
        box_regression = head_outputs['bbox_regression']
        costum_embedding = head_outputs['embedding']

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]
            custom_embedding_per_image = [emb[index] for emb in costum_embedding]

            image_boxes = []
            image_scores = []
            image_labels = []
            image_embedding = []
            image_custom_embedding = []

            for box_regression_per_level, logits_per_level, anchors_per_level, custom_embedding_per_level in \
                    zip(box_regression_per_image, logits_per_image, anchors_per_image, custom_embedding_per_image):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]
                indxembd = torch.div(idxs, num_classes, rounding_mode='floor')
                embeddings_per_level = logits_per_level[indxembd]
                custom_embedding_per_level = custom_embedding_per_level[indxembd]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode='floor')
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                               anchors_per_level[anchor_idxs])
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)
                image_embedding.append(embeddings_per_level)
                image_custom_embedding.append(custom_embedding_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_embedding = torch.cat(image_embedding, dim=0)
            image_custom_embedding = torch.cat(image_custom_embedding, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
                'embeddings': image_embedding[keep],
                'custom_embedding': image_custom_embedding[keep],
            })

        return detections

            

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None

            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs['cls_logits'].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections, head_outputs['embedding'], head_outputs['cls_logits'])


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

    def get_classifier(self):
        return self.classifier

    def get_model(self):
        return self.model

    def get_extractor(self):
        return self.extractor
            
class RetinaNetLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
            )
        self.anchor_generator = anchor_generator
        self.backbone = self.backbone1(True)
        self.head = HeadJDE(self.backbone.out_channels, self.anchor_generator.num_anchors_per_location()[0], 195, self.args)
        self.model = RetinaNetEmbedding(self.backbone, num_classes = 195, head= self.head)  
        self.save_hyperparameters()
        self.teacher_model = self.teacher(args)
        self.tm_full = self.teacher_model.get_model()
        self.tm_extractor = self.teacher_model.get_extractor()
        self.data_dir = args.data_dir

    def teacher(self, args):
        teacher = RecognitionModel(args)
        teacher_model = teacher.load_from_checkpoint(checkpoint_path=args.checkpoint, args=args)
        return teacher_model


    def backbone1(self, pretrained_backbone, pretrained=True, trainable_backbone_layers=None):
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
        y = [{'boxes': b, 'labels': l, 'embedding': e}
        for b, l, e in zip(y['boxes'],y['labels'], y['embedding'])
        ]
        
        boxes = y[0]['boxes'].int()
        counter = 0
        for idx in boxes:
            height = idx[3]-idx[1]
            width = idx[2]-idx[0]
            if height < 7:
                height = 7
            if width < 7:
                width = 7

            image = torchvision.transforms.functional.crop(x, idx[1], idx[0], height, width)
            self.tm_full.eval()
            self.tm_extractor.eval()
            predictions = self.tm_full(image)
            _, predicted = torch.max(predictions.data, 1)
            y[0]['labels'][counter] = predicted 
            counter = counter + 1

        losses = self.model(x,y)
        tot = (losses['classification'] + losses['bbox_regression'] + losses['embedding'])/3
        self.log("loss_training_class", losses['classification'], on_step=True, on_epoch=True)
        self.log("loss_training_bb", losses['bbox_regression'], on_step=True, on_epoch=True)
        self.log("loss_training_embedding", losses['embedding'], on_step=True, on_epoch=True)
        self.log("loss_training", tot, on_step=True, on_epoch=True)
        return (losses['classification'] + losses['bbox_regression'] + losses['embedding'])/3
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = [{'boxes': b, 'labels': l, 'embedding': e}
        for b, l, e in zip(y['boxes'],y['labels'], y['embedding'])
        ]

        boxes = y[0]['boxes'].int()
        counter = 0
        for idx in boxes:
            height = idx[3]-idx[1]
            width = idx[2]-idx[0]
            if height < 7:
                height = 7
            if width < 7:
                width = 7

            image = torchvision.transforms.functional.crop(x, idx[1], idx[0], height, width)
            self.tm_full.eval()
            self.tm_extractor.eval()
            predictions = self.tm_full(image)
            _, predicted = torch.max(predictions.data, 1)
            y[0]['labels'][counter] = predicted 
            counter = counter + 1

        detections, embedding512, embedding195 = self.model(x,y)
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
