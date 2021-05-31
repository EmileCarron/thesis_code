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
from torch.nn import CosineEmbeddingLoss
from torch.nn import CosineSimilarity
import wandb
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from torch import nn, Tensor
from argparse import ArgumentParser
import cv2
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd

class RecognitionModel(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        #ckpt = '../../../Masterproef/thesis_code/recognition/wandb/run-20210409_112741-28fdpx5s/files/thesis/28fdpx5s/checkpoints/epoch=299-step=18899.ckpt' 
        
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 2378, True)
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



class Predictions(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.predclass = self.predictionclass(args)
        self.model = self.predclass.get_model()
        self.extractor = self.predclass.get_extractor()

    def predictionclass(self, args):
        pc = RecognitionModel(args)
        pc_model = pc.load_from_checkpoint(checkpoint_path=args.checkpoint, args=args)
        return pc_model

    def predictions(self, args):
        import pdb; pdb.set_trace()
        imagein = cv2.imread(args.image)
        imagelist = []
        bbox = torch.load('/Users/emilecarron/test/tankstation_results/retinanet_embedding_rp2k/labeledp1.pt')
        bbox = bbox.clone().detach()
        bbox = bbox.int()
        pil_image = Image.fromarray(imagein)
        image = ToTensor()(pil_image)
        imageor = image.unsqueeze(0)

        counter = 0
        for box in bbox:
            height = box[3]-box[1]
            width = box[2]-box[0]
            if height < 7:
                height = 7
            if width < 7:
                width = 7

            image = torchvision.transforms.functional.crop(imageor, box[1], box[0], height, width)
            self.model.eval()
            pred = self.model(image)
            _, predicted = torch.max(pred.data, 1)
            image = torchvision.transforms.functional.to_pil_image(image[0])
            image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            cv2.imwrite("/Users/emilecarron/test/tankstation/p10/ima"+str(counter)+str(predicted)+".png", image)
            counter = counter + 1

    def predictions2(self, args):
        imagein = cv2.imread(args.image)
        pil_image = Image.fromarray(imagein)
        image = ToTensor()(pil_image)
        img = image.unsqueeze(0)
        self.model.eval()
        pred = self.model(img)
        _, predicted = torch.max(pred.data, 1)
        print(predicted)

    def predictions3(self, args):
        list_of_path = []
        list_of_image = []
        for root, directories, file in os.walk(args.path):
            for file in file:
                if(file.endswith(".jpg")):
                    list_of_path.append(os.path.join(root,file))
                    file = file.split('.')
                    list_of_image.append(file[0])

        for l in range(len(list_of_image)):            
            imagein = cv2.imread(list_of_path[l])
            imagelist = []

            xmin = []
            ymin = []
            width = []
            height = []
            label = []

            tensor = torch.load(args.path'/labeled' + list_of_image[l] +'.pt')
            bbox = tensor[0]['boxes']
            pil_image = Image.fromarray(imagein)
            image = ToTensor()(pil_image)
            #imagelist = []
            imageor = image.unsqueeze(0)
            
            counter = 0
            for box in bbox:
                height = int(box[3])-int(box[1])
                width = int(box[2])-int(box[0])
                if height < 7:
                    height = 7
                if width < 7:
                    width = 7
           

                image = torchvision.transforms.functional.crop(imageor, int(box[1]), int(box[0]), height, width)
                self.model.eval()
                pred = self.model(image)
                pred = pred.detach().numpy()
                tensor[0]['embedding'][counter]=pred
                image = torchvision.transforms.functional.to_pil_image(image[0])
                image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
                image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
                cv2.imwrite(args.path + "/" + list_of_image[l] +"/image"+ str(counter)+'_' +'label' +str(tensor[0]['labels'][counter])+".png", image)
                counter = counter + 1
            torch.save(tensor, args.path + "/" + list_of_image[l] + '/tensor' + list_of_image[l] + '.pt')




def main(args):
    pred = Predictions(args)
    pred.predictions3(args)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--checkpoint', type=str, default='')
	parser.add_argument('--image', type=str, default='')
	parser.add_argument('--loss', type=str, default= 'CrossEntropy',
                            help='The name of the loss function to use.',
                            choices=['CrossEntropy', 'ArcFace',
                                     'TripletMargin', 'ContrastiveLoss',
                                     'CircleLoss', 'LargeMarginSoftmaxLoss'])
    parser.add_argument('--path', type=str, default='',help='The path to the test data')
    parser.add_argument('--save', type=str, default='', help='The path where to save the results')
	args = parser.parse_args()
	main(args)
