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
from torch.nn import CosineEmbeddingLoss
import wandb
from collections import OrderedDict
from argparse import ArgumentParser
import cv2
from torchvision.transforms import ToTensor
from PIL import Image

# Test code for stock RetinaNet

class RetinaNetLightning(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.model = models.detection.retinanet_resnet50_fpn(pretrained=True)
		self.args = args
		self.save_hyperparameters()


	def getModel(self):
		return self.model
        

class Predictions(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.predclass = self.predictionclass(args)
		self.model = self.predclass.getModel()

	def predictionclass(self, args):
		pc = RetinaNetLightning(args)
		pc_model = pc.load_from_checkpoint(checkpoint_path=args.checkpoint, args=args)
		return pc_model

	def predicitions(self, args):
		imagein = cv2.imread(args.path)
		self.model.eval()
		imagelist = []
		
		pil_image = Image.fromarray(imagein)
		image = ToTensor()(pil_image)
		imagelist.append(image)
		predictions = self.model(imagelist)
		#print(predictions)
		boxes = predictions[0]['boxes']
		torch.save(predictions, args.path + '/tensor_boxes.pt')
		#print(predictions)
		color = (0,0,255)
		#import pdb; pdb.set_trace()
		for idx in range(len(predictions[0]['boxes'])):
			if predictions[0]['scores'][idx] > 0.3:
				cv2.rectangle(imagein,(predictions[0]['boxes'][idx][0], predictions[0]['boxes'][idx][1]), (predictions[0]['boxes'][idx][2], predictions[0]['boxes'][idx][3]), color, 5)
		cv2.imwrite(args.path+ "/output_image.jpg", imagein)




def main(args):
	pred = Predictions(args)
	pred.predicitions(args)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--checkpoint', type=str, default='')
	parser.add_argument('--path', type=str, default='',help='The path to the test data')
    parser.add_argument('--save', type=str, default='', help='The path where to save the results')
	args = parser.parse_args()
	main(args)



