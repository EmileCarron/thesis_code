import torch
from torch import nn 
from torch import optim
import pytorch_lightning as pl 
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import sku
from sku import Sku

model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()
x = [torch.rand(3, 300,400), torch.rand(3,500,400)]
predictions = models(x)

