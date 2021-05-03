import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import sku
from sku import Sku
from retinanet import RetinaNetLightning
import sys
import wandb
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import albumentations as A


class RetinaNetDataModule(pl.LightningDataModule):
    
    def __init__(self):
        super().__init__()
        self.data_dir = args.data_dir + '/SKU110K'
        self.batch_size = args.batch_size
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = Sku(csv_file = self.data_dir + '/annotations/annotations_train_embedding.csv', root_dir = self.data_dir +'/images', transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.ShiftScaleRotate(p=0.5),
                            A.RandomBrightnessContrast(p=0.2),
                            A.RGBShift(p=0.2),
                            A.RandomSizedBBoxSafeCrop(width=1333, height=800),
                            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])))

def main(args):

    wandb_logger = WandbLogger()
    wandb.init(project='thesis', entity='mille')

    model = RetinaNetLightning(args)
    dm = RetinaNetDataModule()
    
    model.embedding(self.train_set

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='../../../dataset')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--loss', type=str, default= 'CrossEntropy',
                            help='The name of the loss function to use.',
                            choices=['CrossEntropy', 'ArcFace',
                                     'TripletMargin', 'ContrastiveLoss',
                                     'CircleLoss', 'LargeMarginSoftmaxLoss'])
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
