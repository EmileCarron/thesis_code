import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import sys
from argparse import ArgumentParser
from recognition import RecognitionModel
import json
from torchvision.datasets.utils import download_url#)download_and_extract_archive,
from aliproducts import AliProducts
from PIL import Image
from copy import deepcopy
import wandb
from pytorch_lightning.loggers import WandbLogger
#from test_tube import Experiment
#from pytorch_lightning.callbacks import ModelCheckpoint



BASE_URL = ("https://tianchi-public-us-east-download.oss-us-east-1."
            "aliyuncs.com/231780/")

SAMPLE_DATA_URL = {
    'data': BASE_URL + "AliProducts_train_sample.tar.gz",
    'json': BASE_URL + "AliProducts_train_sample.json"
}


class Prod10kDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = args.data_dir + '/Products10k'
        self.batch_size = args.batch_size
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = Prod10k(csv_file = self.data_dir + '/train.csv', root_dir = self.data_dir +'/train')
            self.val_set = Prod10k(csv_file = self.data_dir + '/val.csv', root_dir = self.data_dir + '/train')
            train_set, val_set = torch.utils.data.random_split(train_set, [1000, 7219])
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = self.batch_size, num_workers = args.num_workers)
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size = self.batch_size, num_workers = args.num_workers)
        
class AliproductsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.root = data_dir + '/Aliproducts'
        imgs_json = self.root + '/AliProducts_train_sample.json'
        fObj = open(imgs_json,)
        self.img_labels = [
            (img['id'], img['category_id'])
            for img in json.load(fObj)['images']
        ]
        self.img_dir = self.root + '/train'
        self.num_classes = 195
    
    def get_transform(self, normalize=True, to_tensor=True):
        """Return the image transformation for the given dataset type.

        Args:
            data_type (str): The type of dataset. Should be one of 'train',
            'val' or 'test'.
            normalize (bool): If True, include ImageNet-style normalization.
            to_tensor (bool): If True, transform the PIL Image to a tensor.
        """
       
        tfms = [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()]
        
        tfms.append(transforms.ToTensor())
        tfms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]))

        return transforms.Compose(tfms)

    
    #def prepare_data(self):
        #download_and_extract_archive(SAMPLE_DATA_URL['data'], self.root)
        #download_url(SAMPLE_DATA_URL['json'], self.root)
    
    def setup(self, stage=None):
        train_tfm = self.get_transform()
        if stage == 'fit' or stage is None:
            self.train_set = AliProducts(root = self.root, img_labels = self.img_labels, transform = train_tfm)
            self.train_set, self.val_set = torch.utils.data.random_split(self.train_set, [4000,937])
            
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = self.batch_size, num_workers = self.num_workers)
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size = self.batch_size, num_workers = self.num_workers)

def main(args):
    wandb_logger = WandbLogger()
    wandb.init(project = 'masterproef', entity = 'mille')

    # exp = Experiment(
    #     name='test_tube_exp',
    #     debug=True,
    #     save_dir='/checkpoint/',
    #     version=0,
    #     autosave=False,
    #     description='test demo'
    # )

    # set the hparams for the experiment
    # exp.argparse(args)
    # exp.save()
    
    

    
    dm = AliproductsDataModule(data_dir = args.data_dir ,batch_size = args.batch_size, num_workers = args.num_workers)
    #model = wandb.restore('masterproef/2zf8e0nu/checkpoints/epoch=999-step=62999.ckpt',run_path='mille/masterproef/2zf8e0nu')

    model = RecognitionModel(args)

    # checkpoint = ModelCheckpoint(
    #     filepath='/checkpoint/weights.ckpt',
    #     save_function=None,
    #     save_best_only=True,
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min'
    # )
   
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=args.max_epochs, logger=wandb_logger)
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='../../../dataset')
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--loss', type=str, default= 'CrossEntropy',
                            help='The name of the loss function to use.',
                            choices=['CrossEntropy', 'ArcFace',
                                     'TripletMargin', 'ContrastiveLoss',
                                     'CircleLoss', 'LargeMarginSoftmaxLoss'])
    parser.add_argument('--embedding_size', type=int, default=512)
    #parser.add_argument('--checkpoint', type=str, default='')
    
    args = parser.parse_args()
    main(args)
