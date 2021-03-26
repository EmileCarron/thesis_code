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
from torchvision.datasets.utils import (download_and_extract_archive,
                                        download_url)
from aliproducts import AliProducts
from PIL import Image
from copy import deepcopy



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
    def __init__(self):
        super().__init__()
        self.root = args.data_dir + '/Aliproducts'
        imgs_json = self.root + '/AliProducts_train_sample.json'
        fObj = open(imgs_json,)
        self.img_labels = [
            (img['id'], img['category_id'])
            for img in json.load(fObj)['images']
        ]
        self.img_dir = self.root + '/train'

    def prepare_data(self):
        download_and_extract_archive(SAMPLE_DATA_URL['data'], self.root)
        download_url(SAMPLE_DATA_URL['json'], self.root)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = AliProducts(root = self.root, img_labels = self.img_labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = args.batch_size, num_workers = args.num_workers)

def main(arg):
    
    model = RecognitionModel()
    dm = AliproductsDataModule()
    
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=args.max_epochs)
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='../../../dataset')
    args = parser.parse_args()
    main(args)
