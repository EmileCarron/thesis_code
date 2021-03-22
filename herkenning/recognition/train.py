import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import Prod10k
from Prod10k import Prod10k
from retinanet import RetinaNetLightning
import sys
from argparse import ArgumentParser

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


def main(arguments):
    
    model = RetinaNetLightning()
    dm = RetinaNetDataModule()
    
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=int(arguments[2]))
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='../../../dataset')
    args = parser.parse_args()
    main(args)
