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

def main(arguments):

    train_set = Sku(csv_file = '../../../dataset/SKU110K_fixed/annotations/annotations_train.csv',root_dir = '../../../dataset/SKU110K_fixed/images')
    val_set = Sku(csv_file = '../../../dataset/SKU110K_fixed/annotations/annotations_val.csv',root_dir = '../../../dataset/SKU110K_fixed/images')
    test_set = Sku(csv_file = '../../../dataset/SKU110K_fixed/annotations/annotations_test.csv',root_dir = '../../../dataset/SKU110K_fixed/images')
    
    train_set, train2 = torch.utils.data.random_split(train_set, [100, 8119])
    
    train= DataLoader(train_set, batch_size=1, num_workers=int(arguments[1]))
    val = DataLoader(val_set, batch_size=1, num_workers=int(arguments[1]))
    test = DataLoader(test_set, batch_size=1, num_workers=int(arguments[1]))
    
    wandb_logger = WandbLogger()
    wandb.init(project='thesis', entity='mille')

    
    model = RetinaNetLightning()
    
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=int(arguments[2]),
        logger=wandb_logger)
    trainer.fit(model, train, val)
    trainer.test(model, test)
    

if __name__ == '__main__':
    print("Num_workers: " + sys.argv[1])
    print("Max_epochs: " + sys.argv[2])
    main(sys.argv)
