import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import prod10k
from prod10k import Prod10k
from retinanet import RetinaNetLightning
import sys

def main(arguments):

    train_set = Prod10k(csv_file = '../../../Dataset/Products10k/train.csv',root_dir = '../../../Dataset/Products10k/train')
    #val_set = Prod10k(csv_file = '../../../DatasetProducts10k/val.csv',root_dir = '../../../Dataset/Products10k/train')
    print(len(train_set))
    train_set, val_set = torch.utils.data.random_split(train_set, [1000, 7219])
    
    train= DataLoader(train_set, batch_size=1, num_workers=int(arguments[1]))
    val = DataLoader(val_set, batch_size=1, num_workers=int(arguments[1]))
    
    model = RetinaNetLightning()
    
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=int(arguments[2]))
    trainer.fit(model, train, val)

if __name__ == '__main__':
    print("Num_workers: " + sys.argv[1])
    print("Max_epochs: " + sys.argv[2])
    main(sys.argv)
