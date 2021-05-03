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




class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(786432, 3), nn.ReLU(), nn.Linear(3, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 786432))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = batch['image']
        y = batch['boundingbox']
        x = x.view(x.size(1), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    
    train_set = Sku(csv_file = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/annotations/annotations_train.csv',root_dir = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/images')
    val_set = Sku(csv_file = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef//Tutorial/5/SKU110K_fixed/annotations/annotations_val.csv',root_dir = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/images')

    train_set, train_over = torch.utils.data.random_split(train_set, [1000, 1207481])
    val_set, val_over = torch.utils.data.random_split(val_set, [1000, 89967])
    #print(train_set[0]['image'])

    train= DataLoader(train_set, batch_size=12)#, num_workers=4)
    val = DataLoader(val_set, batch_size=12)#, num_workers=4)

    model = LitAutoEncoder()

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                        max_epochs=10)
    trainer.fit(model, train, val)



