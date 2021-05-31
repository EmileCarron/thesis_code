from torch.utils.data import DataLoader
from detectie.sku import Sku
from detectie.retinanet import RetinaNetLightning
import torch

retinanet = RetinaNetLightning()
sku = Sku(csv_file = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/annotations/annotations_train.csv',root_dir = '/Users/emilecarron/Documents/School/Universiteit/1ma/Masterproef/Tutorial/5/SKU110K_fixed/images')
dl = DataLoader(sku, batch_size=1, num_workers=0)
retinanet.model.train = True
batch = next(iter(dl))
retinanet.training_step(batch, 0)
