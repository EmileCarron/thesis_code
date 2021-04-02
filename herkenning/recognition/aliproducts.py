from torch.utils.data import Dataset
from torchvision.datasets.utils import (download_and_extract_archive,
                                        download_url)
from pathlib import Path
import json
from PIL import Image
from copy import deepcopy
import random
from torchvision.transforms import ToTensor

import numpy as np

BASE_URL = ("https://tianchi-public-us-east-download.oss-us-east-1."
            "aliyuncs.com/231780/")

FULL_DATA_URLS = {
    'data': [
        BASE_URL + "AliProducts/000141585539496/train_val.part0.tar.gz",
        BASE_URL + "AliProducts/100001585554035/train_val.part1.tar.gz",
        BASE_URL + "AliProducts/200001585540031/train_val.part2.tar.gz",
        BASE_URL + "AliProducts/300001585559032/train_val.part3.tar.gz",
        BASE_URL + "AliProducts/400001585578035/train_val.part4.tar.gz",
        BASE_URL + "AliProducts/500001585599038/train_val.part5.tar.gz",
        BASE_URL + "AliProducts/600001585536030/train_val.part6.tar.gz",
        BASE_URL + "AliProducts/700001585524033/train_val.part7.tar.gz",
        BASE_URL + "AliProducts/800001585502035/train_val.part8.tar.gz",
        BASE_URL + "AliProducts/900001585552031/train_val.part9.tar.gz"
    ],
    'json': [
        BASE_URL + "AliProducts/train.json",
        BASE_URL + "AliProducts/val.json",
        BASE_URL + "AliProducts/product_tree.json"
    ]
}

SAMPLE_DATA_URL = {
    'data': BASE_URL + "AliProducts_train_sample.tar.gz",
    'json': BASE_URL + "AliProducts_train_sample.json"
}


PAL_IMGS_WITH_BYTE_TRANSPAR = [
    # Calling `.convert("RGB")` on these images, cause the warning
    # "Palette images with Transparency expressed in bytes should be converted
    # to RGBA images"
    '2836148.png',
    '2127303.png'
]


class AliProducts(Dataset):
    def __init__(self, root, img_labels, data_type='train', transform=None, sample=False,
                 download=False):
        """
        Initialize the AliProducts dataset located at the given root directory.

        Args:
            root (str): The root directory of the dataset
                data_type (str): Which version of the AliProducts dataset to
                use. Should be one of 'train', 'val', 'test' or 'sample'
            transform (callable): A callable transforming a PIL image into the
                format that is compatible with the input of the neural network
            download (bool): If True, download the dataset if it doesn't yet
                exist in the given root directory.
        """
        self.root = Path(root)
        self.data_type = data_type
        self.transform = transform
        self.img_labels = img_labels
        self.img_dir = self.root / 'train'
        

    @property
    def img_labels(self):
        return self._img_labels

    @img_labels.setter
    def img_labels(self, value):
        self._img_labels = value
        self.label_idxs = {lab: idx
                           for idx, lab in enumerate(
                               sorted({lab for _, lab in self._img_labels}))}

    def _check_exists(self):
        """Return True if the AliProducts dataset exists already.
        """
        if self.data_type == 'sample':
            return (
                (self.root / 'AliProducts_train_sample.json').exists()
                and (self.root / 'train').is_dir()
            )
        elif self.data_type in ['train', 'val']:
            return (
                (self.root / 'train.json').exists()
                and (self.root / 'val.json').exists()
                and (self.root / 'product_tree.json').exists()
                and (self.root / 'train').is_dir()
                and (self.root / 'val').is_dir()
            )
        elif self.data_type == 'test':
            raise NotImplementedError('The AliProducts test set is not '
                                      'available yet...')


    def __getitem__(self, index):
        img, label = self.img_labels[index]
        #path = self.img_dir / label / img
        im = Image.open(self.img_dir / label / img).convert('RGB')
    
        if self.transform is not None:
            #im = self.transform(im)
            im = ToTensor()(im)
            #print(im)
        return (im, self.label_idxs[label])

    def __len__(self):
        return len(self.img_labels)

    def random_split(self, fraction=0.7):
        """
        Randomly split this dataset into two copies and return the splits. The
        current dataset object will not be modified.

        Args:
            fraction (float): the relative fraction of images to use in the
            first part of the split.
        """
        all_idxs = [i for i in range(len(self))]
        k = int(fraction * len(self.img_labels))
        split_1_idxs = random.sample(all_idxs, k)
        split_2_idxs = [i for i in all_idxs if i not in split_1_idxs]

        split_1 = deepcopy(self)
        split_2 = deepcopy(self)

        split_1.img_labels = [self.img_labels[i] for i in split_1_idxs]
        split_2.img_labels = [self.img_labels[i] for i in split_2_idxs]

        return split_1, split_2
