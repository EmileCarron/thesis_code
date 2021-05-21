from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image


class RP2KDataset(Dataset):
    def __init__(self, data_root: str, phase: str = 'train',
                 max_per_label=-1, transforms: callable = None):
        """
        Args:
            data_root (str): The root data directory that contains a
                subdirectory "rp2k_dataset" with the raw data.
            phase (str): Either "train", "val" or "test". Currently, "val" and
                "test" will return the same images.
            transforms (callable): Callable that takes in a PIL image and
                applies a set of transforms on it.
            max_per_label (int): The maximum number of samples per label. If
                -1, the number of samples present will be used.
        """
        if phase not in ['train', 'test', 'val']:
            raise ValueError(f'Unsupported training phase "{phase}"')

        self.data_root = Path(data_root)
        self.transforms = transforms
        phase = phase if not phase == 'val' else 'test'

        self.df = pd.DataFrame([{
            'label': cls_path.name,
            'image': im_path
        }
            for cls_path in self.data_root.glob(f'rp2k_dataset/all/{phase}/*')
            for im_path in sorted(cls_path.glob('*'))[:max_per_label]
        ])
        self.labels = sorted(self.df['label'].unique())
        self.label_to_idx = {
            lab: idx for idx, lab in enumerate(self.labels)
        }
        self.num_classes = len(self.labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        im = Image.open(row['image']).convert('RGB')
        label = row['label']
        label_idx = self.label_to_idx[label]

        if self.transforms is not None:
            im = self.transforms(im)

        return im, label_idx
