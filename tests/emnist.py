from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

def get_emnist():
    dataloader = DataLoader(
        EMNIST(),
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    return dataloader

class EMNIST(Dataset):

    def __init__(self, root, selection='emnist-balanced', split='train'):
        super(EMNIST, self).__init__()

        allowed_selections = [
            'emnist-balanced',
            'emnist-byclass',
            'emnist-bymerge',
            'emnist-digits',
            'emnist-letters',
        ]

        split_selection = [
            'train',
            'test',
        ]

        if selection not in allowed_selections:
            raise (ValueError(
                f'Selection {selection} is not valid. Please select from {" | ".join(allowed_selections)}'
            ))

        if split not in split_selection:
            raise (ValueError(
                f'Split {split} is not valid. Please select from {" | ".join(split_selection)}'
            ))

        if not Path(root).exists():
            raise (ValueError(f'Directory {root} does not exist.'))

        if not Path(f'{root}/{selection}-{split}.csv').exists():
            raise (ValueError(
                f'Files {selection}-{split}.csv not found in {root}'))

        if not Path(f'{root}/{selection}-mapping.txt').exists():
            raise (ValueError(
                f'Files {selection}-mapping.txt not found in {root}'))

        dataset = pd.read_csv(f'{root}/{selection}-{split}.csv', sep=',')

        self.mapping = pd.read_csv(f'{root}/{selection}-mapping.txt', sep=' ')

        self.labels = np.array(dataset.iloc[:, 0])
        self.data = np.array(dataset.iloc[:, 0:])

        del dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pass

# if __name__ == '__main__':
#     ds = EMNIST(root=r'C:\Users\User\Desktop\Python\ml_efficiency\archive')
#     print(ds.__len__())
