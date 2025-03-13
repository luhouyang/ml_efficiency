from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd


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
        self.data = np.array(dataset.iloc[:, 1:]).astype('float32').reshape(
            -1, 28, 28, 1).transpose([0, 2, 1, 3])

        del dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])
        return ts(self.data[index]), torch.tensor(self.labels[index],
                                                  dtype=torch.long)


def get_emnist(root):

    return {
        x:
        DataLoader(
            EMNIST(root=root, split=x),
            batch_size=32,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            prefetch_factor=2,
        )
        for x in ['train', 'test']
    }


# if __name__ == '__main__':
#     import timeit

#     def pros():
#         ds = EMNIST(root=r'C:\Users\User\Desktop\Python\ml_efficiency\archive')
#         print(ds.__len__())
#         print(ds.__getitem__(0))

#     print("TIME:", timeit.timeit("pros()", globals=globals(), number=1))
