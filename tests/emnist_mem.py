from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import polars as pl
import pickle

from memory_profiler import profile

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

    @profile
    def __init__(self,
                 root,
                 selection='emnist-balanced',
                 split='train',
                 use_file='csv'):
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
            raise ValueError(
                f'Selection {selection} is not valid. Please select from {" | ".join(allowed_selections)}'
            )

        if split not in split_selection:
            raise ValueError(
                f'Split {split} is not valid. Please select from {" | ".join(split_selection)}'
            )

        if not Path(root).exists():
            raise ValueError(f'Directory {root} does not exist.')

        if not Path(f'{root}/{selection}-mapping.txt').exists():
            raise ValueError(
                f'Files {selection}-mapping.txt not found in {root}')

        self.mapping = pd.read_csv(f'{root}/{selection}-mapping.txt', sep=' ')

        if use_file == 'csv':
            if not Path(f'{root}/{selection}-{split}.csv').exists():
                raise ValueError(
                    f'Files {selection}-{split}.csv not found in {root}')

            dataset = pd.read_csv(f'{root}/{selection}-{split}.csv', sep=',')

            self.labels = np.array(dataset.iloc[:, 0])
            self.data = np.array(dataset.iloc[:, 1:])

        elif use_file == 'pkl':
            pickle_path = Path(root) / f"{selection}-{split}.pkl"
            if not pickle_path.exists():
                raise FileNotFoundError(
                    f"Pickle file {pickle_path} not found. Please run the preprocessing script."
                )

            with open(pickle_path, 'rb') as f:
                self.data, self.labels, self.mapping = pickle.load(f)

        elif use_file == 'parquet':
            parquet_path = Path(root) / f"{selection}-{split}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(
                    f"Parquet file {parquet_path} not found. Please run the preprocessing script."
                )

            dataset = pl.read_parquet(parquet_path)
            self.labels = dataset[:, 0].to_numpy()
            self.data = dataset[:, 1:].to_numpy()

    @profile
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

@profile
def get_1000_data(ds):
    for i in range(1000):
        _, _ = ds.__getitem__(i)

if __name__ == '__main__':
    ds = EMNIST(root=r'C:\Users\User\Desktop\Python\ml_efficiency\archive',
                use_file='parquet')
    print(ds.__len__())
    get_1000_data(ds)
