from pathlib import Path
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import polars as pl  # https://docs.pola.rs/api/python/stable/reference/index.html

import line_profiler

profile = line_profiler.LineProfiler()


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
                 use_loader='pd',
                 use_np=True,
                 use_file='csv'):
        super(EMNIST, self).__init__()

        self.use_loader = use_loader
        self.use_np = use_np

        allowed_selections = [
            'emnist-balanced',
            'emnist-byclass',
            'emnist-bymerge',
            'emnist-digits',
            'emnist-letters',
        ]

        split_selection = ['train', 'test']

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

        mapping_path = Path(f'{root}/{selection}-mapping.txt')
        if not mapping_path.exists():
            raise ValueError(f'File {mapping_path} not found.')

        if use_loader == 'pd':
            self.mapping = pd.read_csv(mapping_path, sep=' ')
        elif use_loader == 'pl':
            self.mapping = pl.read_csv(mapping_path, separator=' ')

        # If using Pickle (.pkl) format
        if use_file == 'pkl':
            pkl_path = Path(f'{root}/{selection}-{split}.pkl')
            if not pkl_path.exists():
                raise ValueError(f'File {pkl_path} not found.')

            with open(pkl_path, 'rb') as f:
                dataset = pickle.load(f)

            self.data, self.labels, *_ = dataset

        # If using CSV format
        elif use_file == 'csv':
            csv_path = Path(f'{root}/{selection}-{split}.csv')
            if not csv_path.exists():
                raise ValueError(f'File {csv_path} not found.')

            dataset = pd.read_csv(
                csv_path, sep=',') if use_loader == 'pd' else pl.read_csv(
                    csv_path, separator=',')
            if use_np:
                self.labels = dataset.iloc[:, 0].to_numpy(
                ) if use_loader == 'pd' else dataset[:, 0].to_numpy()
                self.data = dataset.iloc[:, 1:].to_numpy(
                ) if use_loader == 'pd' else dataset[:, 1:].to_numpy()
            else:
                self.labels = dataset.iloc[:,
                                           0] if use_loader == 'pd' else dataset[:,
                                                                                 0]
                self.data = dataset.iloc[:,
                                         1:] if use_loader == 'pd' else dataset[:,
                                                                                1:]

        # If using Parquet format
        elif use_file == 'parquet':
            parquet_path = Path(f'{root}/{selection}-{split}.parquet')
            if not parquet_path.exists():
                raise ValueError(f'File {parquet_path} not found.')

            dataset = pd.read_parquet(
                parquet_path) if use_loader == 'pd' else pl.read_parquet(
                    parquet_path)
            if use_np:
                self.labels = dataset.iloc[:, 0].to_numpy(
                ) if use_loader == 'pd' else dataset[:, 0].to_numpy()
                self.data = dataset.iloc[:, 1:].to_numpy(
                ) if use_loader == 'pd' else dataset[:, 1:].to_numpy()
            else:
                self.labels = dataset.iloc[:,
                                           0] if use_loader == 'pd' else dataset[:,
                                                                                 0]
                self.data = dataset.iloc[:,
                                         1:] if use_loader == 'pd' else dataset[:,
                                                                                1:]

    @profile
    def __len__(self):
        return len(self.data)

    @profile
    def __getitem__(self, index):
        if self.use_np:
            return self.data[index], self.labels[index]
        else:
            return (self.data.iloc[index],
                    self.labels.iloc[index]) if self.use_loader == 'pd' else (
                        self.data.row(index), self.labels[index])


@profile
def get_1000_data(ds):
    for i in range(1000):
        _, _ = ds.__getitem__(i)


if __name__ == '__main__':
    ds = EMNIST(
        root=r'C:\Users\User\Desktop\Python\ml_efficiency\archive',
        use_loader='pl',
        use_np=True,
        use_file='parquet',
    )
    print(ds.__len__())

    get_1000_data(ds)

    profile.print_stats()
