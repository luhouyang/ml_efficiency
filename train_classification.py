import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import argparse

import numpy as np

from data.datasets.emnist_enhanced import get_emnist
# from cnn_model import VanilaCNNModel

import torchinfo
import time

from vit_model import ViT


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='specify gpu device',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='number of training epochs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch size',
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='root directory of dataset',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='vanilla_cnn',
        help='model name [default: vanilla_cnn]',
    )
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='learning rate in training',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        help='optimizer for training',
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='experiment root',
    )
    parser.add_argument(
        '--decay_rate',
        type=float,
        default=1e-4,
        help='decay rate',
    )

    return parser.parse_args()


def inplace_relu(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('ReLU') != 1:
        m.inplace = True


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    dataloaders = get_emnist(root=args.dataset_root)
    traindl = dataloaders['train']
    testdl = dataloaders['test']

    print("DATA LOADED")

    # model = VanilaCNNModel(channels=1, num_classes=47).cuda()
    model = ViT(
        image_size=28,
        patch_size=7,
        num_classes=47,
        channels=1,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1,
    ).cuda()
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.decay_rate,
    )

    torchinfo.summary(model)

    torch.compile(model)

    # use_amp = True
    # scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("MODEL LOADED")

    torch.backends.cudnn.benchmark = True

    start_time = time.time()

    for i in range(args.epochs):
        mean_correct = []
        mean_loss = []

        model = model.train()

        print("TRAIN")

        for batch_id, (data, labels) in tqdm(enumerate(traindl, 0),
                                             total=len(traindl),
                                             smoothing=0.9):

            data = data.cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)

            # with torch.autocast(device_type=DEVICE,
            #                     dtype=torch.float16,
            #                     enabled=use_amp):
            pred = model(data)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)

            choice = pred.data.max(1)[1]
            correct = choice.eq(labels.long().data).cpu().sum()
            mean_correct.append(correct.item() / args.batch_size)
            mean_loss.append(loss.item())

        train_instance_acc = np.mean(mean_correct)
        print('Train Accuracy: %f' % train_instance_acc)

        train_instance_loss = np.mean(mean_loss)
        print('Train Loss: %f' % train_instance_loss)

    end_time = time.time()
    print('TIME:', (end_time - start_time))


if __name__ == '__main__':
    args = arguments()
    main(args)
