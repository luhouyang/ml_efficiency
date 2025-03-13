import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Download data here: https://www.kaggle.com/datasets/crawford/emnist
"""


class VanilaCNNModel(nn.Module):

    def __init__(self, channels=1, num_classes=47):
        super(VanilaCNNModel, self).__init__()

        self.channel = channels
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(self.channel, 32, 3, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class BigCNNModel(nn.Module):

    def __init__(self, channels=3, num_classes=47):
        super().__init__()

        self.channels = channels
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Conv2d(self.channels, 32, 3, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class InPlaceCNNModel(nn.Module):

    def __init__(self, channels=3, num_classes=47):
        super().__init__()

        self.channels = channels
        self.num_classes = num_classes

        self.mxp = nn.MaxPool2d(2)
        # self.softmax = nn.Softmax(dim=-1)
        # self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(self.channels,
                               32,
                               3,
                               padding='same',
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.mlp1 = nn.Linear(128 * 5 * 5, 256)
        self.dropout = nn.Dropout(0.4)
        self.mlp2 = nn.Linear(256, 128)
        self.head = nn.Linear(128, self.num_classes)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.mxp(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.mxp(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.dropout(self.mlp1(x.view(B, 3200))), inplace=True)
        x = F.relu(self.dropout(self.mlp2(x)), inplace=True)
        return F.log_softmax(self.head(x), dim=-1)


if __name__ == '__main__':
    import timeit

    # model = VanilaCNNModel(channels=1, num_classes=47).cuda()
    # model = BigCNNModel(channels=1, num_classes=47).cuda()

    # torch.backends.cudnn.benchmark = True

    model = InPlaceCNNModel(channels=1, num_classes=47).cuda()
    # torch.compile(model, mode='max-autotune')

    sample = torch.randn((2, 1, 28, 28)).cuda()

    times = []
    for i in range(10):
        times.append(
            timeit.timeit("model(sample)", globals=globals(), number=25000))

    print('TIME:', np.mean(times))

# vanilla: 13.021086010000726
# big: 13.135676969999622
# inplace:
#   compile: 12.970229969998764
#   no compile: 12.84468963999825           <- BEST ->
#
#   backend, benchmark
#       compile: 12.938241380002001
#       no compile: 13.255993210001908
