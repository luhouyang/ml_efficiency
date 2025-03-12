import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights

from torchinfo import summary

import numpy as np
"""
Download data here: https://www.kaggle.com/datasets/crawford/emnist
"""

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
with open("data/VGG16_CLASSES.txt", 'r') as f:
    classes = f.readlines()
model = vgg16(weights=VGG16_Weights.DEFAULT)
model = model.to(DEVICE)

transforms_vgg16 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

summary(model=model, input_size=(1, 3, 224, 224))


class VanilaCNNModel(nn.Module):

    def __init__(self, input_size, num_classes):
        super(VanilaCNNModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        self.conv = nn.Sequential([
            nn.Conv2d(self.input_size[-1], 32, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(inplace=True),
        ])

        self.flatten = nn.Flatten()
        self.dense = nn.Sequential([
            nn.Linear(49 * 128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, self.num_classes),
        ])

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x
