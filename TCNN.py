import torch
import torch.nn as nn
from torchvision.models import inception_v3 as incep
from torchvision.models.inception import InceptionA as IncA
from torchvision.models.inception import InceptionB as IncB
from torchvision.models.inception import InceptionC as IncC
from torchvision.models.inception import InceptionD as IncD
from torchvision.models.inception import InceptionE as IncE
from torchvision.models.inception import BasicConv2d as Basic2D
import torch.nn.functional as F
from torch.nn import AvgPool2d as AveragePooling
from torch.nn import Softmax as Softmax, Sigmoid, BCELoss, Linear, Softmax

class TCNN(nn.Module):
    def __init__(self, output_size=8):
        super(TCNN, self).__init__()

        self.output_size = output_size

        self.conv0 = Basic2D(3, 64, kernel_size=3, padding=1, stride=1)
        self.pool0 = nn.MaxPool2d(3,stride=2, padding=1)
        self.conv1 = Basic2D(64, 128, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(3,stride=2, padding=1)
        self.conv2 = Basic2D(128, self.output_size, kernel_size=3, padding=1, stride=1)
        self.avgpool0 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.avgpool0(x)
        x = x.view(-1, 1 * 1 * self.output_size)
        return x
