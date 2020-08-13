import torch
import torch.nn as nn
import torch.nn.functional as F


# Create double convolution class
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# Create maxpool class
class Maxpool(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.maxpool(x)


# Use DoubleConv and Maxpool classes to construct 'AlongDown' class
class AlongDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_maxpool = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            Maxpool()
        )

    def forward(self, x):
        return self.conv_maxpool(x)


# Create model
class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.down1 = AlongDown(n_channels, 64)
        self.down2 = AlongDown(64, 128)
        self.down3 = AlongDown(128, 256)
        self.down4 = AlongDown(256, 512)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x4


# Test model shapes
t = torch.ones([1, 1, 572, 572])
network = UNet(n_channels=1)
pred = network(t)
print(pred.shape)
