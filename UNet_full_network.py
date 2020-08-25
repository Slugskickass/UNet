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
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
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


# Use DoubleConv and Maxpool classes to construct 'DownAlong' class
class DownAlong(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            Maxpool(),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


#Expansive path
class UpAlong(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, a1, a2):
        a1 = self.up(a1)

        diffY = a2.size()[2] - a1.size()[2]
        diffX = a2.size()[3] - a1.size()[3]
# This causes a problem
        a2 = F.pad(a2, [-diffX // 2, diffX // 2 - diffX,
                        -diffY // 2, diffY // 2 - diffY])

        x = torch.cat([a2, a1], dim=1)
        return self.conv(x)


# Final output conv
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Create model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownAlong(64, 128)
        self.down2 = DownAlong(128, 256)
        self.down3 = DownAlong(256, 512)
        self.down4 = DownAlong(512, 1024)
        self.up1 = UpAlong(1024, 512)
        self.up2 = UpAlong(512, 256)
        self.up3 = UpAlong(256, 128)
        self.up4 = UpAlong(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        output = self.outc(x9)
        return output


# Test model shapes
t = torch.ones([1, 1, 572, 572])
network = UNet(n_channels=1, n_classes=2)
pred = network(t)
print(pred.shape)
