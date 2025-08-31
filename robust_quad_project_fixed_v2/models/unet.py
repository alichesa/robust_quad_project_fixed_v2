import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base=32, num_keypoints=4):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1= nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2)
        self.pool2= nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool3= nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base*4, base*8)
        self.pool4= nn.MaxPool2d(2)

        self.bot  = DoubleConv(base*8, base*16)

        self.up4  = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = DoubleConv(base*16, base*8)
        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.out_heat = nn.Conv2d(base, num_keypoints, 1)
        self.out_vis  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base, num_keypoints, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)

        b  = self.bot(p4)

        d4 = self.up4(b); d4 = torch.cat([d4, e4], 1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], 1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], 1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], 1); d1 = self.dec1(d1)

        heat = self.out_heat(d1)
        vis  = self.out_vis(d1).squeeze(-1).squeeze(-1)
        return heat, vis
