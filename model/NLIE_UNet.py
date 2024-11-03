import torch.nn.functional as F
import torch.nn as nn
import torch
import math
__all__ = ['NLIE_UNet']
from CDCB import  CDCB
from NEB import  NEB
from HIFM import  HIFM2
from base_model.DoubleConv import DoubleConv

class NLIE_UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=256, **kwargs):
        super().__init__()
        # self.filters = [8, 16, 32, 64, 128]  # T
        # self.filters = [16, 32, 64, 128, 256]  # S
        self.filters = [32, 64, 128, 256, 512]  # B
        # self.filters = [64, 128, 256, 512, 1024]  # L

        self.sizes = [img_size, img_size // 2, img_size // 4, img_size // 8, img_size // 16, img_size // 32]

        self.conv1 = DoubleConv(3, self.filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(self.filters[0], self.filters[1])
        self.pool2 = nn.MaxPool2d(2)
        ### CDCB
        self.conv3 = CDCB(self.filters[1], self.filters[2], 2)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = CDCB(self.filters[2], self.filters[3], 4)
        self.pool4 = nn.MaxPool2d(2)
        self.conv51 = CDCB(self.filters[3], self.filters[4], 8)

        self.up6 = nn.ConvTranspose2d(self.filters[4], self.filters[3], 2, stride=2)
        self.conv6 = CDCB(self.filters[3], self.filters[3], 4)
        self.up7 = nn.ConvTranspose2d(self.filters[3], self.filters[2], 2, stride=2)
        self.conv7 = CDCB(self.filters[2], self.filters[2], 2)
        self.up8 = nn.ConvTranspose2d(self.filters[2], self.filters[1], 2, stride=2)
        self.conv8 = DoubleConv(self.filters[1], self.filters[1])
        self.up9 = nn.ConvTranspose2d(self.filters[1], self.filters[0], 2, stride=2)
        self.conv9 = DoubleConv(self.filters[0], self.filters[0])

        ### NEB
        self.Skip1 = NEB(self.filters[0], self.filters[1], self.filters[2], self.filters[3], self.filters[4])
        ### HIFM
        self.hifm6 = HIFM2(self.filters[3], self.filters[3])
        self.hifm7 = HIFM2(self.filters[2], self.filters[2])
        self.hifm8 = HIFM2(self.filters[1], self.filters[1])
        self.hifm9 = HIFM2(self.filters[0], self.filters[0])

        self.final = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv51(self.pool4(c4))

        c1, c2, c3, c4, c5 = self.Skip1(c1, c2, c3, c4, c5)

        up_6 = self.up6(c5)
        merge6 = self.hifm6(up_6, c4)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = self.hifm7(up_7, c3)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = self.hifm8(up_8, c2)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = self.hifm9(up_9, c1)
        c9 = self.conv9(merge9)
        out = c9
        return self.final(out)

if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model = NLIE_UNet(1)

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('params:', params)