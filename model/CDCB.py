import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from base_model.DeformConv2d import DeformConv2d
from base_model.eca import eca_layer

class CDC(nn.Module):
    def __init__(self, dim, scales=1):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for i in range(self.nums):
            convs.append(DeformConv2d(width, width, kernel_size=3))
        self.convs = nn.ModuleList(convs)
    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        return x

class CDCB(nn.Module):
    def __init__(self, in_ch, out_ch, scales):
        super(CDCB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.CDC = CDC(dim=out_ch, scales=scales)  # CDC
        self.CA = eca_layer(out_ch)
    def forward(self, input):
        x = self.conv(input)
        x_ = x + self.CDC(x)
        out = x + self.conv1(x_)
        return self.CA(out)