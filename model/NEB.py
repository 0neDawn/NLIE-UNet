import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from HIFM import HIFM2, HIFM3

class Down_Cov(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(Down_Cov, self).__init__()
        self.Down = nn.Conv2d(in_ch, ou_ch, 3, stride=2, padding=1, groups=1)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ou_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ou_ch, ou_ch, 11, padding=5, groups=ou_ch),
        )
    def forward(self, x):
        x_ = self.Down(x)
        out = self.conv(x_) + x_
        return out
class Up_Cov(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(Up_Cov, self).__init__()
        self.Up = nn.ConvTranspose2d(in_ch, ou_ch, 2, stride=2, groups=1)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ou_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ou_ch, ou_ch, 11, padding=5, groups=ou_ch),
        )
    def forward(self, x):
        x_ = self.Up(x)
        out = self.conv(x_) + x_
        return out

class NEB(torch.nn.Module):
    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, in_filters5):
        super().__init__()
        self.Down11 = Down_Cov(in_filters1, in_filters2)
        self.Down12 = Down_Cov(in_filters2, in_filters3)
        self.Down2 = Down_Cov(in_filters2, in_filters3)
        self.Up4 = Up_Cov(in_filters4, in_filters3)
        self.Up51 = Up_Cov(in_filters5, in_filters4)
        self.Up52 = Up_Cov(in_filters4, in_filters3)

        self.Up11 = Up_Cov(in_filters3, in_filters2)
        self.Up12 = Up_Cov(in_filters2, in_filters1)
        self.Up2 = Up_Cov(in_filters3, in_filters2)
        self.Down4 = Down_Cov(in_filters3, in_filters4)
        self.Down51 = Down_Cov(in_filters3, in_filters4)
        self.Down52 = Down_Cov(in_filters4, in_filters5)

        self.hifm1 = HIFM2(in_filters3, in_filters3)
        self.hifm2 = HIFM3(in_filters3, in_filters3, in_filters3)
        self.hifm3 = HIFM3(in_filters3, in_filters3, in_filters3)
        self.hifm4 = HIFM3(in_filters3, in_filters3, in_filters3)
        self.hifm5 = HIFM2(in_filters3, in_filters3)

    def forward(self, x1, x2, x3, x4, x5):
        x_1 = self.Down12(self.Down11(x1))
        x_2 = (self.Down2(x2))
        x_3 = (x3)
        x_4 = (self.Up4(x4))
        x_5 = self.Up52(self.Up51(x5))

        att1 = self.Up12(self.Up11(self.hifm1(x_1, x_2)))
        att2 = (self.Up2(self.hifm2(x_1, x_2, x_3)))
        att3 = self.hifm3(x_2, x_3, x_4)
        att4 = (self.Down4(self.hifm4(x_3, x_4, x_5)))
        att5 = self.Down52(self.Down51(self.hifm5(x_4, x_5)))

        x1 = (att1)
        x2 = (att2)
        x3 = (att3)
        x4 = (att4)
        x5 = (att5)
        return x1, x2, x3, x4, x5