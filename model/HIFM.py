import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class HIFM2(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)
        self.convg = nn.Conv2d(F_g, F_g, 3, padding=1, groups=F_g)
        self.convg3 = nn.Conv2d(F_g, F_g, 3, padding=1, groups=F_g)
        self.convg5 = nn.Conv2d(F_g, F_g, 5, padding=2, groups=F_g)
        self.convg7 = nn.Conv2d(F_g, F_g, 7, padding=3, groups=F_g)
        self.convg9 = nn.Conv2d(F_g, F_g, 9, padding=4, groups=F_g)
        self.convg_ = nn.Conv2d(F_g, F_g, 1, padding=0, groups=1)

        self.convx = nn.Conv2d(F_x, F_x, 3, padding=1, groups=F_x)
        self.convx3 = nn.Conv2d(F_x, F_x, 3, padding=1, groups=F_x)
        self.convx5 = nn.Conv2d(F_x, F_x, 5, padding=2, groups=F_x)
        self.convx7 = nn.Conv2d(F_x, F_x, 7, padding=3, groups=F_x)
        self.convx9 = nn.Conv2d(F_x, F_x, 9, padding=4, groups=F_x)
        self.convx_ = nn.Conv2d(F_x, F_x, 1, padding=0, groups=1)
    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        cout = self.relu(x_after_channel)

        # Spatial-wise attention
        x_ = self.convx(x)
        out_att1 = self.convx3(x_)
        out_att2 = self.convx5(x_)
        out_att3 = self.convx7(x_)
        out_att4 = self.convx9(x_)
        out_att = self.convx_((x_ + out_att1 + out_att2 + out_att3 + out_att4))

        g_ = self.convg(g)
        Sp_att_g1 = self.convg3(g_)
        Sp_att_g2 = self.convg5(g_)
        Sp_att_g3 = self.convg7(g_)
        Sp_att_g4 = self.convg9(g_)
        Sp_att_g = self.convg_((Sp_att_g1 + Sp_att_g2 + Sp_att_g3 + Sp_att_g4))

        Sp_att = (out_att + Sp_att_g) / 2.0
        sout = x + Sp_att
        out = cout + sout
        return out
class HIFM3(nn.Module):
    def __init__(self, F_g, F_d, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.mlp_d = nn.Sequential(
            Flatten(),
            nn.Linear(F_d, F_x))
        self.relu = nn.ReLU(inplace=True)
        self.convg = nn.Conv2d(F_g, F_g, 3, padding=1, groups=F_g)
        self.convg3 = nn.Conv2d(F_g, F_g, 3, padding=1, groups=F_g)
        self.convg5 = nn.Conv2d(F_g, F_g, 5, padding=2, groups=F_g)
        self.convg7 = nn.Conv2d(F_g, F_g, 7, padding=3, groups=F_g)
        self.convg9 = nn.Conv2d(F_g, F_g, 9, padding=4, groups=F_g)
        self.convg_ = nn.Conv2d(F_g, F_g, 1, padding=0, groups=1)

        self.convx = nn.Conv2d(F_x, F_x, 3, padding=1, groups=F_x)
        self.convx3 = nn.Conv2d(F_x, F_x, 3, padding=1, groups=F_x)
        self.convx5 = nn.Conv2d(F_x, F_x, 5, padding=2, groups=F_x)
        self.convx7 = nn.Conv2d(F_x, F_x, 7, padding=3, groups=F_x)
        self.convx9 = nn.Conv2d(F_x, F_x, 9, padding=4, groups=F_x)
        self.convx_ = nn.Conv2d(F_x, F_x, 1, padding=0, groups=1)

        self.convd = nn.Conv2d(F_d, F_d, 3, padding=1, groups=F_d)
        self.convd3 = nn.Conv2d(F_d, F_d, 3, padding=1, groups=F_d)
        self.convd5 = nn.Conv2d(F_d, F_d, 5, padding=2, groups=F_d)
        self.convd7 = nn.Conv2d(F_d, F_d, 7, padding=3, groups=F_d)
        self.convd9 = nn.Conv2d(F_d, F_d, 9, padding=4, groups=F_d)
        self.convd_ = nn.Conv2d(F_d, F_d, 1, padding=0, groups=1)
    def forward(self, g, x, d):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        avg_pool_d = F.avg_pool2d(d, (d.size(2), d.size(3)), stride=(d.size(2), d.size(3)))
        channel_att_d = self.mlp_d(avg_pool_d)
        channel_att_sum = (channel_att_x + channel_att_g + channel_att_d)/3.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        cout = self.relu(x_after_channel)

        # Spatial-wise attention
        x_ = self.convx(x)
        out_att1 = self.convx3(x_)
        out_att2 = self.convx5(x_)
        out_att3 = self.convx7(x_)
        out_att4 = self.convx9(x_)
        out_att = self.convx_((x_ + out_att1 + out_att2 + out_att3 + out_att4))

        g_ = self.convg(g)
        Sp_att_g1 = self.convg3(g_)
        Sp_att_g2 = self.convg5(g_)
        Sp_att_g3 = self.convg7(g_)
        Sp_att_g4 = self.convg9(g_)
        Sp_att_g = self.convg_((Sp_att_g1 + Sp_att_g2 + Sp_att_g3 + Sp_att_g4))

        d_ = self.convd(d)
        Sp_att_d1 = self.convd3(d_)
        Sp_att_d2 = self.convd5(d_)
        Sp_att_d3 = self.convd7(d_)
        Sp_att_d4 = self.convd9(d_)
        Sp_att_d = self.convd_((Sp_att_d1 + Sp_att_d2 + Sp_att_d3 + Sp_att_d4))

        Sp_att = (out_att + Sp_att_g + Sp_att_d) / 3.0
        sout = x + Sp_att

        out = cout + sout
        return out