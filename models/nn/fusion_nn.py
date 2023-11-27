from torch import nn
import torch.nn.functional as F
import torch

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim=8, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()
        rates = [2 * r for r in rates]
        self.features = []
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True)))
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)))
        self.features = nn.ModuleList(self.features)
        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(2, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),)

    def forward(self, x, edge):
        x_size = x.size()
        img_feature = self.img_pooling(x)
        img_features = self.img_conv(img_feature)
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear', align_corners=True)
        out = img_features

        edge_features = self.edge_conv(edge)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out