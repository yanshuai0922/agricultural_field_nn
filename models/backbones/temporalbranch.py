from collections import OrderedDict
import torch.nn as nn
import torch

class SpatialTemporalExcitationBlock(nn.Module):
    def __init__(self):
        super(SpatialTemporalExcitationBlock, self).__init__()
        self.spatialtemporal_excitation = torch.nn.Sequential(OrderedDict([
            ("conv3d", nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False))
        ]))
        self.sigmoid = torch.nn.Sequential(OrderedDict([("sigomid", nn.Sigmoid())]))
    def forward(self, x):
        B, C, T, H, W = x.size()
        x_p1 = x.transpose(2,1).contiguous().view(B*T, C, H, W)
        x_p2 = x.mean(dim=1, keepdim=True)
        x_p2 = self.spatialtemporal_excitation(x_p2)
        x_p2 = x_p2.transpose(2,1).contiguous().view(B*T, 1, H, W)
        x_p2 = self.sigmoid(x_p2)

        x_p3, _ = x.max(dim=1, keepdim=True)
        x_p3 = self.spatialtemporal_excitation(x_p3)
        x_p3 = x_p3.transpose(2,1).contiguous().view(B*T, 1, H, W)
        x_p3 = self.sigmoid(x_p3)
        out = x_p1 * (x_p2 + x_p3)
        out = out.contiguous().view(B, T, C, H, W).transpose(2,1)
        return out

class ChannelTemporalExcitationBlock(nn.Module):
    def __init__(self, in_channel):
        super(ChannelTemporalExcitationBlock, self).__init__()
        self.in_channel = in_channel
        self.reduced_channel = self.in_channel // 2
        self.spatial_avgsqueeze = torch.nn.Sequential(OrderedDict(
            [("SpatialPool", nn.AdaptiveAvgPool2d(1)),
             ("SpatialSqueeze", nn.Conv2d(self.in_channel, self.reduced_channel, kernel_size=(1, 1), bias=False, stride=(1, 1))),
             ("relu", nn.ReLU(inplace=True))]))
        self.channel_process = torch.nn.Sequential(OrderedDict(
            [("conv1d", nn.Conv1d(self.reduced_channel, self.reduced_channel, kernel_size=3, stride=1,bias=False, padding=1,groups=1)),
             ("relu", nn.ReLU(inplace=True))]))
        self.spatial_excitation = torch.nn.Sequential(OrderedDict(
            [("SpatialExcitation", nn.Conv2d(self.reduced_channel, self.in_channel, kernel_size=(1, 1), stride=(1, 1))),
             ("sigmoid", nn.Sigmoid())]))
    def forward(self, x):
        B, C, T, H, W = x.size()
        x_p1 = x.transpose(2, 1).contiguous().view(B*T, C, H, W)
        x_p2 = self.spatial_avgsqueeze(x_p1)
        x_p2 = x_p2.contiguous().view(B, T, C//2, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1)
        x_p2 = self.channel_process(x_p2)
        x_p2 = x_p2.transpose(2,1).contiguous().view(-1, C//2, 1, 1)
        x_p2 = self.spatial_excitation(x_p2)
        out = x_p1 * x_p2
        out = out.contiguous().view(B, T, C, H, W).transpose(2,1)
        return out

class AttentionResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionResidualBlock, self).__init__()
        self.STEBlock = SpatialTemporalExcitationBlock()
        self.CTEBlock = ChannelTemporalExcitationBlock(in_channels)

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=in_channels)
        self.conv2 = ConvBlock(in_channels=in_channels, out_channels=in_channels, k=1,s=1,p=0)
    def forward(self, x):
        x1 = self.STEBlock(x)
        x2 = self.CTEBlock(x)
        out = self.conv1.smart_forward(x1+x2)
        out = self.conv2.smart_forward(out)
        return out

class TemporalSemanticStream(nn.Module):
    def __init__(self, in_channel, n_classes, pad_value=None):
        super(TemporalSemanticStream, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.pad_value = pad_value

        self.conv1 = ConvBlock(in_channels=in_channel, out_channels=32)
        self.ARBlock1 = AttentionResidualBlock(32)

        in_channels = 32
        channels = [64, 128, 256, 128, 64, 64]

        for mod_id in range(len(channels)):
            self.add_module("conv%d" % (mod_id + 2), ConvBlock(in_channels=in_channels, out_channels=channels[mod_id]))
            in_channels = channels[mod_id]

            self.add_module("ARBlock%d" % (mod_id + 2), AttentionResidualBlock(in_channels))

            if mod_id < 3:
                self.add_module("DownBlock%d" %(mod_id + 1), PoolBlock(in_channels))
            elif mod_id >= 3:
                self.add_module("UpBlock%d" %(mod_id - 2), TransConvBlock(in_channels))

        self.final = nn.Conv3d(64, n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = x.permute(0, 2, 1, 3, 4) 

        if self.pad_value is not None:
            pad_mask = (x0 == self.pad_value).all(dim=-1).all(dim=-1).all(dim=1)  # BxT pad mask

        x1 = self.conv1.smart_forward(x0)
        x1 = self.ARBlock1(x1)

        x2 = self.conv2.smart_forward(x1)
        x2 = self.ARBlock2(x2)
        x2  = self.DownBlock1.smart_forward(x2)

        x3 = self.conv3.smart_forward(x2)
        x3 = self.ARBlock3(x3)
        x3 = self.DownBlock2.smart_forward(x3)

        x4 = self.conv4.smart_forward(x3)
        x4 = self.ARBlock4(x4)

        x5 = self.conv5.smart_forward(x4)
        x5 = self.ARBlock5(x5)

        x6 = self.conv6.smart_forward(x5)
        x6 = self.ARBlock6(x6)

        x7 = self.conv7.smart_forward(x6)
        x7 = self.ARBlock7(x7)
        x7 = self.UpBlock3.smart_forward(x7)
        x7 = self.UpBlock3.smart_forward(x7)
        final = self.final(x7)

        final = final.permute(0, 1, 3, 4, 2)
        if self.pad_value is not None:
            if pad_mask.any():
                # masked mean
                pad_mask = pad_mask[:, :final.shape[-1]]
                pad_mask = ~pad_mask
                out = (final.permute(1, 2, 3, 0, 4) * pad_mask[None, None, None, :, :]).sum(dim=-1) / pad_mask.sum(
                    dim=-1)[None, None, None, :]
                out = out.permute(3, 0, 1, 2)
            else:
                out = final.mean(dim=-1)
        else:
            out = final.mean(dim=-1)
        return out