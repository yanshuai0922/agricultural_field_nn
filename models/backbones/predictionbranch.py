import torch.nn as nn
import torch
from backbones.ltae import LTAE2d
import torch.nn.functional as F
import math


def createConvFunc(op_type):
    assert op_type in ['cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'
            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'
            shape = weights.shape
            weights = weights.contiguous().view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).contiguous().view(shape) # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation
            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.contiguous().view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.contiguous().view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        print('impossible to be here unless you force that')
        return None

class Conv2d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = createConvFunc(pdc)
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, input):
        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class MapReduce2d(nn.Module):
    def __init__(self, channels, n_classes):
        super(MapReduce2d, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.conv = nn.Conv2d(self.channels, self.n_classes, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.conv(x)

class EdgeExtractionBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EdgeExtractionBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.cd_conv = PDCBlock(pdc="cd", inplane=self.in_channel, ouplane=self.in_channel)
        self.ad_conv = PDCBlock(pdc="ad", inplane=self.in_channel, ouplane=self.in_channel)
        self.rd_conv = PDCBlock(pdc="rd", inplane=self.in_channel, ouplane=self.in_channel)
        self.relu = nn.ReLU()
        self.mapreduce = MapReduce2d(self.in_channel, self.out_channel)

    def forward(self, x):
        x1 = self.cd_conv(x)
        x2 = self.ad_conv(x)
        x3 = self.rd_conv(x)
        out = self.relu(x1+x2+x3)
        out = self.mapreduce(out)
        return out


class PredictionStream(nn.Module):
    def __init__(self, in_channel, n_classes, pad_value=None):
        super(PredictionStream, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.pad_value = pad_value

        self.conv1 = ConvBlock(in_channels=in_channel, out_channels=32)
        self.EEBlock1 = EdgeExtractionBlock(in_channel=32, out_channel=self.n_classes)
        self.TEBlock1 = LTAE2d(in_channels=32, d_model=256, n_head=4, d_k=32, mlp=[256 * 4, 32, 32])

        in_channels = 32
        channels = [64, 128, 256]
        for mod_id in range(len(channels)):
            self.add_module("conv%d" % (mod_id + 2), ConvBlock(in_channels=in_channels, out_channels=channels[mod_id]))

            in_channels = channels[mod_id]
            self.add_module("EEBlock%d" % (mod_id + 2), EdgeExtractionBlock(in_channel=in_channels, out_channel=self.n_classes))
            self.add_module("TEBlock%d" % (mod_id + 2), LTAE2d(in_channels=in_channels, d_model=256, n_head=4, d_k=32, mlp=[256 * 4, in_channels, in_channels]))
            if mod_id == 0:
                self.add_module("DownBlock%d" % (mod_id + 1), PoolBlock(in_channels))
                self.add_module("UpSample%d" % (mod_id + 1), nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1))
            else:
                self.add_module("DownBlock%d" % (mod_id + 1), DownConvBlock(in_channels, in_channels))
                self.add_module("UpSample%d" % (mod_id + 1), nn.ConvTranspose2d(in_channels, in_channels, kernel_size=6, stride=4, padding=1))

        self.classifier = nn.Conv2d(self.n_classes*3, self.n_classes, kernel_size=1)

    def forward(self, x, batch_positions=None):
        x0 = x.permute(0, 2, 1, 3, 4) 

        x1 = self.conv1.smart_forward(x0)
        s1 = self.TEBlock1(x1, batch_positions=batch_positions)
        s1 = self.EEBlock1(s1)

        x2 = self.conv2.smart_forward(x1)
        x2 = self.DownBlock1.smart_forward(x2)
        s2 = self.TEBlock2(x2, batch_positions=batch_positions)
        s2 = self.UpSample1(s2.contiguous())
        s2 = self.EEBlock2(s2)

        x3 = self.conv3.smart_forward(x2)
        x3 = self.DownBlock2.smart_forward(x3)
        s3 = self.TEBlock3(x3, batch_positions=batch_positions)
        s3 = self.UpSample2(s3.contiguous())
        s3 = self.EEBlock3(s3)

        x4 = self.conv4.smart_forward(x3)
        s4 = self.TEBlock4(x4, batch_positions=batch_positions)
        s4 = self.UpSample3(s4.contiguous())
        s4 = self.EEBlock4(s4)

        outputs = [s1, s2, s3, s4]
        outputs = self.classifier(torch.cat(outputs, dim=1))
        return outputs