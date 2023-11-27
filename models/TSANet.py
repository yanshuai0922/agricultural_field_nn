from torch import nn
import torch
from backbones.temporalbranch import TemporalSemanticStream
from backbones.predictionbranch import PredictionStream
from nn.fusion_nn import _AtrousSpatialPyramidPoolingModule_v2


class TSANet(nn.Module):
    def __init__(self, in_channel, n_classes, pad_value=None):
        super(TSANet, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.pad_value = pad_value

        temproalbranch = TemporalSemanticStream(in_channel=self.in_channel, n_classes=self.n_classes)
        self.conv1 = temproalbranch.conv1
        self.conv2 = temproalbranch.conv2
        self.conv3 = temproalbranch.conv3
        self.conv4 = temproalbranch.conv4
        self.conv5 = temproalbranch.conv5
        self.conv6 = temproalbranch.conv6
        self.conv7 = temproalbranch.conv7
        self.ARBlock1 = temproalbranch.ARBlock1
        self.ARBlock2 = temproalbranch.ARBlock2
        self.ARBlock3 = temproalbranch.ARBlock3
        self.ARBlock4 = temproalbranch.ARBlock4
        self.ARBlock5 = temproalbranch.ARBlock5
        self.ARBlock6 = temproalbranch.ARBlock6
        self.ARBlock7 = temproalbranch.ARBlock7
        self.DownBlock1 = temproalbranch.DownBlock1
        self.DownBlock2 = temproalbranch.DownBlock2
        self.UpBlock3 = temproalbranch.UpBlock3
        self.final = temproalbranch.final

        predictbranch = PredictionStream(in_channel=self.in_channel, n_classes=2)
        self.TEBlock1 = predictbranch.TEBlock1
        self.TEBlock2 = predictbranch.TEBlock2
        self.TEBlock3 = predictbranch.TEBlock3
        self.TEBlock4 = predictbranch.TEBlock4
        self.UpSample1 = predictbranch.UpSample1
        self.UpSample2 = predictbranch.UpSample2
        self.UpSample3 = predictbranch.UpSample3
        self.EEBlock1 = predictbranch.EEBlock1
        self.EEBlock2 = predictbranch.EEBlock2
        self.EEBlock3 = predictbranch.EEBlock3
        self.EEBlock4 = predictbranch.EEBlock4
        self.regression = predictbranch.classifier

        self.sigmoid = nn.Sigmoid()
        self.aspp = _AtrousSpatialPyramidPoolingModule_v2(self.n_classes, self.n_classes)
        self.bot_aspp = nn.Conv2d(6 * self.n_classes, self.n_classes, kernel_size=3, stride=1, padding=1)
        self.classificaton = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=3, stride=1, padding=1)


    def forward(self, x, batch_positions=None):
        x0 = x.permute(0, 2, 1, 3, 4)

        x1 = self.conv1.smart_forward(x0)
        x1 = self.ARBlock1(x1)
        s1 = self.TEBlock1(x1, batch_positions=batch_positions)
        s1 = self.EEBlock1(s1)

        x2 = self.conv2.smart_forward(x1)
        x2 = self.ARBlock2(x2)
        x2 = self.DownBlock1.smart_forward(x2)
        s2 = self.TEBlock2(x2, batch_positions=batch_positions)
        s2 = self.UpSample1(s2.contiguous())
        s2 = self.EEBlock2(s2)

        x3 = self.conv3.smart_forward(x2)
        x3 = self.ARBlock3(x3)
        x3 = self.DownBlock2.smart_forward(x3)
        s3 = self.TEBlock3(x3, batch_positions=batch_positions)
        s3 = self.UpSample2(s3.contiguous())
        s3 = self.EEBlock3(s3)

        x4 = self.conv4.smart_forward(x3)
        x4 = self.ARBlock4(x4)
        s4 = self.TEBlock4(x4, batch_positions=batch_positions)
        s4 = self.UpSample3(s4.contiguous())
        s4 = self.EEBlock4(s4)

        x5 = self.conv5.smart_forward(x4)
        x5 = self.ARBlock5(x5)

        x6 = self.conv6.smart_forward(x5)
        x6 = self.ARBlock6(x6)

        x7 = self.conv7.smart_forward(x6) 
        x7 = self.ARBlock7(x7)
        x7 = self.UpBlock3.smart_forward(x7)
        x7 = self.UpBlock3.smart_forward(x7)
        body_out = self.final(x7)
        body_out = body_out.permute(0, 1, 3, 4, 2)
        body_out = body_out.mean(dim=-1)

        edge_out = [s1, s2, s3, s4]
        seg_edge_out = self.regression(torch.cat(edge_out, dim=1))

        seg_final_out = self.aspp(body_out, seg_edge_out)
        seg_final_out = self.bot_aspp(seg_final_out)

        return seg_final_out, seg_edge_out
