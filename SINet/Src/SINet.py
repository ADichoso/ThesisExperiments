import torch
import torch.nn as nn
import torchvision.models as models
from .SearchAttention import SA
from Src.backbone.ResNet import ResNet_2Branch
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class PDC_SM(nn.Module):
    # Partial Decoder Component (Search Module)
    def __init__(self, channel):
        super(PDC_SM, self).__init__()
        self.relu = nn.ReLU(True)

        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        # print x1.shape, x2.shape, x3.shape, x4.shape
        x1_1 = x1

        # x2_1: upsample x1 to x2's size, then conv, then elementwise multiply with x2
        x1_up_x2 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x2_1 = self.conv_upsample1(x1_up_x2) * x2

        # x3_1: upsample x1 twice to x3's size, x2 once to x3's size, then apply convs and multiply
        x1_up_x3 = F.interpolate(x1, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x2_up_x3 = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x3_1 = self.conv_upsample2(x1_up_x3) * self.conv_upsample3(x2_up_x3) * x3

        # x2_2: concat x2_1 and upsampled x1_1 to x2_1 size, then conv
        x1_1_up_x2 = F.interpolate(x1_1, size=x2_1.shape[2:], mode='bilinear', align_corners=True)
        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_1_up_x2)), dim=1)
        x2_2 = self.conv_concat2(x2_2)

        # x3_2: concat x3_1, upsampled x2_2 to x3_1 size, and x4
        x2_2_up_x3 = F.interpolate(x2_2, size=x3_1.shape[2:], mode='bilinear', align_corners=True)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2_up_x3), x4), dim=1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class PDC_IM(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(PDC_IM, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1

        # Upsample x1 to x2's size and apply conv then elementwise multiply with x2
        x1_up_x2 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x2_1 = self.conv_upsample1(x1_up_x2) * x2

        # Upsample x1 twice and x2 once to x3's size, then apply convs and multiply with x3
        x1_up_x3 = F.interpolate(x1, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x2_up_x3 = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x3_1 = self.conv_upsample2(x1_up_x3) * self.conv_upsample3(x2_up_x3) * x3

        # Concatenate x2_1 and upsampled x1_1 to x2_1's size
        x1_1_up_x2 = F.interpolate(x1_1, size=x2_1.shape[2:], mode='bilinear', align_corners=True)
        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_1_up_x2)), dim=1)
        x2_2 = self.conv_concat2(x2_2)

        # Concatenate x3_1 and upsampled x2_2 to x3_1's size
        x2_2_up_x3 = F.interpolate(x2_2, size=x3_1.shape[2:], mode='bilinear', align_corners=True)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2_up_x3)), dim=1)

        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SINet_ResNet50(nn.Module):
    # ResNet-based encoder-decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_ResNet50, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_SM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

        self.SA = SA()

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        x0 = self.resnet.maxpool(x0)         # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x0)          # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)          # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        x01 = torch.cat((x0, x1), dim=1)     # (BS, 320, 88, 88)
        x01_down = self.downSample(x01)      # (BS, 320, 44, 44)
        x01_sm_rf = self.rf_low_sm(x01_down) # (BS, 32, 44, 44)

        x2_sm = x2
        x3_sm = self.resnet.layer3_1(x2_sm)  # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)  # (2048, 11, 11)

        # Resize for concat using F.interpolate
        x3_sm_up = F.interpolate(x3_sm, size=x2_sm.shape[2:], mode='bilinear', align_corners=True)
        x4_sm_up2 = F.interpolate(x4_sm, size=x2_sm.shape[2:], mode='bilinear', align_corners=True)
        x4_sm_up = F.interpolate(x4_sm, size=x3_sm.shape[2:], mode='bilinear', align_corners=True)

        x2_sm_cat = torch.cat((x2_sm, x3_sm_up, x4_sm_up2), dim=1)  # 3584 channels
        x3_sm_cat = torch.cat((x3_sm, x4_sm_up), dim=1)             # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)           # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                        # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                        # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)

        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        out_size = x.shape[2:]  # Original input size
        return (
            F.interpolate(camouflage_map_sm, size=out_size, mode='bilinear', align_corners=True),
            F.interpolate(camouflage_map_im, size=out_size, mode='bilinear', align_corners=True)
        )

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict:
                all_params[k] = pretrained_dict[k]
            elif '_1' in k:
                base = k.replace('_1', '')
                all_params[k] = pretrained_dict[base]
            elif '_2' in k:
                base = k.replace('_2', '')
                all_params[k] = pretrained_dict[base]

        assert len(all_params) == len(self.resnet.state_dict())
        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')