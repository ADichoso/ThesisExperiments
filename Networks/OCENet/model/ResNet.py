import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class B2_ResNet(nn.Module):
    # ResNet50 with two branches
    def __init__(self):
        # self.inplanes = 128
        self.inplanes = 64
        super(B2_ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3_1 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_1 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.inplanes = 512
        self.layer3_2 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_2 = self._make_layer(Bottleneck, 512, 3, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3_1(x)
        x1 = self.layer4_1(x1)

        x2 = self.layer3_2(x)
        x2 = self.layer4_2(x2)

        return x1, x2



class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=None):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv1.bias, 0)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv2.bias, 0)
        self.activation = nn.ReLU(inplace=False) if activation is None else activation
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn2(self.conv2(out1)))
        if self.in_size!=self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)

        return output


class UNetUpResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=None, space_dropout=False):
        super(UNetUpResBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm2d(out_size)

        init.xavier_uniform(self.up.weight, gain = np.sqrt(2.0))
        init.constant(self.up.bias,0)

        self.activation = nn.ReLU(inplace=False) if activation is None else activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size = kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.activation(self.bnup(self.up(x)))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out




class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.bn = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.dropout = nn.Dropout(p=0.5)

        init.kaiming_normal_(self.conv.weight)
        init.kaiming_normal_(self.conv2.weight)

    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = out.clone() #To Prevent Batch Norm inplace modifications
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        # self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2_1 = nn.BatchNorm2d(out_size)
        self.bn2_2 = nn.BatchNorm2d(out_size)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        # self.interpolate = F.interpolate(scale_factor=2, mode='bilinear', align_corners=True)

        init.kaiming_normal_(self.conv1.weight)
        init.kaiming_normal_(self.conv2_1.weight)
        init.kaiming_normal_(self.conv2_2.weight)

    # def center_crop(self, layer, target_size):
    #     batch_size, n_channels, layer_width, layer_height = layer.size()
    #     xy1 = (layer_width - target_size) // 2
    #     return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, u):#bridge is the corresponding lower layer
        # _, _, w, h = u.shape()
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout_1(out)
        out = torch.cat([out, u], dim=1)
        out = self.conv2_1(out)
        #out = out.clone() #To Prevent Batch Norm inplace modifications
        out = self.bn2_1(out)
        out = self.activation(out)
        out = self.conv2_2(out)
        #out = out.clone() #To Prevent Batch Norm inplace modifications
        out = self.bn2_2(out)
        out = self.activation(out)
        out = self.dropout_2(out)

        # crop1 = self.center_crop(bridge, up.size()[2])
        # out = torch.cat([up, crop1], 1)
        # out = self.activation(self.bn(self.conv(out)))
        # out = self.activation(self.bn2(self.conv2(out)))
        return out
