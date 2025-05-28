import os

import torch
import torch.nn as nn
from .resnet1 import resnet18
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CrossAtt(nn.Module):  # JointAtt
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1), stride=(1, 1))
        self.key1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 1), stride=(1, 1))
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1))

        self.query2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1), stride=(1, 1))
        self.key2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 1), stride=(1, 1))
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, (3, 3), padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())  # conv_f

    def forward(self, input1, input2):  # cross4为例
        batch_size, channels, height, width = input1.shape  # 8x64x64x64,  8x128x32x32,  8x256x16x16
        q1 = self.query1(input1)  # 8x8x64x64
        k1 = self.key1(input1).view(batch_size, -1, height * width)  # 8x16x4096
        v1 = self.value1(input1).view(batch_size, -1, height * width)  # 8x64x4096

        q2 = self.query2(input2)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        q = torch.cat([q1, q2], 1).view(batch_size, -1, height * width).permute(0, 2, 1)  # 拼接Q1,Q2, 8x4096x16
        attn_matrix1 = torch.bmm(q, k1)  # 8x4096x4096
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))  # 8x64x4096
        out1 = out1.view(*input1.shape)  # 8x64x64x64
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2)
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2  # 8x64x64x64

        feat_sum = self.conv_cat(torch.cat([out1, out2], 1))
        return feat_sum, out1, out2


class Classifier(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Classifier, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, (kernel_size, kernel_size), (stride, stride),
                              padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=(stride, stride),
                              padding=padding, dilation=(dilation, dilation), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel, norm_layer=nn.BatchNorm2d):
        super(Decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)  # conv(D1)
        left_mask = self.conv_l(left)  # conv(D2)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')  # Up(D1)
            z1 = F.relu(left_mask * down_, inplace=True)  # D1上采样与D2卷积的乘积
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')  # Up(conv(D1))

        z2 = F.relu(down_mask * left, inplace=True)  # D1卷积的上采样与D2的乘积

        out = torch.cat((z1, z2), dim=1)  # concat(Up(conv(D1))*D2,conv(D2)*Up(D1))
        return F.relu(self.bn3(self.conv3(out)), inplace=True)  # Dsp


class DMINet(nn.Module):
    def __init__(self, num_classes=2, normal_init=True, show_feature_maps=False):
        super(DMINet, self).__init__()
        self.show_Feature_Maps = show_feature_maps
        self.resnet = resnet18(pretrained=True)
        self.resnet.layer4 = nn.Identity()
        self.cross2 = CrossAtt(256, 256)  # JointAtt模块
        self.cross3 = CrossAtt(128, 128)
        self.cross4 = CrossAtt(64, 64)
        self.Translayer2_1 = BasicConv2d(256, 128, 1)  # 降维
        self.fam32_1 = Decode(128, 128, 128)  # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128, 64, 1)
        self.fam43_1 = Decode(64, 64, 64)  # AlignBlock(64) # decode(64,64,64)
        self.Translayer2_2 = BasicConv2d(256, 128, 1)
        self.fam32_2 = Decode(128, 128, 128)
        self.Translayer3_2 = BasicConv2d(128, 64, 1)
        self.fam43_2 = Decode(64, 64, 64)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Classifier(64, 32, 3, bn=True, relu=True),
            Classifier(32, num_classes, 3, bn=False, relu=False)
        )
        self.final2 = nn.Sequential(
            Classifier(64, 32, 3, bn=True, relu=True),
            Classifier(32, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Classifier(128, 32, 3, bn=True, relu=True),
            Classifier(32, num_classes, 3, bn=False, relu=False)
        )
        self.final2_2 = nn.Sequential(
            Classifier(128, 32, 3, bn=True, relu=True),
            Classifier(32, num_classes, 3, bn=False, relu=False)
        )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2):
                # ,project_name, name):

        c0 = self.resnet.conv1(imgs1)  # 1/2, 3->64
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)  # 1/4
        c1 = self.resnet.layer1(c1)  # 1/4, 64->64
        c2 = self.resnet.layer2(c1)  # 1/8, 64->128
        c3 = self.resnet.layer3(c2)  # 1/16, 128->256

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)  # JointAtt模块，拼接运算支路 8x256x16x16
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)  # 8x128x32x32
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2)  # 8x64x64x64

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(
            cross_result2))  # Dsp    D2, D1->Dsp  8x128x32x32, 8x128x16x16--->8x128x32x32
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(
            out3))  # Dagg     D3, Dsp->Dagg    8x64x64x64 , 8x64x32x32 --->8x64x64x64

        out3_2 = self.fam32_2(torch.abs(cur1_3 - cur2_3), self.Translayer2_2(torch.abs(cur1_2 - cur2_2)))  # 差分运算支路
        out4_2 = self.fam43_2(torch.abs(cur1_4 - cur2_4), self.Translayer3_2(out3_2))  # 8x64x64x64

        out4_up = self.upsamplex4(out4)  # 上采样  8x64x256x256
        out4_2_up = self.upsamplex4(out4_2)  # 8x64x256x256
        out_1 = self.final(out4_up)  # 预测头    # 8x2x256x256
        out_2 = self.final2(out4_2_up)  # 8x2x256x256

        out_1_2 = self.final_2(self.upsamplex8(out3))  # 只融合了后两层的特征图
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))
            # You can show more.

        return out_1, out_2, out_1_2, out_2_2

    def init_weights(self):
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)
