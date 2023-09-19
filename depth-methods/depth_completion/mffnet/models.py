#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:50 PM

import torch
import torch.nn as nn
from scipy.stats import truncnorm
import math
from torch.autograd import Function
import encoding
# import GuideConv
from criteria import MSEloss, ABSLoss

__all__ = [
    'GN',
    'GNS',
]


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Conv2dLocal_F(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = GuideConv.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = GuideConv.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight


class Conv2dLocal(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, input, weight):
        output = Conv2dLocal_F.apply(input, weight)
        return output


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Basic2dLocal(nn.Module):
    def __init__(self, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = Conv2dLocal()
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, weight):
        out = self.conv(input, weight)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Guide(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.local = Basic2dLocal(input_planes, norm_layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv11 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv12 = nn.Conv2d(input_planes, input_planes * 9, kernel_size=weight_ks, padding=weight_ks // 2)
        self.conv21 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv22 = nn.Conv2d(input_planes, input_planes * input_planes, kernel_size=1, padding=0)
        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic2d(input_planes, input_planes, norm_layer)

    def forward(self, input, weight):
        B, Ci, H, W = input.shape
        weight = torch.cat([input, weight], 1)
        weight11 = self.conv11(weight)
        weight12 = self.conv12(weight11)
        weight21 = self.conv21(weight)
        weight21 = self.pool(weight21)
        weight22 = self.conv22(weight21).view(B, -1, Ci)
        out = self.local(input, weight12).view(B, Ci, -1)
        out = torch.bmm(weight22, out).view(B, Ci, H, W)
        out = self.br(out)
        out = self.conv3(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out

def channel_shuffle(convx, convx_2, convx_3):
    b, c, h, w = convx.size()
    cat_all = torch.cat((convx, convx_2, convx_3),1)
    cat_reshape = torch.reshape(cat_all,(b,3,c,h,w))
    cat_transpose = cat_reshape.transpose(1,2).contiguous()
    cat_view = cat_transpose.view(b, -1, h, w)
    convx = cat_view[:,:c,:,:]
    convx_2 = cat_view[:,c:2*c,:,:]
    convx_3 = cat_view[:,2*c:,:,:]

    return convx, convx_2, convx_3


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

class MMAF(nn.Module):

    def __init__(self, input_planes, out_planes, norm_layer=None, weight_ks=3, notmix=0):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv11 = nn.Sequential(Conv3x3(input_planes, out_planes),
                                #    nn.BatchNorm2d(out_planes),
                                   nn.Sigmoid())
        self.conv12 = nn.Sequential(Conv3x3(input_planes, out_planes),
                                    # nn.BatchNorm2d(out_planes),
                                   nn.PReLU())
                                    
        self.conv21 = nn.Sequential(Conv3x3(input_planes, out_planes),
                                    # nn.BatchNorm2d(out_planes),
                                   nn.Sigmoid())
        self.conv22 = nn.Sequential(Conv3x3(input_planes, out_planes),
                                    # nn.BatchNorm2d(out_planes),
                                   nn.PReLU())
        
        self.conv31 = nn.Sequential(Conv3x3(input_planes, out_planes),
                                    # nn.BatchNorm2d(out_planes),
                                   nn.Sigmoid())
        self.conv32 = nn.Sequential(Conv3x3(input_planes, out_planes),
                                    # nn.BatchNorm2d(out_planes),
                                   nn.PReLU())
        
        if notmix==1:
            self.conv33 = nn.Sequential(Conv3x3(input_planes*2, out_planes*2),
                                    # nn.BatchNorm2d(out_planes),
                                   nn.PReLU())
            self.conv41 = nn.Sequential(Conv1x1(input_planes*2, out_planes),
                                   nn.PReLU())
            self.conv42 = nn.Sequential(Conv1x1(input_planes*2, out_planes),
                                   nn.PReLU())
            self.conv43 = nn.Sequential(Conv1x1(input_planes*2, out_planes),
                                   nn.PReLU())
        else:
            self.conv34 = nn.Sequential(Conv3x3(input_planes*3, out_planes*3),
                                    # nn.BatchNorm2d(out_planes),
                                   nn.PReLU())
            self.conv41 = nn.Sequential(Conv1x1(input_planes*3, out_planes),
                                   nn.PReLU())
            self.conv42 = nn.Sequential(Conv1x1(input_planes*3, out_planes),
                                   nn.PReLU())
            self.conv43 = nn.Sequential(Conv1x1(input_planes*3, out_planes),
                                   nn.PReLU())
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.stdpool = nn.std()
        
        
        

    def forward(self, input1, input2, input3=None):
        B, Ci, H, W = input1.shape
        
        f11 = self.conv11(input1)
        f12 = self.conv12(input1)
        f1 = f11*f12
        
        f21 = self.conv21(input2)
        f22 = self.conv22(input2)
        f2 = f21*f22
        
        if input3==None:
            fs = self.conv33(torch.cat((f1, f2),1))
            fe1 = fs.view(B,-1,H*W).mean(-1)
            fe2 = torch.std(fs.view(B,-1,H*W),2)
            fe2 = torch.unsqueeze(fe2,-1)
            fe2 = torch.unsqueeze(fe2,-1)
            fe1 = torch.unsqueeze(fe1,-1)
            fe1 = torch.unsqueeze(fe1,-1)
            fe = fe1+fe2
            
            E1 = self.conv41(fe)
            E2 = self.conv42(fe)
            out = E1*f1 + E2*f2
            
        else:
            f31 = self.conv31(input3)
            f32 = self.conv32(input3)
            f3 = f31*f32
            fs = self.conv34(torch.cat((f1, f2, f3),1))
            fe1 = fs.view(B,-1,H*W).mean(-1)
            fe2 = torch.std(fs.view(B,-1,H*W),2)
            fe2 = torch.unsqueeze(fe2,-1)
            fe2 = torch.unsqueeze(fe2,-1)
            fe1 = torch.unsqueeze(fe1,-1)
            fe1 = torch.unsqueeze(fe1,-1)
            fe = fe1+fe2
#             print(fs.shape, fe.shape)
        
            E1 = self.conv41(fe)
            E2 = self.conv42(fe)
            E3 = self.conv43(fe) 
            out = E1*f1 + E2*f2 + E3*f3
        
        
        return out

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

#     # initialize the weights
#     for m in layers.modules():
#         init_weights(m)

    return layers


class MMAF_fuse(nn.Module):
    def __init__(self,inc, outc,notmix=0):
        super().__init__()
#         self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.conv13 = conv_bn_relu(inc,outc,kernel_size=3,stride=1,padding=1)
        self.conv14 = conv_bn_relu(outc,outc,kernel_size=3,stride=1,padding=1)
        self.conv15 = conv_bn_relu(outc,outc,kernel_size=3,stride=1,padding=1)
        self.conv16 = conv_bn_relu(outc,outc,kernel_size=3,stride=1,padding=1)
        self.conv13_2 = conv_bn_relu(inc,outc,kernel_size=3,stride=1,padding=1)
        self.conv14_2 = conv_bn_relu(outc,outc,kernel_size=3,stride=1,padding=1)
        self.conv15_2 = conv_bn_relu(outc,outc,kernel_size=3,stride=1,padding=1)
        self.conv16_2 = conv_bn_relu(outc,outc,kernel_size=3,stride=1,padding=1)
        self.conv13_3 = conv_bn_relu(inc,outc,kernel_size=3,stride=1,padding=1)
        self.conv14_3 = conv_bn_relu(outc,outc,kernel_size=3,stride=1,padding=1)
        self.conv15_3 = conv_bn_relu(outc,outc,kernel_size=3,stride=1,padding=1)
        self.conv16_3 = conv_bn_relu(outc,outc,kernel_size=3,stride=1,padding=1)
        self.mmaf6 = MMAF(outc,outc,notmix=notmix)
        self.mmaf5 = MMAF(outc,outc,notmix=notmix)
        self.mmaf4 = MMAF(outc,outc,notmix=notmix)
        self.mmaf3 = MMAF(outc,outc,notmix=notmix) 
    def forward(self, conv6, conv6_2, c_mix=None):
        conv13 = self.conv13(conv6)  # batchsize * ? * 176 * 608
        conv13_2 = self.conv13_2(conv6_2)  # batchsize * ? * 176 * 608
        if c_mix==None:
            conv13_all = self.mmaf3(conv13, conv13_2)
            conv14 = self.conv14(conv13)  # batchsize * ? * 88 * 304
            conv14_2 = self.conv14_2(conv13_2)  # batchsize * ? * 88 * 304
            conv14_all = self.mmaf4(conv14, conv14_2)
            conv15 = self.conv15(conv14)  # batchsize * ? * 44 * 152
            conv15_2 = self.conv15_2(conv14_2)  # batchsize * ? * 44 * 152
            conv15_all = self.mmaf5(conv15, conv15_2)
            conv16 = self.conv16(conv15)  # batchsize * ? * 22 * 76
            conv16_2 = self.conv16_2(conv15_2)  # batchsize * ? * 22 * 76
            conv16_all = self.mmaf6(conv16, conv16_2)
        else:
            conv13_3 = self.conv13_3(c_mix)  # batchsize * ? * 176 * 608
            conv13_all = self.mmaf3(conv13, conv13_2, conv13_3)
            conv14 = self.conv14(conv13)  # batchsize * ? * 88 * 304
            conv14_2 = self.conv14_2(conv13_2)  # batchsize * ? * 88 * 304
            conv14_3 = self.conv14_3(conv13_3)  # batchsize * ? * 88 * 304
            conv14_all = self.mmaf4(conv14, conv14_2, conv14_3)
            conv15 = self.conv15(conv14)  # batchsize * ? * 44 * 152
            conv15_2 = self.conv15_2(conv14_2)  # batchsize * ? * 44 * 152
            conv15_3 = self.conv15_3(conv14_3)  # batchsize * ? * 44 * 152
            conv15_all = self.mmaf5(conv15, conv15_2, conv15_3)
            conv16 = self.conv16(conv15)  # batchsize * ? * 22 * 76
            conv16_2 = self.conv16_2(conv15_2)  # batchsize * ? * 22 * 76
            conv16_3 = self.conv16_3(conv15_3)  # batchsize * ? * 22 * 76
            conv16_all = self.mmaf6(conv16, conv16_2, conv16_3)
        convmmaf_1 = torch.cat((conv13_all, conv14_all, conv15_all, conv16_all),1)
        
        return convmmaf_1

    
class GuideNet(nn.Module):
    """
    Not activate at the ref
    Init change to trunctated norm
    """

    def __init__(self, block=BasicBlock, bc=16, img_layers=[2, 2, 2, 2, 2],
                 depth_layers=[2, 2, 2, 2, 2], norm_layer=nn.BatchNorm2d,  weight_ks=3):
        super().__init__()
        self._norm_layer = norm_layer

        self.conv_img = Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)
        in_channels = bc * 2
        self.inplanes = in_channels
        self.layer1_img = self._make_layer(block, in_channels * 2, img_layers[0], stride=2)

#         self.guide1 = guide(in_channels * 2, in_channels * 2, norm_layer, weight_ks)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_img = self._make_layer(block, in_channels * 4, img_layers[1], stride=2)

#         self.guide2 = guide(in_channels * 4, in_channels * 4, norm_layer, weight_ks)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_img = self._make_layer(block, in_channels * 8, img_layers[2], stride=2)

#         self.guide3 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_img = self._make_layer(block, in_channels * 8, img_layers[3], stride=2)

#         self.guide4 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_img = self._make_layer(block, in_channels * 8, img_layers[4], stride=2)

        self.layer2d_img = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d_img = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d_img = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer5d_img = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)

        self.globalconv_lidar = Basic2d(1, bc * 2, norm_layer=None, kernel_size=5, padding=2)
        self.globalref = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)
        self.globalconv = nn.Conv2d(bc * 2, 1, kernel_size=3, stride=1, padding=1)
        
        self.conv_lidar = Basic2d(1, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)

        self.inplanes = in_channels
        self.layer1_lidar = self._make_layer(block, in_channels * 2, depth_layers[0], stride=2)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_lidar = self._make_layer(block, in_channels * 4, depth_layers[1], stride=2)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_lidar = self._make_layer(block, in_channels * 8, depth_layers[2], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_lidar = self._make_layer(block, in_channels * 8, depth_layers[3], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_lidar = self._make_layer(block, in_channels * 8, depth_layers[4], stride=2)
        
        self.conv_img_cs = Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)

        self.inplanes = in_channels
        self.layer1_img_cs = self._make_layer(block, in_channels * 2, depth_layers[0], stride=2)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_img_cs = self._make_layer(block, in_channels * 4, depth_layers[1], stride=2)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_img_cs = self._make_layer(block, in_channels * 8, depth_layers[2], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_img_cs = self._make_layer(block, in_channels * 8, depth_layers[3], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_img_cs = self._make_layer(block, in_channels * 8, depth_layers[4], stride=2)

        self.layer1d = Basic2dTrans(in_channels * 2*4, in_channels, norm_layer)
        self.layer2d = Basic2dTrans(in_channels * 4*4, in_channels * 2, norm_layer)
        self.layer3d = Basic2dTrans(in_channels * 8*4, in_channels * 4, norm_layer)
        self.layer4d = Basic2dTrans(in_channels * 8*4, in_channels * 8, norm_layer)
        self.layer5d = Basic2dTrans(in_channels * 8*3, in_channels * 8, norm_layer)

        self.conv = nn.Conv2d(bc * 2*2, 1, kernel_size=3, stride=1, padding=1)
        self.ref = block(bc * 2*2, bc * 2*2, norm_layer=norm_layer, act=False)
        
        
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.mmaf_fuse1 = MMAF_fuse(in_channels * 8,in_channels * 8, notmix=1)
        
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        self.mmaf_fuse2 = MMAF_fuse(in_channels * 8,in_channels * 8)
        
        self.pixel_shuffle3 = nn.PixelShuffle(2)
        self.mmaf_fuse3 = MMAF_fuse(in_channels * 8,in_channels * 4)
        
        self.pixel_shuffle4 = nn.PixelShuffle(2)
        self.mmaf_fuse4 = MMAF_fuse(in_channels * 4,in_channels * 2)
        
        self.pixel_shuffle5 = nn.PixelShuffle(2)
        self.mmaf_fuse5 = MMAF_fuse(in_channels * 2,in_channels * 1)
        
        self.convm = nn.Conv2d(61, 1, kernel_size=3, stride=1, padding=1)
        self.refm = block(61, 61, norm_layer=norm_layer, act=False)
        
        self.a = nn.Parameter(torch.ones(1))

        self._initialize_weights()
        
        self.loss = MSEloss()
        self.absloss = ABSLoss()

    def forward(self, img, lidar, gt, mode):
        
#         img2 = img.clone()
#         img2[:,0,:,:] = img2[:,0,:,:]*79.2382 + 90.995
#         img2[:,1,:,:] = img2[:,1,:,:]*80.5267 + 96.2278
#         img2[:,2,:,:] = img2[:,2,:,:]*82.1483 + 94.3213
#         img2 = img2 / 255.0 * 85.0
        
        
        c0_img = self.conv_img(img)
        c1_img = self.layer1_img(c0_img)
        c2_img = self.layer2_img(c1_img)
        c3_img = self.layer3_img(c2_img)
        c4_img = self.layer4_img(c3_img)
        c5_img = self.layer5_img(c4_img)
        dc5_img = self.layer5d_img(c5_img)
        c4_mix = dc5_img + c4_img
        dc4_img = self.layer4d_img(c4_mix)
        c3_mix = dc4_img + c3_img
        dc3_img = self.layer3d_img(c3_mix)
        c2_mix = dc3_img + c2_img
        dc2_img = self.layer2d_img(c2_mix)
        c1_mix = dc2_img + c1_img
        
        global_lidar = self.globalconv_lidar(lidar)
        globalref = self.globalref(global_lidar)
        globalref = self.globalconv(globalref)
        
        
        c0_lidar = self.conv_lidar(globalref)
        c0_img_cs = self.conv_img_cs(img)
        c1_lidar = self.layer1_lidar(c0_lidar)
        c1_img_cs = self.layer1_img_cs(c0_img_cs)
#         c1_lidar_dyn = self.guide1(c1_lidar, c1_mix)
        
        c1_lidar,c1_img_cs,  c1_mix = channel_shuffle(c1_lidar,c1_img_cs,  c1_mix)
        
        c2_lidar = self.layer2_lidar(c1_lidar)
        c2_img_cs = self.layer2_img_cs(c1_img_cs)
#         c2_lidar_dyn = self.guide2(c2_lidar, c2_mix)
        
        c2_lidar, c2_img_cs, c2_mix = channel_shuffle(c2_lidar, c2_img_cs, c2_mix)
        
        c3_lidar = self.layer3_lidar(c2_lidar)
        c3_img_cs = self.layer3_img_cs(c2_img_cs)
#         c3_lidar_dyn = self.guide3(c3_lidar, c3_mix)
        
        c3_lidar, c3_img_cs, c3_mix = channel_shuffle(c3_lidar,c3_img_cs,  c3_mix)
        
        c4_lidar = self.layer4_lidar(c3_lidar)
        c4_img_cs = self.layer4_img_cs(c3_img_cs)
#         c4_lidar_dyn = self.guide4(c4_lidar, c4_mix)
        
        c4_lidar, c4_img_cs, c4_mix = channel_shuffle(c4_lidar,c4_img_cs,  c4_mix)
        
        c5_lidar = self.layer5_lidar(c4_lidar)
        c5_img_cs = self.layer5_img_cs(c4_img_cs)
        c5 = torch.cat((c5_img, c5_lidar, c5_img_cs),1)#c5_img + c5_lidar + c5_img_cs
        
        dc5 = self.layer5d(c5) 
        c4 = torch.cat((dc5 , c4_lidar , c4_img_cs, c4_mix),1) #dc5 + c4_lidar_dyn + c4_img_cs
        
        dc4 = self.layer4d(c4) 
        c3 = torch.cat((dc4 , c3_lidar , c3_img_cs, c3_mix),1) #dc4 + c3_lidar_dyn + c3_img_cs
        
        dc3 = self.layer3d(c3) 
        c2 = torch.cat((dc3 , c2_lidar , c2_img_cs, c2_mix),1) #dc3 + c2_lidar_dyn + c2_img_cs
        
        dc2 = self.layer2d(c2) 
        c1 = torch.cat((dc2 , c1_lidar , c1_img_cs, c1_mix),1)#torch.cat((dc5 , c4_lidar_dyn , c4_img_cs),1)dc2 + c1_lidar_dyn + c1_img_cs
        
        dc1 = self.layer1d(c1) 
        c0 = torch.cat((dc1 , c0_lidar),1)#dc1 + c0_lidar
        output = self.ref(c0)
        output = self.conv(output)
        output = output + globalref
        
        convmmaf_1 = self.pixel_shuffle1(self.mmaf_fuse1(c5_lidar, c5_img_cs))
        convmmaf_2 = self.pixel_shuffle2(torch.cat((convmmaf_1, self.mmaf_fuse2(c4_lidar, c4_img_cs, c4_mix)),1))
        convmmaf_3 = self.pixel_shuffle3(torch.cat((convmmaf_2, self.mmaf_fuse3(c3_lidar, c3_img_cs, c3_mix)),1))
        convmmaf_4 = self.pixel_shuffle4(torch.cat((convmmaf_3, self.mmaf_fuse4(c2_lidar, c2_img_cs, c2_mix)),1))
        convmmaf_5 = self.pixel_shuffle5(torch.cat((convmmaf_4, self.mmaf_fuse5(c1_lidar, c1_img_cs, c1_mix)),1))
        outputm = self.refm(convmmaf_5)
        outputm = self.convm(outputm)
        outputm = outputm + globalref
        
        output_all = self.a * output + (1-self.a) * outputm
        
        if mode == 'train':
        
            loss1 = self.loss(output, gt) + self.absloss(output, gt)
            loss2 = self.loss(outputm, gt) + self.absloss(outputm, gt)
            loss3 = self.loss(output_all, gt) + self.absloss(output_all, gt)
            loss = loss1 + loss2 + loss3
        
            return (loss,)#(output,)
        else:
            return (output_all,)
    

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def GN():
    return GuideNet(norm_layer=encoding.nn.SyncBatchNorm)


def GNS():
    return GuideNet(norm_layer=encoding.nn.SyncBatchNorm,  weight_ks=1)