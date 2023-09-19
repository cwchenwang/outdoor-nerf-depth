#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    criteria.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:51 PM

import torch
import torch.nn as nn
import numpy as np

__all__ = [
    'RMSE',
    'MSE',
]

import math
lg_e_10 = math.log(10)


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / lg_e_10
class OTHERS(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        valid_mask = (target > 1e-3).cuda()
        output_mm =  outputs[valid_mask]#.cuda()
        target_mm = target[valid_mask]#.cuda()

        abs_diff = (output_mm - target_mm).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.rmse_log = (torch.log(target_mm) - torch.log(output_mm)) ** 2
        self.rmse_log = math.sqrt(self.rmse_log.mean())
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        self.squared_rel = float((((abs_diff )**2)/ target_mm).mean())

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25**2).float().mean())
        self.delta3 = float((maxRatio < 1.25**3).float().mean())

        
        # convert from meters to km
        inv_output_km = (1e-3 * outputs[valid_mask].cuda())**(-1)
        inv_target_km = (1e-3 * target[valid_mask].cuda())**(-1)
#         print(inv_target_km, inv_output_km)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        return self.rmse, self.mae, self.absrel, self.delta1, self.delta2, self.delta3, self.irmse, self.imae, self.squared_rel, self.rmse_log

class RMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 1e-3).float().cuda()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.sqrt(loss / cnt)

class MSEloss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
#         print(outputs.shape, target.shape)
#         print(outputs)
        val_pixels = (target > 1e-3).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        return loss ** 2
    

class ABSLoss(nn.Module):
    def __init__(self,ABSLoss_weight=1):
        super(ABSLoss,self).__init__()
        self.ABSLoss_weight = ABSLoss_weight
        self.crit = torch.nn.SmoothL1Loss(reduction='sum')

    def forward(self, outputs, target, *args):
#         print(outputs.shape, target.shape)
#         print(outputs)
        beta = 1.0
        normalizer = 1.0
        val_pixels = (target > 1e-3).float().cuda()
        ldiff = target * val_pixels - outputs * val_pixels
        diff = torch.abs(ldiff)
        cond = diff < beta
        loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return torch.sum(loss, dim=1) / normalizer
#         print((outputs * val_pixels).shape)
#         return self.crit(outputs * val_pixels, target * val_pixels)
    
    
class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):

        loss = outputs
        return loss 
