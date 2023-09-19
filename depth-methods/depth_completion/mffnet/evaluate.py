#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/16 4:47 PM

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
import yaml
from easydict import EasyDict as edict
import datasets
import encoding
# from metrics import AverageMeter_all, Result
import csv
from criteria import OTHERS

fieldnames = [
    'epoch', 'rmse', 'mae', 'irmse', 'imae', 'mse', 'absrel', 'delta1', 'delta2', 'delta3', 'sqrel', 'rmselog'
]

def test():
    global best_metric
    Avg = AverageMeter()
    Avgrmse = AverageMeter()
    Avgmae = AverageMeter()
    Avgabsrel = AverageMeter()
    Avgdelta1 = AverageMeter()
    Avgdelta2 = AverageMeter()
    Avgdelta3 = AverageMeter()
    Avgirmse = AverageMeter()
    Avgimae = AverageMeter()
    Avgsqrel = AverageMeter()
    Avgrmselog = AverageMeter()

#     block_average_meter = AverageMeter_all()
#     average_meter = AverageMeter_all()
#     meters = [block_average_meter, average_meter]
    net.eval()
    for batch_idx, (rgb, lidar, depth, idx, ori_size) in enumerate(testloader):
        with torch.no_grad():
            if 1:
                rgbf = torch.flip(rgb, [-1])
                lidarf = torch.flip(lidar, [-1])
                rgbs = torch.cat([rgb, rgbf], 0)
                lidars = torch.cat([lidar, lidarf], 0)
                rgbs, lidars = rgbs.cuda(), lidars.cuda()
                depth_preds, = net(rgbs, lidars, depth, 'val')
                depth_pred, depth_predf = depth_preds.split(depth_preds.shape[0] // 2)
                depth_predf = torch.flip(depth_predf, [-1])
                depth_pred = (depth_pred + depth_predf) / 2.
            else:
                rgb, lidar = rgb.cuda(), lidar.cuda()
                depth_pred, = net(rgb, lidar, depth, 'val')
#             rgb, lidar = rgb.cuda(), lidar.cuda()
#             depth_pred, = net(rgb, lidar, depth, 'val')
            depth_pred[depth_pred < 0.9] = 0.9
            depth_pred[depth_pred >85] = 85
            prec = metric(depth_pred, depth).mean()
            rmse, mae, absrel, delta1, delta2, delta3, irmse, imae, sqrel, rmselog = others(depth_pred, depth)#.mean()
#             mini_batch_size = rgb.size(0)
#             result = Result()
# #             result = result.cuda()
#             result.evaluate(depth_pred, depth)
#             [
#                 m.update(result, mini_batch_size)
#                 for m in meters
#             ]
        Avg.update(prec.item(), rgb.size(0))
        Avgrmse.update(rmse, rgb.size(0))
        Avgmae.update(mae, rgb.size(0))
        Avgabsrel.update(absrel, rgb.size(0))
        Avgdelta1.update(delta1, rgb.size(0))
        Avgdelta2.update(delta2, rgb.size(0))
        Avgdelta3.update(delta3, rgb.size(0))
        Avgirmse.update(irmse, rgb.size(0))
        Avgimae.update(imae, rgb.size(0))
        Avgsqrel.update(sqrel, rgb.size(0))
        Avgrmselog.update(rmselog, rgb.size(0))
    
#     avg = average_meter.average()
    best_metric = Avg.avg
    print('Best Result: {:.4f}\n'.format(best_metric))
    
    best_metricrmse = Avgrmse.avg
    best_metricmae = Avgmae.avg
    best_metricabsrel = Avgabsrel.avg
    best_metricdelta1 = Avgdelta1.avg
    best_metricdelta2 = Avgdelta2.avg
    best_metricdelta3 = Avgdelta3.avg
    best_metricirmse = Avgirmse.avg
    best_metricimae = Avgimae.avg
    best_metricsqrel = Avgsqrel.avg
    best_metricrmselog = Avgrmselog.avg
    print('Best Result: rmse mae absrel delta1 delta2 delta3 irmse imae sqrel rmselog\n')
    print('Best Result: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(best_metricrmse, best_metricmae, best_metricabsrel, best_metricdelta1, best_metricdelta2, best_metricdelta3, best_metricirmse, best_metricimae, best_metricsqrel, best_metricrmselog))

    with open('val.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
                'epoch': 0,
                'rmse': best_metricrmse,
                'mae': best_metricmae,
                'irmse': best_metricirmse,
                'imae': best_metricimae,
                'absrel': best_metricabsrel,
                'delta1': best_metricdelta1,
                'delta2': best_metricdelta2,
                'delta3': best_metricdelta3
        })

        # depth_pred = depth_pred.cpu().squeeze(1).numpy()
        # idx = idx.cpu().squeeze(1).numpy()
        # ori_size = ori_size.cpu().numpy()
        # name = [testset.names[i] for i in idx]
        # save_result(config, depth_pred, name, ori_size)


if __name__ == '__main__':
    # config_name = 'GN.yaml'
    config_name = 'GNS.yaml'
    with open(os.path.join('configs', config_name), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    config = edict(config_data)
    from utils import *

    transform = init_aug(config.test_aug_configs)
    key, params = config.data_config.popitem()
    dataset = getattr(datasets, key)
    testset = dataset(**params, mode='selval', transform=transform, return_idx=True, return_size=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=config.num_workers,
                                             shuffle=False, pin_memory=True)
    print('num_test = {}'.format(len(testset)))
    metric = init_metric(config)
    net = init_net(config)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    net.cuda()
    net = encoding.parallel.DataParallelModel(net)
    others = OTHERS()
    others.cuda()
    others = encoding.parallel.DataParallelCriterion(others)
    metric.cuda()
    metric = encoding.parallel.DataParallelCriterion(metric)
    net = resume_state(config, net)
    test()
