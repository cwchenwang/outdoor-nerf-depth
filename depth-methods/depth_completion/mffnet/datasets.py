#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    datasets.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 8:08 PM

import os
import numpy as np
import glob
from PIL import Image
import torch.utils.data as data

__all__ = [
    'kitti',
]


class kitti(data.Dataset):
    """
    kitti depth completion dataset: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """

    def __init__(self, path='../datas/kitti', mode='train', height=256, width=1216, return_idx=False, return_size=False,
                 transform=None):
        self.base_dir = path
        self.height = height
        self.width = width
        self.mode = mode
        self.return_idx = return_idx
        self.return_size = return_size
        self.transform = transform
        if mode in ['train', 'val']:
            self.depth_path = os.path.join(self.base_dir, 'data_depth_annotated', mode)
            self.lidar_path = os.path.join(self.base_dir, 'data_depth_velodyne', mode)
            self.depths = list(sorted(glob.iglob(self.depth_path + "/**/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/**/*.png", recursive=True)))
            
#             # add
#             part = list(range(0,85898,4))
#             path_part = {}
#             depths_part = []
#             lidars_part = []
#             for p in part:
#                 depths_part.append(self.depths[p])
#                 lidars_part.append(self.lidars[p])
#             self.depths = depths_part
#             self.lidars = lidars_part
# #             print(len(self.depths), len(self.lidars))
        elif mode == 'selval':
            self.lidar_path = os.path.join(self.base_dir, 'data_depth_velodyne', 'train')
            self.image_path = os.path.join(self.base_dir, 'data_rgb', 'train')
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/2011_10_03_drive_0027_sync/proj_depth/velodyne_raw/image_02/*.png", recursive=True)))
            self.images = list(sorted(glob.iglob(self.image_path + "/2011_10_03_drive_0027_sync/image_02/data/*.png", recursive=True)))
            self.lidars = self.lidars[2695:2996]
            self.images = self.images[2695:2996]
            self.depth_path = os.path.join(self.base_dir, 'data_depth_annotated', 'train')
            self.depths = list(sorted(glob.iglob(self.depth_path + "/2011_10_03_drive_0027_sync/proj_depth/groundtruth/image_02/*.png", recursive=True)))
            self.depths = self.depths[2695:2996]
#             print(len(self.depths), len(self.lidars))
        elif mode == 'test':
#         data_depth_velodyne/train/2011_10_03_drive_0027_sync/proj_depth/velodyne_raw/image_02/*.png
            self.lidar_path = os.path.join(self.base_dir, 'data_depth_velodyne', 'train')
            self.image_path = os.path.join(self.base_dir, 'data_rgb', 'train')
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/2011_10_03_drive_0027_sync/proj_depth/velodyne_raw/image_02/*.png", recursive=True)))
            self.images = list(sorted(glob.iglob(self.image_path + "/2011_10_03_drive_0027_sync/image_02/data/*.png", recursive=True)))
            self.lidars = self.lidars[2695:2996]
            self.images = self.images[2695:2996]
#             print(self.lidars)
            self.depths = self.lidars
        else:
            raise ValueError("Unknown mode: {}".format(mode))
        assert (len(self.depths) == len(self.lidars))
        self.names = [os.path.split(path)[-1] for path in self.depths]

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, index):

        depth = self.pull_DEPTH(self.depths[index])
        depth = np.expand_dims(depth, axis=2)
        lidar = self.pull_DEPTH(self.lidars[index])
        lidar = np.expand_dims(lidar, axis=2)
        file_names = self.depths[index].split('/')
        if self.mode in ['train', 'val']:
            rgb_path = '/'.join([self.base_dir] + ['data_rgb'] + file_names[-6:-4] +
                            file_names[-2:-1] + ['data'] + file_names[-1:])
        elif self.mode in ['selval', 'test']:
            rgb_path = self.images[index]
        else:
            ValueError("Unknown mode: {}".format(self.mode))
        rgb = self.pull_RGB(rgb_path)
        rgb = rgb.astype(np.float32)
        lidar = lidar.astype(np.float32)
        depth = depth.astype(np.float32)
        shape = lidar.shape
        if self.transform:
            rgb, lidar, depth = self.transform(rgb, lidar, depth)
        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        lidar = lidar.transpose(2, 0, 1).astype(np.float32)
        depth = depth.transpose(2, 0, 1).astype(np.float32)
        lp = (rgb.shape[2] - self.width) // 2
        rgb = rgb[:, -self.height:, lp:lp + self.width]
        lidar = lidar[:, -self.height:, lp:lp + self.width]
        depth = depth[:, -self.height:, lp:lp + self.width]
        output = [rgb, lidar, depth]
        if self.return_idx:
            output.append(np.array([index], dtype=int))
        if self.return_size:
            output.append(np.array(shape[:2], dtype=int))
        return output

    def pull_RGB(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img

    def pull_DEPTH(self, path):
        depth_png = np.array(Image.open(path), dtype=int)
        assert (np.max(depth_png) > 255)
        depth_image = (depth_png / 256.).astype(np.float32)
        return depth_image
