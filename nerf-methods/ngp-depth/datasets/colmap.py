import torch
import numpy as np
import os
import glob
from tqdm import tqdm

import cv2
from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary,\
        read_cameras_text, read_images_text, read_points3D_text

from .base import BaseDataset
from icecream import ic

class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()
        self.depth_pose_scale = None

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)


    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        # camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        cameras_bin_path = os.path.join(self.root_dir, 'sparse/0/cameras.bin')
        cameras_txt_path = os.path.join(self.root_dir, 'sparse/0/cameras.txt')
        if os.path.exists(cameras_bin_path):
            camdata = read_cameras_binary(cameras_bin_path)
        else:
            camdata = read_cameras_text(cameras_txt_path)
        
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, mod_ratio=1, depth_folder=None, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        # imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        images_bin_path = os.path.join(self.root_dir, 'sparse/0/images.bin')
        images_txt_path = os.path.join(self.root_dir, 'sparse/0/images.txt')
        if os.path.exists(images_bin_path):
            imdata = read_images_binary(images_bin_path)
        else:
            imdata = read_images_text(images_txt_path)

        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        if '360_v2' in self.root_dir and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        
        # depth_folder = "depths_lina"
        if depth_folder is None:
            raise ValueError("Please input the depth_folder !")
        depth_paths = [os.path.join(self.root_dir, depth_folder, name.replace('.jpg', '.png'))
                     for name in sorted(img_names)]
  
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        # pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        points3D_bin_path = os.path.join(self.root_dir, 'sparse/0/points3D.bin')
        points3D_txt_path = os.path.join(self.root_dir, 'sparse/0/points3D.txt')
        if os.path.exists(points3D_bin_path):
            pts3d = read_points3d_binary(points3D_bin_path)
        else:
            pts3d = read_points3D_text(points3D_txt_path)

        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)
        pts3d = None if pts3d.shape[0] == 0 else pts3d
        
        if pts3d is None:
            self.poses = center_poses(poses, pts3d)
        else:
            self.poses, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        # scale = np.linalg.norm(self.poses[..., 3], axis=-1).max()
        # if scale < 3:
        #     scale = 10

        # scale = 15.839978283413982 
        ic(scale)

        self.depth_pose_scale = scale

        self.poses[..., 3] /= scale
        self.pts3d = self.pts3d / scale if pts3d is not None else None

        self.rays = []
        if split == 'test_traj': # use precomputed test poses
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses)
            return

        if 'HDR-NeRF' in self.root_dir: # HDR-NeRF data
            if 'syndata' in self.root_dir: # synthetic
                # first 17 are test, last 18 are train
                self.unit_exposure_rgb = 0.73
                if split=='train':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                            f'train/*[024].png')))
                    self.poses = np.repeat(self.poses[-18:], 3, 0)
                elif split=='test':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                            f'test/*[13].png')))
                    self.poses = np.repeat(self.poses[:17], 2, 0)
                else:
                    raise ValueError(f"split {split} is invalid for HDR-NeRF!")
            else: # real
                self.unit_exposure_rgb = 0.5
                # even numbers are train, odd numbers are test
                if split=='train':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*0.jpg')))[::2]
                    img_paths+= sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*2.jpg')))[::2]
                    img_paths+= sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*4.jpg')))[::2]
                    self.poses = np.tile(self.poses[::2], (3, 1, 1))
                elif split=='test':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*1.jpg')))[1::2]
                    img_paths+= sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*3.jpg')))[1::2]
                    self.poses = np.tile(self.poses[1::2], (2, 1, 1))
                else:
                    raise ValueError(f"split {split} is invalid for HDR-NeRF!")
        else:
            # use every 8th image as test set
            # if split=='train':
            #     img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
            #     depth_paths = [x for i, x in enumerate(depth_paths) if i%8!=0]
            #     self.poses = np.array([x for i, x in enumerate(self.poses) if i%8!=0])
            # elif split=='test':
            #     img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
            #     depth_paths = [x for i, x in enumerate(depth_paths) if i%8==0]
            #     self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])

            def extract_trainval(L, ratios=1): # ratios=1,2,3,4
                vals = [i for i in range(9, L, 10)]
                trains = list(set(range(L)) - set(vals))
                if ratios == 1:
                    return trains, vals
                else:
                    trains_tmp = [trains[i] for i in range(0, len(trains), ratios)]
                    return trains_tmp, vals
            assert len(img_paths) == len(depth_paths)
            trains, vals = extract_trainval(len(img_paths), ratios=mod_ratio)
            print('train_id', trains)
            print('val_id',vals)

            if split=='train':
                img_paths = [img_paths[i] for i in trains]
                depth_paths = [depth_paths[i] for i in trains]
                self.poses = np.array([self.poses[i] for i in trains])
            elif split=='test':
                img_paths = [img_paths[i] for i in vals]
                depth_paths = [depth_paths[i] for i in vals]
                self.poses = np.array([self.poses[i] for i in vals])
            elif split=='render':
                # img_paths = [img_paths[i] for i in trains]
                # depth_paths = [depth_paths[i] for i in trains]
                # self.poses = np.array([self.poses[i] for i in trains])
                pass

        print(f'Loading {len(img_paths)} {split} images ...')
        for i, img_path in tqdm(enumerate(img_paths)):
            buf = [] # buffer for ray attributes: rgb, etc

            img = read_image(img_path, self.img_wh, blend_a=False)
            depth = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED).reshape(-1, 1).astype(float) / 256.0 / scale
            
            img = torch.FloatTensor(np.concatenate([img, depth], axis=-1))
            buf += [img]

            if 'HDR-NeRF' in self.root_dir: # get exposure
                folder = self.root_dir.split('/')
                scene = folder[-1] if folder[-1] != '' else folder[-2]
                if scene in ['bathroom', 'bear', 'chair', 'desk']:
                    e_dict = {e: 1/8*4**e for e in range(5)}
                elif scene in ['diningroom', 'dog']:
                    e_dict = {e: 1/16*4**e for e in range(5)}
                elif scene in ['sofa']:
                    e_dict = {0:0.25, 1:1, 2:2, 3:4, 4:16}
                elif scene in ['sponza']:
                    e_dict = {0:0.5, 1:2, 2:4, 3:8, 4:32}
                elif scene in ['box']:
                    e_dict = {0:2/3, 1:1/3, 2:1/6, 3:0.1, 4:0.05}
                elif scene in ['computer']:
                    e_dict = {0:1/3, 1:1/8, 2:1/15, 3:1/30, 4:1/60}
                elif scene in ['flower']:
                    e_dict = {0:1/3, 1:1/6, 2:0.1, 3:0.05, 4:1/45}
                elif scene in ['luckycat']:
                    e_dict = {0:2, 1:1, 2:0.5, 3:0.25, 4:0.125}
                e = int(img_path.split('.')[0][-1])
                buf += [e_dict[e]*torch.ones_like(img[:, :1])]

            self.rays += [torch.cat(buf, 1)]

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
    
    def get_depth_pose_scale(self, ):
        """
        docstring
        """

        assert self.depth_pose_scale is not None

        return self.depth_pose_scale