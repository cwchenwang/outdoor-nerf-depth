# outdoor-nerf-depth
[Note] We have updated our code, but due to data migration issues, data are not available currently.

## Depth Methods

### NeRF++
Train on Kitti data:
```
bash train.sh
```

Train on Argoverse data:
```
bash train_argo.sh
```

### MipNeRF-360
Train on Kitti data:
```
bash train_kitti.sh
```

Train on Argoverse data:
```
bash train_argo.sh
```

For all the training scripts, you can specify the following configs:
- `sample_every`: the sparity of training images, if it is `n`, then sample 1 image out of `n` for training
- `depth_loss_type`: type of depth losses, available options are `mse`, `l1`, `kl`
- `depth_sup_type`: type of depth maps, available options `gt` (lidar data), `stereo_crop` (stereo depth estimation), `mono_crop` (monocular depth estimation), `mff_crop` (depth completion), `rgbonly` (don't use depth)
- `lambda_depth`: the weight of depth loss

## Citation
If you consider our paper or code useful, please cite our paper:
```
@article{wang2023digging,
    title={Digging into Depth Priors for Outdoor Neural Radiance Fields},
    author={Chen Wang and Jiadai Sun and Lina Liu and Chenming Wu 
            and Zhelun Shen and Dayan Wu and Yuchao Dai and Liangjun Zhang},
    journal={Proceedings of the 31th ACM International Conference on Multimedia},
    year={2023}
}
```
