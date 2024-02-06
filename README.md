# Digging into Depth Priors for Outdoor Neural Radiance Fields
Official implementation of our ACM MM 2023 paper "Digging into Depth Priors for Outdoor Neural Radiance Fields". Pull requests and issues are welcome.

### [Project Page](https://cwchenwang.github.io/outdoor-nerf-depth/) | [Paper](https://arxiv.org/abs/2308.04413) | [Dataset](https://drive.google.com/drive/folders/1pTlWLGsLxCjw8DlFaL71yyWFBe4W7X5W?usp=drive_link)

Abstract: *Neural Radiance Fields (NeRF) have demonstrated impressive performance in vision and graphics tasks, such as novel view synthesis and immersive reality. However, the shape-radiance ambiguity of radiance fields remains a challenge, especially in the sparse viewpoints setting. Recent work resorts to integrating depth priors into outdoor NeRF training to alleviate the issue. However, the criteria for selecting depth priors and the relative merits of different priors have not been thoroughly investigated. Moreover, the relative merits of selecting different approaches to use the depth priors is also an unexplored problem. In this paper, we provide a comprehensive study and evaluation of employing depth priors to outdoor neural radiance fields, covering common depth sensing technologies and most application ways. Specifically, we conduct extensive experiments with two representative NeRF methods equipped with four commonly-used depth priors and different depth usages on two widely used outdoor datasets. Our experimental results reveal several interesting findings that can potentially benefit practitioners and researchers in training their NeRF models with depth priors.*

## News
- [23/11/18] After unremitting efforts, we found the KITTI and Argoverse datasets used and made them public.

- [23/09/19] We have updated our code, but due to data migration issues, data are not available currently.

## Requirements
Follow the README for each method to setup the environment.

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


### Instant-ngp | [ngp_pl](https://github.com/kwea123/ngp_pl)
Train on Kitti data:
```
bash auto_batch_run_kittiseq.sh
```

Train on Argoverse data:
```
bash auto_batch_run_argoseq.sh
```

## Dataset

We finally select five sequences from Seq 00, 02, 05, 06 in KITTI (125, 133, 175, 295, 320 frames) and three sequences from Argoverse (73, 72, 73 frames).

Download our selected and reorganized KITTI (1.68G) and Argoverse (890M) data from [here](https://drive.google.com/drive/folders/1pTlWLGsLxCjw8DlFaL71yyWFBe4W7X5W?usp=drive_link)

<details>
  <summary>[Expected directory structure of KITTI & Argoverse (click to expand)]</summary>
  
```shell
kitti_select_static_5seq
├── KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s2700_e3000_densegt
│   ├── cameras.npz
│   ├── depths_gt               # Raw LiDAR depth
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── 00000294.png
│   ├── depths_mff_crop         # Depth completion
│   ├── depths_mono_crop        # Monocular depth estimation
│   ├── depths_ste_conf_-1_crop # Stereo depth estimation
│   ├── images                  # Raw color RGB images
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── 00000294.png
│   └── sparse                  # the colmap format
│       └── 0
│           ├── cameras.txt
│           ├── images.txt
│           └── points3D.txt

argoverse_3seq
├── 2c07fcda-6671-3ac0-ac23-4a232e0e031e
│   ├── depths                  # Raw LiDAR depth
│   ├── depths_mono_crop        # Monocular depth estimation
│   ├── depths_ste_crop         # Stereo depth estimation
│   ├── images                  # Raw color RGB images
│   └── sparse                  # the colmap format
```
</details>

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
