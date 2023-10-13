# Digging into Depth Priors for Outdoor Neural Radiance Fields
This is the official implementation of our ACM MM 2023 paper Digging into Depth Priors for Outdoor Neural Radiance Fields. Pull requests and issues are welcome.

### [Project Page](https://cwchenwang.github.io/outdoor-nerf-depth/) | [Paper](https://arxiv.org/abs/2308.04413)

Abstract: *Neural Radiance Fields (NeRF) have demonstrated impressive performance in vision and graphics tasks, such as novel view synthesis and immersive reality. However, the shape-radiance ambiguity of radiance fields remains a challenge, especially in the sparse viewpoints setting. Recent work resorts to integrating depth priors into outdoor NeRF training to alleviate the issue. However, the criteria for selecting depth priors and the relative merits of different priors have not been thoroughly investigated. Moreover, the relative merits of selecting different approaches to use the depth priors is also an unexplored problem. In this paper, we provide a comprehensive study and evaluation of employing depth priors to outdoor neural radiance fields, covering common depth sensing technologies and most application ways. Specifically, we conduct extensive experiments with two representative NeRF methods equipped with four commonly-used depth priors and different depth usages on two widely used outdoor datasets. Our experimental results reveal several interesting findings that can potentially benefit practitioners and researchers in training their NeRF models with depth priors.*

[Note] We have updated our code, but due to data migration issues, data are not available currently.

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
