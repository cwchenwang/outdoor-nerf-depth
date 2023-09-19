# [MFF-Net for depth supervision nerf](https://ieeexplore.ieee.org/document/10008014).


## Introduction

This is the pytorch implementation of our paper.

> **MFF-Net: Towards Efficient Monocular Depth Completion with Multi-modal Feature Fusion**
> Lina Liu, Xibin Song, Jiadai Sun, Xiaoyang Lyu, Lin Li, Yong Liu and Liangjun Zhang

## Dependency
```
PyTorch 1.4
PyTorch-Encoding v1.4.0
```

## Setup
Compile the C++ and CUDA code:
```
cd exts
python setup.py install
```

## Dataset
Please download KITTI [depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
dataset.
The structure of data directory:
```
└── datas
    └── kitti
        ├── data_depth_annotated
        │   ├── train
        │   └── val
        ├── data_depth_velodyne
        │   ├── train
        │   └── val
        ├── raw
        │   ├── 2011_09_26
        │   ├── 2011_09_28
        │   ├── 2011_09_29
        │   ├── 2011_09_30
        │   └── 2011_10_03
        ├── test_depth_completion_anonymous
        │   ├── image
        │   ├── intrinsics
        │   └── velodyne_raw
        └── val_selection_cropped
            ├── groundtruth_depth
            ├── image
            ├── intrinsics
            └── velodyne_raw
```

## Configs
- GNS.yaml


## Trained Models
You can directly download the trained model and put it in *checkpoints*:
- [MFF-Net](https://drive.google.com/file/d/1eLy1v_EjqeM3yUqTTnot9MugeGDkgauZ/view?usp=share_link)

## Train 
You can also train by yourself:
```
python train.py
```
*Pay attention to the settings in the config file (e.g. gpu id, data_config.kitti.path).*

## Evaluate
With the trained model, 
you can evaluate the metrics.
```
python evaluate.py
```

## Test
With the trained model, 
you can test and save completed depth images.
```
python test.py
```

## Citation
If you find this work useful in your research, please consider citing:
```
@article{liu2023mff,
  title={MFF-Net: Towards Efficient Monocular Depth Completion with Multi-modal Feature Fusion},
  author={Liu, Lina and Song, Xibin and Sun, Jiadai and Lyu, Xiaoyang and Li, Lin and Liu, Yong and Zhang, Liangjun},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgement
Our code is based on [GuideNet](https://github.com/kakaxi314/GuideNet)

