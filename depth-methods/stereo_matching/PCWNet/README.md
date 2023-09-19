# PCWNet (ECCV 2022 oral)
This is the pytorch implementation of the paper [PCW-Net: Pyramid Combination and Warping
Cost Volume for Stereo Matching](https://link.springer.com/chapter/10.1007/978-3-031-19824-3_17), `ECCV 2022 oral`, Zhelun Shen, Yuchao Dai, Xibin Song, Zhibo Rao, Dingfu Zhou and Liangjun Zhang 

# How to use
## Environment
* python 3.74
* Pytorch == 1.1.0
* Numpy == 1.15
## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), [ETH3D](https://www.eth3d.net/), [Middlebury](https://vision.middlebury.edu/stereo/)

**KITTI2015/2012 SceneFlow**

please place the dataset as described in `"./filenames"`, i.e., `"./filenames/sceneflow_train.txt"`, `"./filenames/sceneflow_test.txt"`, `"./filenames/kitticombine.txt"`

**Middlebury/ETH3D**

Our folder structure is as follows:
```
dataset
├── KITTI2015
├── KITTI2012
├── Middlebury
    │ ├── Adirondack
    │   ├── im0.png
    │   ├── im1.png
    │   └── disp0GT.pfm
├── ETH3D
    │ ├── delivery_area_1l
    │   ├── im0.png
    │   ├── im1.png
    │   └── disp0GT.pfm
```
Note that we use the half-resolution dataset of Middlebury for testing. 
## Training
**Scene Flow Datasets Pretraining**

run the script `./scripts/sceneflow.sh` to pre-train on Scene Flow datsets. Please update `DATAPATH` in the bash file as your training data path.
To repeat our pretraining details. You may need to replace the Mish activation function to Relu.  Samples are shown in `./models/relu/`.

**Finetuning**

run the script `./scripts/kitti15.sh` and `./scripts/kitti12.sh` to finetune our pretraining model on the KITTI dataset. Please update `DATAPATH` and `--loadckpt` as your training data path and pretrained SceneFlow checkpoint file.
## Evaluation
**Corss-domain Generalization**

run the script `./scripts/generalization_test.sh"` to test the cross-domain generalizaiton of the model (Table.2 of the main paper). Please update `--loadckpt` as pretrained SceneFlow checkpoint file.

**Finetuning Performance**

run the script `./scripts/kitti15_save.sh"` and `./scripts/kitti12_save.sh"` to generate the corresponding test images of KITTI 2015&2012

## Pretrained Models

[Sceneflow Pretraining Model](https://drive.google.com/file/d/18HglItUO7trfi-klXzqLq7KIDwPSVdAM/view?usp=sharing)

You can use this checkpoint to reproduce the result we reported in Table.2 of the main paper

[KITTI 2012 Finetuneing Moel](https://drive.google.com/file/d/14MANgQJ15Qzukv9SoL9MYobg5xUjE-u0/view?usp=sharing)

You can use this checkpoint to reproduce the result we submitted on KITTI 2012 benchmark.
