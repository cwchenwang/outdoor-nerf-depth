# CFNet(CVPR 2021)
This is the implementation of the paper CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching, `CVPR 2021`, Zhelun Shen, Yuchao Dai, Zhibo Rao.

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
Note that we use the full-resolution image of Middlebury for training as the additional training images don't have half-resolution version. We will down-sample the input image to half-resolution in the data argumentation. In contrast,  we use the half-resolution image and full-resolution disparity of Middlebury for testing. 

## Training
**Scene Flow Datasets Pretraining**

run the script `./scripts/sceneflow.sh` to pre-train on Scene Flow datsets. Please update `DATAPATH` in the bash file as your training data path.

To repeat our pretraining details. You may need to replace the Mish activation function to Relu. Samples is shown in `./models/relu/`.

**Finetuning**

run the script `./scripts/robust.sh` to jointly finetune the pre-train model on four datasets,
i.e., KITTI 2015, KITTI2012, ETH3D, and Middlebury. Please update `DATAPATH` and `--loadckpt` as your training data path and pretrained SceneFlow checkpoint file.

## Evaluation
**Joint Generalization**

run the script `./scripts/eth3d_save.sh"`, `./scripts/mid_save.sh"` and `./scripts/kitti15_save.sh` to save png predictions on the test set of the ETH3D, Middlebury, and KITTI2015 datasets. Note that you may need to update the storage path of save_disp.py, i.e., `fn = os.path.join("/home3/raozhibo/jack/shenzhelun/cfnet/pre_picture/"`, fn.split('/')[-2]).

**Corss-domain Generalization**

run the script `./scripts/robust_test.sh"` to test the cross-domain generalizaiton of the model (Table.3 of the main paper). Please update `--loadckpt` as pretrained SceneFlow checkpoint file.

## Pretrained Models

[Pretraining Model](https://drive.google.com/file/d/1gFNUc4cOCFXbGv6kkjjcPw2cJWmodypv/view?usp=sharing)
You can use this checkpoint to reproduce the result we reported in Table.3 of the main paper

[Finetuneing Moel](https://drive.google.com/file/d/1H6L-lQjF4yOxq23wxs3HW40B-0mLxUiI/view?usp=sharing)
You can use this checkpoint to reproduce the result we reported in the stereo task of Robust Vision Challenge 2020