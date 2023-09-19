# STD2019 depth completion for depth supervision nerf

## Contents
1. [Dependency](#dependency)
0. [Trained Models](#trained-models)
0. [Commands](#commands)


## Dependency
This code was tested with Python 3 and PyTorch 1.0 on Ubuntu 16.04.
```bash
pip install numpy matplotlib Pillow
pip install torch torchvision # pytorch

# for self-supervised training requires opencv, along with the contrib modules
pip install opencv-contrib-python==3.4.2.16
```

## Trained Models
- supervised training (i.e., models trained with semi-dense lidar ground truth): http://datasets.lids.mit.edu/self-supervised-depth-completion/supervised/

## Commands
```
# generate completed depth
python main.py --evaluate [checkpoint-path] --val select
```

Contact with Lina Liu if you need any help.


