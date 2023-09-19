# NeRF++ with Depth Supervision
> python ddp_train_nerf.py --config configs/kitti.txt
Set ``use_depth = True`` if you want to add depth supervision, and set ``depth_loss_type`` and ``lambda_depth`` accordingly.

`depth_loss.py` contains some of the depth losses implemented.