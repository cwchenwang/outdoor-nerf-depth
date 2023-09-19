# DATA_DIR=/ssd1/wangchen/kittiSeq02_llffdtu_2011_10_03_drive_0034_sync_100scans_densegt/DTU_format
# DATA_DIR=/ssd1/wangchen/kitti
# DATA_DIR=/ssd1/wangchen/data/kitti_depth_proj_mipnerf_100
DATA_DIR=/root/paddlejob/workspace/wangchen/data/kittiSeq02_llffdtu_2011_10_03_drive_0034_sync_170scans_densegt/DTU_format

python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.max_steps = 75000" \
  --gin_bindings="Config.sample_every = 1" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.compute_disp_metrics = True" \
  --gin_bindings="Config.depth_loss_type = 'mse'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/logs/checkpoints-1-7.5w-mse-debug'" \
  --logtostderr