
# SCENE=gardenvase
# EXPERIMENT=360
DATA_ROOT=/root/paddlejob/workspace/wangchen/data/kitti_select_static_10seq
SAVE_ROOT=/root/paddlejob/workspace/wangchen/data/results/kitti
seq_names=(KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s2700_e3000_densegt)
depth_loss_types=(mse)
sample_every=(1)
lambda_depths=(10)
# depth_ratios=(0.005 0.01 0.025 0.05)
depth_sup_types=(gt mono_crop ste_conf_-1_crop ste_conf_0.95_crop)
# depth_sup_types=(gt)

# DATA_DIR=/ssd1/wangchen/data/bijc_0
# DATA_DIR=/ssd1/wangchen/data/kitti_depth_proj_mipnerf_100
# CHECKPOINT_DIR="${DATA_DIR}"/"checkpoints-depth"

for l in ${!seq_names[@]}; do
  for i in ${!sample_every[@]}; do
  checkpoint_dir=${SAVE_ROOT}/mipnerf360/${seq_names[l]}/checkpoints-${sample_every[i]}-rgbonly
  python -m eval \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.max_steps = 75000" \
    --gin_bindings="Config.sample_every = ${sample_every[i]}" \
    --gin_bindings="Config.data_dir = '${DATA_ROOT}/${seq_names[l]}'" \
    --gin_bindings="Config.compute_disp_metrics = False" \
    --gin_bindings="Config.checkpoint_dir='${checkpoint_dir}'" \
    --gin_bindings="Config.eval_suffix = 'left'" \
    --logtostderr
    done
done

for i in ${!sample_every[@]}; do
for k in ${!depth_sup_types[@]}; do
  for j in ${!depth_loss_types[@]}; do
    for l in ${!seq_names[@]}; do
      checkpoint_dir=${SAVE_ROOT}/mipnerf360/${seq_names[l]}/checkpoints-${sample_every[i]}-${depth_loss_types[j]}-${depth_sup_types[k]}-lambda${lambda_depths[i]}
      python -m eval \
        --gin_configs=configs/360.gin \
        --gin_bindings="Config.max_steps = 75000" \
        --gin_bindings="Config.sample_every = ${sample_every[i]}" \
        --gin_bindings="Config.data_dir = '${DATA_ROOT}/${seq_names[l]}'" \
        --gin_bindings="Config.compute_disp_metrics = True" \
        --gin_bindings="Config.lambda_depth = ${lambda_depths[i]}" \
        --gin_bindings="Config.depth_loss_type = '${depth_loss_types[j]}'" \
        --gin_bindings="Config.depth_sup_type = '${depth_sup_types[k]}'" \
        --gin_bindings="Config.checkpoint_dir='${checkpoint_dir}'" \
        --gin_bindings="Config.eval_suffix = 'left'" \
        --logtostderr
      done
    done
  done
done
