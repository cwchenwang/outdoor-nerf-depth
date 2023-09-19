set -e

DATA_ROOT=/root/paddlejob/workspace/wangchen/data/argo_highres
SAVE_ROOT=/root/paddlejob/workspace/wangchen/data/results/argo_highres
seq_names=(2c07fcda-6671-3ac0-ac23-4a232e0e031e 70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c cb0cba51-dfaf-34e9-a0c2-d931404c3dd8 ff78e1a3-6deb-34a4-9a1f-b85e34980f06)
# seq_names=(KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s890_e1028_densegt KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s2700_e3000_densegt KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt)
depth_loss_types=(mse)
depth_sup_types=(stereo)
lambda_depths=(10)
sample_every=(2)

for l in ${!seq_names[@]}; do
  for i in ${!sample_every[@]}; do
  checkpoint_dir=${SAVE_ROOT}/mipnerf360/${seq_names[l]}/checkpoints-${sample_every[i]}-rgbonly
  python -m train \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.max_steps = 75000" \
    --gin_bindings="Config.sample_every = ${sample_every[i]}" \
    --gin_bindings="Config.data_dir = '${DATA_ROOT}/${seq_names[l]}'" \
    --gin_bindings="Config.compute_disp_metrics = False" \
    --gin_bindings="Config.checkpoint_dir='${checkpoint_dir}'" \
    --logtostderr
  
  python ../utils/eval.py --pred_dir ${checkpoint_dir}/test_preds_75000 --gt_dir ${DATA_ROOT}/${seq_names[l]}/images --split ${sample_every[i]} --method mipnerf360
  done
done

# for i in ${!sample_every[@]}; do
# for k in ${!depth_sup_types[@]}; do
#   for j in ${!depth_loss_types[@]}; do
#     for l in ${!seq_names[@]}; do
#       checkpoint_dir=${SAVE_ROOT}/mipnerf360/${seq_names[l]}/checkpoints-${sample_every[i]}-${depth_loss_types[j]}-${depth_sup_types[k]}-lambda${lambda_depths[i]}
#       python -m train \
#         --gin_configs=configs/360.gin \
#         --gin_bindings="Config.max_steps = 75000" \
#         --gin_bindings="Config.sample_every = ${sample_every[i]}" \
#         --gin_bindings="Config.data_dir = '${DATA_ROOT}/${seq_names[l]}'" \
#         --gin_bindings="Config.compute_disp_metrics = True" \
#         --gin_bindings="Config.lambda_depth = ${lambda_depths[i]}" \
#         --gin_bindings="Config.depth_loss_type = '${depth_loss_types[j]}'" \
#         --gin_bindings="Config.depth_sup_type = '${depth_sup_types[k]}'" \
#         --gin_bindings="Config.checkpoint_dir='${checkpoint_dir}'" \
#         --logtostderr
#       python ../utils/eval.py --pred_dir ${checkpoint_dir}/test_preds_75000 --gt_dir ${DATA_ROOT}/${seq_names[l]}/images --split ${sample_every[i]} --method mipnerf360
#       done
#     done
#   done
# done
