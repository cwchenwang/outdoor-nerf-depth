set -e
DATA_ROOT=/ssd1/wangchen/data/nerfpp_preprocessed
SAVE_ROOT=/ssd1/wangchen/data/results/kitti/nerfpp
seq_names=(KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s657_e787_densegt KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s890_e1028_densegt KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt)
lambda_depth=0.1
# depth_sigma=0.02
sample_every=(4)
depth_loss_types=(mse)
depth_sup_types=(mono_crop mff_crop ste_conf_-1_crop ste_conf_0.95_crop)

# for l in ${!seq_names[@]}; do
#       for i in ${!sample_every[@]}; do
#           python ddp_train_nerf.py --config configs/kitti.txt \
#           --depth_loss_type ${depth_loss_types[j]} --depth_sup_type ${depth_sup_types[k]} \
#           --lambda_depth ${lambda_depth} --trainskip ${sample_every[i]} --scene ${seq_names[l]} \
#           --basedir ${SAVE_ROOT}/${seq_names[l]} \
#           --expname rgbonly_${sample_every[i]} \
#           --N_iters 100001
#       done
# done

for j in ${!depth_loss_types[@]}; do
  for l in ${!seq_names[@]}; do
    for k in ${!depth_sup_types[@]}; do
      for i in ${!sample_every[@]}; do
          python ddp_train_nerf.py --config configs/kitti.txt \
          --depth_loss_type ${depth_loss_types[j]} --depth_sup_type ${depth_sup_types[k]} \
          --lambda_depth ${lambda_depth} --trainskip ${sample_every[i]} --scene ${seq_names[l]} \
          --basedir ${SAVE_ROOT}/${seq_names[l]} \
          --expname ${depth_loss_types[j]}${lambda_depth}_${depth_sup_types[k]}_${sample_every[i]} \
          --N_iters 100001 --use_depth
      done
    done
  done
done