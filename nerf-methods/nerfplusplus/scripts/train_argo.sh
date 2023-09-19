set -e
DATA_ROOT=/root/paddlejob/workspace/wangchen/data/argo_hr_nerfpp
SAVE_ROOT=/root/paddlejob/workspace/wangchen/data/results/argo_hr/nerfpp
seq_names=(2c07fcda-6671-3ac0-ac23-4a232e0e031e 70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c cb0cba51-dfaf-34e9-a0c2-d931404c3dd8 ff78e1a3-6deb-34a4-9a1f-b85e34980f06)
lambda_depth=1
# depth_sigma=0.02
sample_every=(2)
depth_loss_types=(mse)
depth_sup_types=(gt stereo_crop)

for l in ${!seq_names[@]}; do
    for i in ${!sample_every[@]}; do
        python ddp_train_nerf.py --config configs/kitti.txt \
        --depth_loss_type ${depth_loss_types[j]} --depth_sup_type ${depth_sup_types[k]} \
        --lambda_depth ${lambda_depth} --trainskip ${sample_every[i]} --scene ${seq_names[l]} \
        --basedir ${SAVE_ROOT}/${seq_names[l]}  --datadir ${DATA_ROOT} \
        --expname rgbonly_${sample_every[i]} --port 12345 \
        --N_iters 100001
    done
done

for j in ${!depth_loss_types[@]}; do
  for l in ${!seq_names[@]}; do
    for k in ${!depth_sup_types[@]}; do
      for i in ${!sample_every[@]}; do
          python ddp_train_nerf.py --config configs/kitti.txt \
          --depth_loss_type ${depth_loss_types[j]} --depth_sup_type ${depth_sup_types[k]} \
          --lambda_depth ${lambda_depth} --trainskip ${sample_every[i]} --scene ${seq_names[l]} \
          --basedir ${SAVE_ROOT}/${seq_names[l]} --datadir ${DATA_ROOT} \
          --expname ${depth_loss_types[j]}${lambda_depth}_${depth_sup_types[k]}_${sample_every[i]} \
          --N_iters 100001 --use_depth --port 12345
      done
    done
  done
done