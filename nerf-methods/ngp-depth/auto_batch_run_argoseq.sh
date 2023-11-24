#!/bin/bash

ROOT=/data/ngp-depth/data_argo/argoverse1_original

for sid in '2c07fcda-6671-3ac0-ac23-4a232e0e031e' '70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c' 'cb0cba51-dfaf-34e9-a0c2-d931404c3dd8'
do 
        ROOT_DIR="$ROOT"/"$sid"
        s_str=$(echo ${sid} | awk -F "_" '{print $9}')
        e_str=$(echo ${sid} | awk -F "_" '{print $10}')

        keyname=$sid

        SCALE=10
        MOD_RATIO=1
        DEPTH_LOSS_WEIGHT=0.5
        BS=8192

        for DEPTH_DIR in 'depths' 'depths_mono_crop' 'depths_ste_crop' 
        do
        python train.py --root_dir $ROOT_DIR \
                --scale $SCALE \
                --mod_ratio $MOD_RATIO \
                --eval_lpips \
                --depth_loss_w $DEPTH_LOSS_WEIGHT \
                --depth_folder $DEPTH_DIR \
                --check_val_every_n_epoch 1 \
                --batch_size $BS \
                --exp_name OriginalArgoFixScale_mod"$MOD_RATIO"_"$keyname"_"$DEPTH_DIR"_scale"$SCALE"_depth"$DEPTH_LOSS_WEIGHT"_bs"$BS"
        done

        # SCALE=5
        MOD_RATIO=2
        # DEPTH_LOSS_WEIGHT=7

        for DEPTH_DIR in 'depths' 'depths_mono_crop' 'depths_ste_crop' 
        do
        python train.py --root_dir $ROOT_DIR \
                --scale $SCALE \
                --mod_ratio $MOD_RATIO \
                --eval_lpips \
                --depth_loss_w $DEPTH_LOSS_WEIGHT \
                --depth_folder $DEPTH_DIR \
                --check_val_every_n_epoch 1 \
                --batch_size $BS \
                --exp_name OriginalArgoFixScale_mod"$MOD_RATIO"_"$keyname"_"$DEPTH_DIR"_scale"$SCALE"_depth"$DEPTH_LOSS_WEIGHT"_bs"$BS"
        done
done
