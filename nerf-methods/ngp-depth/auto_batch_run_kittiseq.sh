#!/bin/bash

ROOT=/data/ngp-depth/data_KITTI_10

for sid in 'KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s657_e787_densegt' 'KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s2700_e3000_densegt' 'KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt' 'KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt' 'KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s890_e1028_densegt' 

do 
	ROOT_DIR="$ROOT"/"$sid"
	s_str=$(echo ${sid} | awk -F "_" '{print $9}')
	e_str=$(echo ${sid} | awk -F "_" '{print $10}')
	keyname=${sid:5:5}_${s_str}_${e_str}

	SCALE=10
	MOD_RATIO=1
	DEPTH_LOSS_WEIGHT=1
	BS=15000

	for DEPTH_DIR in 'depths_gt' 'depths_mono_crop' 'depths_mff_crop' 'depths_ste_conf_-1_crop' 
	do
	python train.py --root_dir $ROOT_DIR \
			--scale $SCALE \
			--mod_ratio $MOD_RATIO \
			--eval_lpips \
			--depth_loss_w $DEPTH_LOSS_WEIGHT \
			--depth_folder $DEPTH_DIR \
			--check_val_every_n_epoch 1 \
			--batch_size $BS \
			--exp_name OnlineScale_mod"$MOD_RATIO"_"$keyname"_"$DEPTH_DIR"_scale"$SCALE"_depth"$DEPTH_LOSS_WEIGHT"_bs"$BS"
	done

	# SCALE=5
	MOD_RATIO=4
	# DEPTH_LOSS_WEIGHT=7

	for DEPTH_DIR in 'depths_gt' 'depths_mono_crop' 'depths_mff_crop' 'depths_ste_conf_-1_crop' 
	do
	python train.py --root_dir $ROOT_DIR \
			--scale $SCALE \
			--mod_ratio $MOD_RATIO \
			--eval_lpips \
			--depth_loss_w $DEPTH_LOSS_WEIGHT \
			--depth_folder $DEPTH_DIR \
			--check_val_every_n_epoch 1 \
			--batch_size $BS \
			--exp_name OnlineScale_mod"$MOD_RATIO"_"$keyname"_"$DEPTH_DIR"_scale"$SCALE"_depth"$DEPTH_LOSS_WEIGHT"_bs"$BS"
	done
done