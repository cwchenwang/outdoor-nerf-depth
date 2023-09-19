set -e
steps=(150000 100000 050000)
expname=kittiSeq02_170scans_depth0.1_1

for i in ${!steps[@]}; do
    python ddp_test_nerf.py --config configs/kitti.txt --expname ${expname} --ckpt_path ./logs/${expname}/model_${steps[$i]}.pth --port 12401
done