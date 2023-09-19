from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import lpips
import torch
import numpy as np
import glob
import os
from PIL import Image
import argparse
# import dataset

device='cpu'
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

def lpips_fn(image_gt, image_pred, data_range):
    # normalize data range to [-1, 1]
    (dmin, dmax) = data_range
    image_gt = (torch.tensor(image_gt, dtype=torch.float32) - dmin) / (dmax - dmin) * 2 - 1
    image_pred = (torch.tensor(image_pred, dtype=torch.float32) - dmin) / (dmax - dmin) * 2 - 1
    with torch.no_grad():
        ret = loss_fn_vgg(image_gt.permute(2, 0, 1).unsqueeze(0).to(device), image_pred.permute(2, 0, 1).unsqueeze(0).to(device)).item()
    return ret

metric_dict = {
    'psnr': psnr_fn,
    'ssim': ssim_fn,
    'lpips': lpips_fn
}

def cal_metric(gts, preds, metric_fn, args, kwargs):
    # import pdb; pdb.set_trace();
    results = []
    tot = 0
    for (gt, pred) in zip(gts, preds):
        res = metric_fn(gt, pred, **args)
        results.append(res)
        tot += res
    return tot / len(gts), results

def evaluate(gts, preds, metrics):
    all_avg_metric = []
    all_single_metric = {}
    for metric_name in metrics:
        metric = metrics[metric_name]
        ret = cal_metric(gts, preds, metric['fn'], metric['args'], metric['kwargs'])
        all_avg_metric.append(ret[0])
        all_single_metric[metric_name] = ret[1]
        all_single_metric[metric_name].append(ret[0])
    return all_avg_metric, all_single_metric

metrics = {
    'psnr': {
        'fn': metric_dict['psnr'], 
        'args': {'data_range': 255}, 'kwargs': {}
    },
    'ssim': {
        'fn': metric_dict['ssim'],
        'args': {'data_range': 255, 'multichannel': True}, 'kwargs': {}
    },
    'lpips': {
        'fn': metric_dict['lpips'],
        'args': {'data_range': (0, 255)}, 'kwargs': {}
    }
}

def load_gts_preds(gt_dir, pred_dir, method, sample_every):
    gt_names = sorted(glob.glob(os.path.join(gt_dir, '*.jpg')))
    if len(gt_names) == 0:
        gt_names = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
    if sample_every >= 1:
      test_indices = [i for i in range(9, len(gt_names), 10)]
    #   train_names = list(set(range(len(gt_names))) - set(test_indices))
    gts = [np.array(Image.open(gt_names[idx])) for idx in test_indices]
    if method == 'mipnerf360':
        pred_names = sorted(glob.glob(os.path.join(pred_dir, 'color_*.png')))
    elif method == 'nerfpp':
        pred_names = sorted(glob.glob(os.path.join(pred_dir, '00*.png')))
    preds = [np.array(Image.open(name)) for name in pred_names]
    return gts, preds

args = argparse.ArgumentParser()
args.add_argument('--gt_dir', help="directory with ground truth images", type=str, default="./ground_truth")
args.add_argument('--pred_dir', help="directory with predicted images", type=str, default="./prediction")
args.add_argument('--method', help="method", type=str, default="mipnerf360")
args.add_argument('--split', type=int, default=4)

args = args.parse_args()
# pred_dir = '/root/paddlejob/wangchen/data/kittiSeq02_llffdtu_2011_10_03_drive_0034_sync_170scans_densegt/DTU_format/logs/checkpoints-4-7.5w-mse-ste_conf_0.95-lambda10/test_preds_75000'
# gt_dir = '/root/paddlejob/wangchen/data/kittiSeq02_llffdtu_2011_10_03_drive_0034_sync_170scans_densegt/DTU_format/images'
gts, preds = load_gts_preds(args.gt_dir, args.pred_dir, args.method, args.split)
ret = evaluate(gts, preds, metrics)

for item in ret[1]:
    with open(os.path.join(args.pred_dir, f'eval_{item}.txt'), 'w') as f:
        f.write('\n'.join([str(m) for m in ret[1][item]]))