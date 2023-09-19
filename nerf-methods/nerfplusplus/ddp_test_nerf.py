import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
# from collections import OrderedDict
# from ddp_model import NerfNet
from tensorboardX import SummaryWriter
import time
from data_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, render_single_image, create_nerf
import logging


logger = logging.getLogger(__package__)


def ddp_test_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size, args.port)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]

    if rank == 0:
        writer = SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))

    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth, depth_sup_type=args.depth_sup_type)
        f = open(os.path.join(out_dir, f'psnr_{start:06d}.txt'), 'w')
        psnrs = []
        rmses = []
        abs_rels = []
        rmse_tot, rmse_log_tot, abs_diff, abs_rel, sq_rel = 0, 0, 0, 0, 0
        for idx in range(len(ray_samplers)):
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers[idx].img_path is not None:
                fname = os.path.basename(ray_samplers[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue

            time0 = time.time()
            ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size)
            dt = time.time() - time0
            if rank == 0:    # only main process should do this
                logger.info('Rendered {} in {} seconds'.format(fname, dt))

                # only save last level
                im = ret[-1]['rgb'].numpy()
                # compute psnr if ground-truth is available
                if ray_samplers[idx].img_path is not None:
                    gt_im = ray_samplers[idx].get_img()
                    psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                    logger.info('{}: psnr={}'.format(fname, psnr))
                    psnrs.append(psnr)

                if ray_samplers[idx].depth_gt_path is not None:
                    depth_gt = ray_samplers[idx].get_gt_depth_img()
                    current_gt_sparse = depth_gt / ray_samplers[idx].get_depth_scale()

                    current_pred = ret[-1]['depth'].numpy() / ray_samplers[idx].get_depth_scale()
                    depth_image = (current_pred.clip(1e-3, 80) * 256.0).astype(np.uint16)
                    imageio.imwrite(os.path.join(out_dir, 'depth_' + fname), depth_image)

                    cap = 80
                    valid = (current_gt_sparse < cap)&(current_gt_sparse>1e-3)
                    valid_gt = current_gt_sparse[valid].clip(1e-3, cap)
                    valid_pred = current_pred[valid]
                    valid_pred = valid_pred.clip(1e-3,cap)

                    thresh = np.maximum((valid_gt / valid_pred), (valid_pred / valid_gt))
                    # a1 += (thresh < 1.25).float().mean()
                    # a2 += (thresh < 1.25 ** 2).float().mean()
                    # a3 += (thresh < 1.25 ** 3).float().mean()
                    rmse = (valid_gt - valid_pred) ** 2
                    rmse_tot += np.sqrt(np.mean(rmse))
                    rmses.append(np.sqrt(np.mean(rmse)))
                    rmse_log = (np.log(valid_gt) - np.log(valid_pred)) ** 2
                    rmse_log_tot += np.sqrt(np.mean(rmse_log))

                    abs_diff += np.mean(np.abs(valid_gt - valid_pred))
                    abs_rel += np.mean(np.abs(valid_gt - valid_pred) / valid_gt)

                    abs_rels.append(np.mean(np.abs(valid_gt - valid_pred) / valid_gt))

                    sq_rel += np.mean(((valid_gt - valid_pred)**2) / valid_gt)

                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, fname), im)

                im = ret[-1]['fg_rgb'].numpy()
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'fg_' + fname), im)

                im = ret[-1]['bg_rgb'].numpy()
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'bg_' + fname), im)

                im = ret[-1]['fg_depth'].numpy()
                im = colorize_np(im, cmap_name='jet', append_cbar=True)
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)

                im = ret[-1]['bg_depth'].numpy()
                im = colorize_np(im, cmap_name='jet', append_cbar=True)
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)

            torch.cuda.empty_cache()
        if len(psnrs) > 0:
            psnr_mean = np.mean(psnrs)
            psnrs.append(psnr_mean)
            writer.add_scalar('test_psnr', psnr_mean, start)
            f.write('\n'.join([str(p) for p in psnrs]))
        f.close()

        if len(rmses) > 0:
            rmse_mean = np.mean(rmses)
            rmses.append(rmse_mean)
            with open(os.path.join(out_dir, f'rmse_{start:06d}.txt'), 'w') as f:
                f.write('\n'.join([str(p) for p in rmses]))
            writer.add_scalar('test_rmse', rmse_mean, start)
            absrel_mean = np.mean(abs_rels)
            abs_rels.append(absrel_mean)
            with open(os.path.join(out_dir, f'absrel_{start:06d}.txt'), 'w') as f:
                f.write('\n'.join([str(p) for p in abs_rels]))
            writer.add_scalar('test_absrel', absrel_mean, start)

    # clean up for multi-processing
    cleanup()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_test_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    test()

