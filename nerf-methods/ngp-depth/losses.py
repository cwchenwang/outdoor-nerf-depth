import torch
from torch import nn
import vren


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3, lambda_depth=0.1):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self.lambda_depth = lambda_depth

    def forward(self, results, target, **kwargs):
        d = {}
        d['rgb'] = (results['rgb']-target['rgb'])**2

        # depth_mask = torch.logical_and(target['depth'] > 1e-4, target['depth'] < 80/16.0)
        depth_mask = target['depth'] > 1e-4
        d['depth'] = self.lambda_depth * ((results['depth']-target['depth'])**2 ) * depth_mask

        o = results['opacity']+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity*(-o*torch.log(o))

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d


def compute_depth_metrics(ground_truth_depth, pred_depth, valid, cap):

    valid_gt = ground_truth_depth.view(-1, 1)[valid].clamp(1e-3, cap)
    valid_pred = pred_depth.view(-1, 1)[valid].clamp(1e-3, cap)

    thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    mse = torch.mean((valid_gt - valid_pred) ** 2)
    rmse = torch.sqrt(mse)
    rmse_log = (torch.log(valid_gt) - torch.log(valid_pred)) ** 2
    rmse_log_tot = torch.sqrt(torch.mean(rmse_log))
    abs_diff = torch.mean(torch.abs(valid_gt - valid_pred))
    abs_rel = torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

    sq_rel = torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return mse, rmse, abs_rel