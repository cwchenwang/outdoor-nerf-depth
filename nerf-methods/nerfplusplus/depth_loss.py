import numpy as np
import torch

def depth_mse(depth_gt, depth_pred, weight=None):
    '''
    param: depth_gt, (N_rays, 1)
    param: depth_pred, (N_rays, 1)
    '''
    mask = (depth_gt > 0.0)
    return torch.mean((depth_gt[mask] - depth_pred[mask]) ** 2)

def depth_l1(depth_gt, depth_pred, weight=None):
    '''
    param: depth_gt, (N_rays, 1)
    param: depth_pred, (N_rays, 1)
    '''
    mask = (depth_gt > 0.0)
    return torch.mean(torch.abs(depth_gt[mask] - depth_pred[mask]))

def depth_kl(
    weights,
    termination_depth,
    steps,
    lengths,
    sigma,
    fg_far_depth = None
):
    """Depth loss from Depth-supervised NeRF (Deng et al., 2022).
    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        steps: Sampling distances along rays.
        lengths: Distances between steps.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    mask = (termination_depth > 0)
    if fg_far_depth is not None:
        mask = torch.logical_and(mask, (termination_depth < fg_far_depth))

    loss = -torch.log(weights + 1e-5) * torch.exp(-((steps - termination_depth[:, None]) ** 2) / (2 * sigma)) * lengths
    loss = loss[mask].sum(-2)
    return torch.mean(loss)

def depth_light_of_sight():
    pass

def is_not_in_expected_distribution(depth_mean, depth_var, depth_measurement_mean, depth_measurement_std):
    delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - depth_measurement_std) > 0.
    var_greater_than_expected = depth_measurement_std.pow(2) < depth_var
    return torch.logical_or(delta_greater_than_expected, var_greater_than_expected)

def depth_gaussian_log_likelihood(depth_map, z_vals, weights, target_depth, target_valid_depth):
    '''
    param: depth_map, (N_rays, 1)
    param: z_vals, z_vals in NeRF
    param: target_depth, (N_rays, 1)
    param: target_valid_depth, (N_rays, 1), valid mask for gt
    '''
    pred_mean = depth_map[target_valid_depth]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = ((z_vals[target_valid_depth] - pred_mean.unsqueeze(-1)).pow(2) * weights[target_valid_depth]).sum(-1) + 1e-5
    target_mean = target_depth[..., 0][target_valid_depth]
    target_std = target_depth[..., 1][target_valid_depth]
    apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_std)
    pred_mean = pred_mean[apply_depth_loss]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = pred_var[apply_depth_loss]
    target_mean = target_mean[apply_depth_loss]
    target_std = target_std[apply_depth_loss]
    f = nn.GaussianNLLLoss(eps=0.001)
    return float(pred_mean.shape[0]) / float(target_valid_depth.shape[0]) * f(pred_mean, target_mean, pred_var)

