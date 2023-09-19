import jax.numpy as jnp

URF_SIGMA_SCALE_FACTOR = 3.0

def ds_nerf_depth_loss(
    weights, #: TensorType[..., "num_samples", 1],
    termination_depth, #: TensorType[..., 1],
    steps, #: TensorType[..., "num_samples", 1],
    lengths, #: TensorType[..., "num_samples", 1],
    sigma #: TensorType[0],
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
    depth_mask = termination_depth > 0

    loss = -jnp.log(weights + 1e-7) * jnp.exp(-((steps - termination_depth[:, None]) ** 2) / (2 * sigma)) * lengths
    loss = loss.sum(-2) * depth_mask
    return jnp.mean(loss)

def urban_radiance_field_depth_loss(
    weights, # TensorType[..., "num_samples", 1],
    termination_depth, # TensorType[..., 1],
    predicted_depth, # TensorType[..., 1],
    steps, # TensorType[..., "num_samples", 1],
    sigma, # TensorType[0],
):
    """Lidar losses from Urban Radiance Fields (Rematas et al., 2022).
    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        steps: Sampling distances along rays.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    depth_mask = termination_depth > 0

    # Expected depth loss
    expected_depth_loss = (termination_depth - predicted_depth) ** 2

    # Line of sight losses
    # target_distribution = torch.distributions.normal.Normal(0.0, sigma / URF_SIGMA_SCALE_FACTOR)
    normal_log_prob = lambda value, loc=0, scale=sigma / URF_SIGMA_SCALE_FACTOR: -((value - loc) ** 2) / (2 * (scale**2)) - jnp.log(scale) - jnp.log(jnp.sqrt(2 * jnp.pi))
    termination_depth = termination_depth[:, None]
    line_of_sight_loss_near_mask = jnp.logical_and(
        steps <= termination_depth + sigma, steps >= termination_depth - sigma
    )
    line_of_sight_loss_near = (weights - jnp.exp(normal_log_prob(steps - termination_depth))) ** 2
    line_of_sight_loss_near = (line_of_sight_loss_near_mask * line_of_sight_loss_near).sum(-2)
    line_of_sight_loss_empty_mask = steps < termination_depth - sigma
    line_of_sight_loss_empty = (line_of_sight_loss_empty_mask * weights**2).sum(-2)
    line_of_sight_loss = line_of_sight_loss_near + line_of_sight_loss_empty

    loss = (expected_depth_loss + line_of_sight_loss) * depth_mask
    return jnp.mean(loss)

def depth_loss(
    weights,
    tdist, # ray_samples,
    termination_depth,
    predicted_depth,
    sigma,
    dirs,
    # is_euclidean,
    depth_loss_type,
):
    """Implementation of depth losses.
    Args:
        weights: Weights predicted for each sample.
        ray_samples: Samples along rays corresponding to weights.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        sigma: Uncertainty around depth value.
        directions_norm: Norms of ray direction vectors in the camera frame.
        is_euclidean: Whether ground truth depths corresponds to normalized direction vectors.
        depth_loss_type: Type of depth loss to apply.
    Returns:
        Depth loss scalar.
    """
    # if not is_euclidean:
    #     termination_depth = termination_depth * directions_norm
    # steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    steps = 0.5 * (tdist[..., :-1] + tdist[..., 1:])

    if depth_loss_type == 'kl':
        # lengths = ray_samples.frustums.ends - ray_samples.frustums.starts
        lengths = tdist[..., 1:] - tdist[..., :-1]
        lengths = lengths * jnp.linalg.norm(dirs[..., None, :], axis=-1)
        return ds_nerf_depth_loss(weights, termination_depth, steps, lengths, sigma)

    if depth_loss_type == 'urf':
        return urban_radiance_field_depth_loss(weights, termination_depth, predicted_depth, steps, sigma)

    raise NotImplementedError("Provided depth loss type not implemented.")