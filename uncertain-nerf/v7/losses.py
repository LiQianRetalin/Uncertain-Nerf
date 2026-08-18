import torch

from v5.losses import DepthScaleAligner


def uncertainty_variance(uncertainty, variance_min=1.0e-4, variance_max=0.05):
    return float(variance_min) + (float(variance_max) - float(variance_min)) * uncertainty


def uncertainty_color_nll(predicted_rgb, target_rgb, ray_uncertainty,
                          variance_min=1.0e-4, variance_max=0.05):
    """Calibration-only NLL: gradients are restricted to uncertainty parameters."""
    squared_error = (predicted_rgb.detach() - target_rgb).square().mean(dim=-1)
    variance = uncertainty_variance(ray_uncertainty, variance_min, variance_max)
    return (0.5 * squared_error / variance + 0.5 * variance.log()).mean()


def termination_distribution_loss(weights, terminal_transmittance, z_vals, prior_depth,
                                  mask, scale, relative_sigma=0.01,
                                  interval_sigma=1.0):
    """Supervise the full ray-termination PMF instead of only expected depth."""
    if not mask.any():
        zero = weights.new_zeros(())
        return zero, {"count": 0, "sigma": zero}
    selected_z = z_vals[mask]
    target_depth = float(scale) * prior_depth[mask]
    if selected_z.shape[-1] > 1:
        interval = (selected_z[:, 1:] - selected_z[:, :-1]).abs().median(dim=-1).values
    else:
        interval = torch.full_like(target_depth, 1.0e-3)
    sigma = torch.maximum(target_depth.abs() * float(relative_sigma),
                          interval * float(interval_sigma)).clamp_min(1.0e-4)
    logits = -0.5 * ((selected_z - target_depth[:, None]) / sigma[:, None]).square()
    target_samples = torch.softmax(logits, dim=-1)
    target = torch.cat([target_samples, torch.zeros_like(target_samples[:, :1])], dim=-1)
    predicted = torch.cat([weights[mask], terminal_transmittance[mask, None]], dim=-1)
    predicted = predicted / predicted.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    loss = -(target * predicted.clamp_min(1.0e-8).log()).sum(dim=-1)
    return loss.mean(), {"count": int(mask.sum()), "sigma": sigma.mean().detach()}


__all__ = ["DepthScaleAligner", "termination_distribution_loss",
           "uncertainty_color_nll", "uncertainty_variance"]
