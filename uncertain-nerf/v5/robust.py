import math

import torch


def mad_scale(values, multiplier=1.4826, eps=1.0e-8, mask=None):
    if mask is not None:
        values = values[mask]
    values = values.reshape(-1)
    if values.numel() == 0:
        return torch.as_tensor(eps, device=values.device, dtype=values.dtype)
    median = values.median()
    mad = (values - median).abs().median()
    return (multiplier * mad).clamp_min(eps)


def tukey_biweight(residual, cutoff):
    cutoff = torch.as_tensor(cutoff, device=residual.device, dtype=residual.dtype)
    ratio = residual / cutoff.clamp_min(1.0e-8)
    inlier = ratio.abs() <= 1.0
    bounded = (cutoff**2 / 6.0) * (1.0 - (1.0 - ratio**2) ** 3)
    ceiling = cutoff**2 / 6.0
    return torch.where(inlier, bounded, ceiling)


def pseudo_huber(residual, delta):
    delta = torch.as_tensor(delta, device=residual.device, dtype=residual.dtype)
    delta = delta.clamp_min(1.0e-8)
    return delta**2 * (torch.sqrt(1.0 + (residual / delta) ** 2) - 1.0)


def anneal(start, end, step, duration):
    if duration <= 0:
        return float(end)
    t = min(max(step / duration, 0.0), 1.0)
    return float(start + (end - start) * t)


def warmup_cosine_lr(step, total_steps, warmup=1000, start=1.0e-2, end=1.0e-4):
    if step <= warmup:
        return start * max(step, 1) / max(warmup, 1)
    progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


class LossCalibrator:
    def __init__(self, rho=0.02, calibration_steps=500, eps=1.0e-8):
        self.rho = float(rho)
        self.calibration_steps = int(calibration_steps)
        self.eps = float(eps)
        self.ema = {}
        self.weights = None

    def update(self, step, losses, teacher_ratio, alphas=None):
        if self.weights is not None:
            return self.weights
        for name, value in losses.items():
            scalar = float(value.detach().cpu())
            if name not in self.ema:
                self.ema[name] = scalar
            else:
                self.ema[name] = (1.0 - self.rho) * self.ema[name] + self.rho * scalar
        if step >= self.calibration_steps:
            alphas = alphas or {"geo": 0.5, "u": 0.5, "reg": 0.1}
            color = self.ema.get("color", 1.0)
            self.weights = {
                name: alphas[name] * color / (self.ema.get(name, 0.0) + self.eps)
                for name in ("geo", "u", "reg")
            }
            self.weights["u"] /= max(float(teacher_ratio), self.eps)
        return self.weights or {"geo": 0.5, "u": 0.5, "reg": 0.1}

    def state_dict(self):
        return {"ema": self.ema, "weights": self.weights}

    def load_state_dict(self, state):
        self.ema = dict(state.get("ema", {}))
        self.weights = state.get("weights")
