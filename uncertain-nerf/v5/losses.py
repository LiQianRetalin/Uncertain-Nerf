import torch

from .robust import mad_scale, pseudo_huber


class DepthScaleAligner:
    def __init__(self, warmup=1000, interval=500, ema=0.1, max_observations=65536):
        self.warmup = int(warmup)
        self.interval = int(interval)
        self.ema = float(ema)
        self.max_observations = int(max_observations)
        self.scale = 1.0
        self._ratios = []

    def observe(self, predicted, prior):
        valid = torch.isfinite(predicted) & torch.isfinite(prior) & (predicted > 0) & (prior > 0)
        if valid.any():
            values = (predicted[valid] / prior[valid]).detach().cpu()
            self._ratios.append(values)
            total = sum(len(item) for item in self._ratios)
            while total > self.max_observations and len(self._ratios) > 1:
                total -= len(self._ratios.pop(0))

    def maybe_update(self, step):
        if step < self.warmup or step % self.interval != 0 or not self._ratios:
            return False
        candidate = float(torch.cat(self._ratios).median())
        if candidate > 0 and torch.isfinite(torch.tensor(candidate)):
            self.scale = (1.0 - self.ema) * self.scale + self.ema * candidate
            self._ratios.clear()
            return True
        return False

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state):
        self.scale = float(state.get("scale", 1.0))


def geometry_loss(predicted_depth, student_uncertainty, prior_depth, mask, scale, gamma, huber_multiplier):
    if not mask.any():
        zero = predicted_depth.new_zeros(())
        return zero, {"delta": zero, "count": 0}
    residual = predicted_depth[mask] - float(scale) * prior_depth[mask]
    robust_scale = mad_scale(residual.detach())
    delta = max(float(huber_multiplier), 1.0e-3) * robust_scale
    confidence = torch.exp(-float(gamma) * student_uncertainty[mask])
    loss = (confidence * pseudo_huber(residual, delta)).mean()
    return loss, {"delta": delta.detach(), "count": int(mask.sum())}
