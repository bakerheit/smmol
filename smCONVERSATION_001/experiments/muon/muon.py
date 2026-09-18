"""Small single-device Muon implementation for the controlled optimizer experiment.

The update follows Keller Jordan's MIT-licensed reference implementation and exposes both its
original rectangular-matrix scaling and Moonlight/PyTorch's AdamW-RMS-matching scale.
"""

import math

import torch


def zeropower_newton_schulz(gradient, steps=5, eps=1e-7):
    """Approximate the orthogonal factor of a two-dimensional matrix."""
    if gradient.ndim != 2:
        raise ValueError("Muon only supports two-dimensional parameters")
    if not 1 <= steps < 100:
        raise ValueError("Newton-Schulz steps must be between 1 and 99")
    a, b, c = 3.4445, -4.7750, 2.0315
    update = gradient.to(dtype=torch.bfloat16, copy=True)
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.T
    update.div_(update.norm().clamp(min=eps))
    for _ in range(steps):
        gram = update @ update.T
        polynomial = b * gram + c * (gram @ gram)
        update = a * update + polynomial @ update
    if transposed:
        update = update.T
    return update.to(dtype=gradient.dtype)


def adjustment(shape, mode):
    """Return the published shape-dependent multiplier for a matrix update."""
    rows, columns = shape
    if mode == "original":
        return math.sqrt(max(1.0, rows / columns))
    if mode == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, columns))
    raise ValueError("unknown Muon adjustment %r" % mode)


class SingleDeviceMuon(torch.optim.Optimizer):
    """Nesterov Muon for hidden matrices on one device."""

    def __init__(
        self,
        params,
        lr,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust="original",
    ):
        params = list(params)
        if not params:
            raise ValueError("Muon needs at least one parameter")
        if any(parameter.ndim != 2 for parameter in params):
            raise ValueError("every Muon parameter must be two-dimensional")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust": adjust,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum = state["momentum_buffer"]
                momentum.lerp_(parameter.grad, 1.0 - group["momentum"])
                if group["nesterov"]:
                    raw_update = parameter.grad.lerp(momentum, group["momentum"])
                else:
                    raw_update = momentum
                update = zeropower_newton_schulz(raw_update, group["ns_steps"])
                scaled_lr = group["lr"] * adjustment(parameter.shape, group["adjust"])
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-scaled_lr)
        return loss
