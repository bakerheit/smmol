"""TBPTT/reference loss helpers for smRTS_01.

Phase 0 uses a full unroll of a tiny sequence as the gradient oracle. Later phases call `train_window` and detach
between windows. Loss is a sum over time and a mean over streams, matching online.py one byte at a time.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from cells import RTSModel


def detach_states(states: Sequence[Tensor]) -> List[Tensor]:
    return [state.detach() for state in states]


def sequence_loss(model: RTSModel, inputs: Tensor, targets: Tensor,
                  states: Sequence[Tensor] | None = None) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """Return summed-per-time/mean-per-stream loss, logits, and final recurrent state."""
    if inputs.shape != targets.shape or inputs.ndim != 2:
        raise ValueError("inputs and targets must both have shape [batch, time]")
    logits, final_states = model(inputs, states)
    loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="sum")
    loss = loss / inputs.shape[0]
    return loss, logits, final_states


def sequence_gradients(model: RTSModel, inputs: Tensor, targets: Tensor) -> Tuple[float, Tensor]:
    model.zero_grad(set_to_none=True)
    loss, logits, _ = sequence_loss(model, inputs, targets)
    loss.backward()
    return float(loss.detach()), logits.detach()


def train_window(model: RTSModel, optimizer: torch.optim.Optimizer, inputs: Tensor, targets: Tensor,
                 states: Sequence[Tensor] | None = None) -> Tuple[float, List[Tensor]]:
    optimizer.zero_grad(set_to_none=True)
    loss, _, final_states = sequence_loss(model, inputs, targets, states)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), detach_states(final_states)

