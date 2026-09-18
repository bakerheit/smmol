"""Constant-history online eligibility traces for smRTS_01.

Only `leaky`, `gated`, and additive `fast` are supported. Each recurrent state is rebuilt as a leaf every byte.
Write-path parameters are detached from ordinary autograd and receive their recurrent gradient only from the closed
form traces below. Read-path parameters use ordinary current-step autograd.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from cells import FastCell, GatedCell, LeakyCell, RTSModel


@dataclass
class OnlineState:
    states: List[Tensor]
    traces: List[Dict[str, Tensor]]

    def tensors(self) -> Iterable[Tensor]:
        yield from self.states
        for group in self.traces:
            yield from group.values()


def initialize(model: RTSModel, batch: int) -> OnlineState:
    if not model.supports_online:
        raise ValueError("%s has no proved online trace" % model.cell_type)
    states = model.zero_state(batch)
    traces: List[Dict[str, Tensor]] = []
    for index, (cell, state) in enumerate(zip(model.cells, states)):
        common = {"decay": torch.zeros_like(state)}
        if isinstance(cell, LeakyCell):
            if index == 0:
                common["embedding"] = state.new_zeros(batch, 256, cell.dim)
        elif isinstance(cell, GatedCell):
            common["value"] = state.new_zeros(batch, cell.dim, cell.input_dim)
            common["gate"] = state.new_zeros(batch, cell.dim, cell.input_dim)
        elif isinstance(cell, FastCell):
            common["key"] = state.new_zeros(batch, cell.heads, cell.value_dim, cell.input_dim)
            common["value"] = state.new_zeros(batch, cell.heads, cell.key_dim, cell.input_dim)
        else:  # pragma: no cover - guarded by model.supports_online
            raise TypeError(type(cell))
        traces.append(common)
    return OnlineState(states, traces)


def trace_bytes(state: OnlineState) -> int:
    return sum(tensor.numel() * tensor.element_size()
               for group in state.traces for tensor in group.values())


def graph_is_severed(state: OnlineState) -> bool:
    return all(not tensor.requires_grad and tensor.grad_fn is None for tensor in state.tensors())


def _add_grad(parameter: Tensor, gradient: Tensor) -> None:
    gradient = gradient.detach()
    if parameter.grad is None:
        parameter.grad = gradient.clone()
    else:
        parameter.grad.add_(gradient)


def _detached_linear(x: Tensor, layer: torch.nn.Linear) -> Tensor:
    return F.linear(x, layer.weight.detach(), None)


def _leaky_step(cell: LeakyCell, x: Tensor, previous: Tensor, traces: Dict[str, Tensor],
                bridge_input: bool, byte_ids: Tensor | None) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    with torch.no_grad():
        decay = cell.decay_logit.sigmoid()
        numeric = decay * previous + x.detach()
        traces["decay"].mul_(decay).add_(decay * (1.0 - decay) * previous)
        if "embedding" in traces:
            if byte_ids is None:
                raise ValueError("layer-0 leaky trace needs byte ids")
            traces["embedding"].mul_(decay.view(1, 1, -1))
            rows = torch.arange(x.shape[0], device=x.device)
            # Advanced indexing returns a copy for `.add_`; augmented assignment writes it back.
            traces["embedding"][rows, byte_ids.long(), :] += 1.0
    bridge = x - x.detach() if bridge_input else torch.zeros_like(numeric)
    leaf = numeric.detach().requires_grad_(True)
    return cell.read_state(x, leaf + bridge), leaf, traces


def _gated_step(cell: GatedCell, x: Tensor, previous: Tensor, traces: Dict[str, Tensor],
                bridge_input: bool) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    with torch.no_grad():
        x_value = x.detach()
        value = F.linear(x_value, cell.value.weight)
        gate = F.linear(x_value, cell.gate.weight).sigmoid()
        write = gate * value
        decay = cell.decay_logit.sigmoid()
        numeric = decay * previous + write
        traces["decay"].mul_(decay).add_(decay * (1.0 - decay) * previous)
        traces["value"].mul_(decay.view(1, -1, 1)).add_(
            torch.einsum("bd,bi->bdi", gate, x_value))
        gate_drive = gate * (1.0 - gate) * value
        traces["gate"].mul_(decay.view(1, -1, 1)).add_(
            torch.einsum("bd,bi->bdi", gate_drive, x_value))
    if bridge_input and x.requires_grad:
        live_value = _detached_linear(x, cell.value)
        live_gate = _detached_linear(x, cell.gate).sigmoid()
        live_write = live_gate * live_value
        bridge = live_write - live_write.detach()
    else:
        bridge = torch.zeros_like(numeric)
    leaf = numeric.detach().requires_grad_(True)
    return cell.read_state(x, leaf + bridge), leaf, traces


def _fast_step(cell: FastCell, x: Tensor, previous: Tensor, traces: Dict[str, Tensor],
               bridge_input: bool) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    batch = x.shape[0]
    with torch.no_grad():
        x_value = x.detach()
        key = F.linear(x_value, cell.key.weight).view(batch, cell.heads, cell.key_dim)
        key = key * cell.key_scale
        value = F.linear(x_value, cell.value.weight).view(batch, cell.heads, cell.value_dim)
        decay = cell.decay_logit.sigmoid().view(1, cell.heads, 1, 1)
        numeric = decay * previous + torch.einsum("bhk,bhv->bhkv", key, value)
        traces["decay"].mul_(decay).add_(decay * (1.0 - decay) * previous)
        traces["key"].mul_(decay).add_(
            torch.einsum("bhv,bi->bhvi", value, x_value) * cell.key_scale)
        traces["value"].mul_(decay).add_(torch.einsum("bhk,bi->bhki", key, x_value))
    if bridge_input and x.requires_grad:
        live_key = _detached_linear(x, cell.key).view(batch, cell.heads, cell.key_dim)
        live_key = live_key * cell.key_scale
        live_value = _detached_linear(x, cell.value).view(batch, cell.heads, cell.value_dim)
        live_write = torch.einsum("bhk,bhv->bhkv", live_key, live_value)
        bridge = live_write - live_write.detach()
    else:
        bridge = torch.zeros_like(numeric)
    leaf = numeric.detach().requires_grad_(True)
    return cell.read_state(x, leaf + bridge), leaf, traces


def _apply_trace_gradients(model: RTSModel, state: OnlineState, leaves: Sequence[Tensor]) -> None:
    with torch.no_grad():
        for index, (cell, traces, leaf) in enumerate(zip(model.cells, state.traces, leaves)):
            immediate = leaf.grad
            if immediate is None:
                raise RuntimeError("state leaf received no immediate gradient")
            if isinstance(cell, LeakyCell):
                _add_grad(cell.decay_logit, (immediate * traces["decay"]).sum(dim=0))
                if "embedding" in traces:
                    _add_grad(model.embedding.weight,
                              torch.einsum("bd,bcd->cd", immediate, traces["embedding"]))
            elif isinstance(cell, GatedCell):
                _add_grad(cell.decay_logit, (immediate * traces["decay"]).sum(dim=0))
                _add_grad(cell.value.weight,
                          torch.einsum("bd,bdi->di", immediate, traces["value"]))
                _add_grad(cell.gate.weight,
                          torch.einsum("bd,bdi->di", immediate, traces["gate"]))
            elif isinstance(cell, FastCell):
                _add_grad(cell.decay_logit,
                          (immediate * traces["decay"]).sum(dim=(0, 2, 3)))
                _add_grad(cell.key.weight,
                          torch.einsum("bhkv,bhvi->hki", immediate, traces["key"])
                          .reshape_as(cell.key.weight))
                _add_grad(cell.value.weight,
                          torch.einsum("bhkv,bhki->hvi", immediate, traces["value"])
                          .reshape_as(cell.value.weight))


def step(model: RTSModel, byte_ids: Tensor, targets: Tensor, state: OnlineState,
         optimizer: torch.optim.Optimizer | None = None) -> Tuple[float, Tensor, OnlineState]:
    """Process and optionally optimise one byte column for independent streams."""
    if byte_ids.ndim != 1 or targets.shape != byte_ids.shape:
        raise ValueError("byte_ids and targets must have shape [batch]")
    x = model.encode(byte_ids)
    leaves: List[Tensor] = []
    next_states: List[Tensor] = []
    for index, (cell, previous, traces) in enumerate(zip(model.cells, state.states, state.traces)):
        if isinstance(cell, LeakyCell):
            x, leaf, traces = _leaky_step(cell, x, previous, traces, index > 0,
                                           byte_ids if index == 0 else None)
        elif isinstance(cell, GatedCell):
            x, leaf, traces = _gated_step(cell, x, previous, traces, index > 0)
        elif isinstance(cell, FastCell):
            x, leaf, traces = _fast_step(cell, x, previous, traces, index > 0)
        else:  # pragma: no cover - guarded by initialize
            raise TypeError(type(cell))
        leaves.append(leaf)
        next_states.append(leaf.detach())
        state.traces[index] = traces
    logits = model.decoder(x)
    loss = F.cross_entropy(logits, targets, reduction="mean")
    loss.backward()
    _apply_trace_gradients(model, state, leaves)
    state.states = next_states
    if not graph_is_severed(state):
        raise RuntimeError("online state retained an autograd graph")
    if optimizer is not None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(loss.detach()), logits.detach(), state


def sequence_gradients(model: RTSModel, inputs: Tensor, targets: Tensor
                       ) -> Tuple[float, Tensor, OnlineState]:
    """Accumulate fixed-parameter online gradients over a sequence; the Phase 0 BPTT comparison."""
    if inputs.shape != targets.shape or inputs.ndim != 2:
        raise ValueError("inputs and targets must both have shape [batch, time]")
    model.zero_grad(set_to_none=True)
    state = initialize(model, inputs.shape[0])
    losses, logits = [], []
    for index in range(inputs.shape[1]):
        loss, step_logits, state = step(model, inputs[:, index], targets[:, index], state)
        losses.append(loss)
        logits.append(step_logits)
    return sum(losses), torch.stack(logits, dim=1), state


def _rss_bytes() -> int:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:  # pragma: no cover - psutil ships in the workspace runtime
        return 0


def _live_tensor_count() -> int:
    # `isinstance` walks a few deprecated torch proxy types and emits warnings on PyTorch 2.8.
    return sum(type(item) is Tensor for item in gc.get_objects())


def constant_memory_check(model: RTSModel, steps: int = 10_000, warmup: int = 500) -> Dict[str, float]:
    """Run the Phase 0 graph/RSS smoke. Allocator noise is reported, never hidden."""
    if steps <= warmup:
        raise ValueError("steps must exceed warmup")
    state = initialize(model, 1)
    byte = torch.tensor([17], device=model.decoder.weight.device)
    target = torch.tensor([23], device=byte.device)
    rss_start = tensor_start = None
    started = time.perf_counter()
    for index in range(steps):
        model.zero_grad(set_to_none=True)
        _, _, state = step(model, byte, target, state)
        model.zero_grad(set_to_none=True)
        if index == warmup:
            gc.collect()
            rss_start = _rss_bytes()
            tensor_start = _live_tensor_count()
    gc.collect()
    rss_end = _rss_bytes()
    tensor_end = _live_tensor_count()
    return {
        "steps": steps,
        "seconds": round(time.perf_counter() - started, 3),
        "trace_bytes": trace_bytes(state),
        "rss_growth_bytes": (rss_end - rss_start) if rss_start else 0,
        "tensor_growth": tensor_end - tensor_start,
        "graph_severed": graph_is_severed(state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("leaky", "gated", "fast"), default="leaky")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(7)
    model = RTSModel(args.cell, dim=args.dim, layers=1, heads=2, key_dim=4, value_dim=4)
    result = constant_memory_check(model, args.steps, min(500, args.steps // 4))
    print(result)


if __name__ == "__main__":
    main()
