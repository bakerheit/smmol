"""Recurrent cells for smRTS_01 Phase 0.

The model owns parameters; callers own recurrent state.  Keeping state out of the modules makes resets explicit,
lets TBPTT choose where to detach, and lets the online trainer prove it retained no graph.

`leaky`, `gated`, and additive `fast` have closed-form one-layer online traces in online.py.  `delta` and `gru` are
TBPTT controls only: DeltaNet's state transition is not diagonal, and a GRU has a dense recurrent Jacobian.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ONLINE_CELLS = ("leaky", "gated", "fast")
CONTROL_CELLS = ("delta", "gru")
CELL_TYPES = ONLINE_CELLS + CONTROL_CELLS


def half_life_to_logit(half_life: Tensor) -> Tensor:
    """Decay logits whose sigmoid has the requested half-life in recurrent steps."""
    decay = torch.pow(torch.as_tensor(2.0, dtype=half_life.dtype, device=half_life.device),
                      -1.0 / half_life)
    return torch.logit(decay)


def spread_decay_logits(count: int, minimum: float = 1.0, maximum: float = 1024.0,
                        *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Deterministic log-spaced half-lives, so long traces exist before learning starts."""
    if count < 1 or minimum <= 0 or maximum < minimum:
        raise ValueError("bad half-life spread")
    half_lives = torch.logspace(math.log2(minimum), math.log2(maximum), count,
                                base=2.0, dtype=dtype)
    return half_life_to_logit(half_lives)


def decay_half_life(decay_logit: Tensor) -> Tensor:
    decay = decay_logit.sigmoid()
    return math.log(0.5) / torch.log(decay)


def _skip(input_dim: int, dim: int) -> nn.Module:
    return nn.Identity() if input_dim == dim else nn.Linear(input_dim, dim, bias=False)


class LeakyCell(nn.Module):
    """The source-repo leaky integrator. Input and state widths are identical."""

    online_trace = True

    def __init__(self, dim: int):
        super().__init__()
        self.input_dim = self.dim = dim
        self.decay_logit = nn.Parameter(spread_decay_logits(dim))
        self.norm = nn.LayerNorm(dim)
        self.read = nn.Linear(dim, dim, bias=False)

    def zero_state(self, batch: int, *, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch, self.dim, device=device, dtype=dtype)

    def write(self, x: Tensor) -> Tensor:
        return x

    def read_state(self, x: Tensor, state: Tensor) -> Tensor:
        return x + F.silu(self.read(self.norm(state)))

    def step(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        new_state = self.decay_logit.sigmoid() * state + self.write(x)
        return self.read_state(x, new_state), new_state


class GatedCell(nn.Module):
    """Row-local gated write: every state row has a parameter-shaped exact trace."""

    online_trace = True

    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.input_dim, self.dim = input_dim, dim
        self.decay_logit = nn.Parameter(spread_decay_logits(dim))
        self.value = nn.Linear(input_dim, dim, bias=False)
        self.gate = nn.Linear(input_dim, dim, bias=False)
        self.skip = _skip(input_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.read = nn.Linear(dim, dim, bias=False)

    def zero_state(self, batch: int, *, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch, self.dim, device=device, dtype=dtype)

    def write_parts(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        value = self.value(x)
        gate = self.gate(x).sigmoid()
        return gate * value, gate, value

    def read_state(self, x: Tensor, state: Tensor) -> Tensor:
        return self.skip(x) + F.silu(self.read(self.norm(state)))

    def step(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        write, _, _ = self.write_parts(x)
        new_state = self.decay_logit.sigmoid() * state + write
        return self.read_state(x, new_state), new_state


class FastCell(nn.Module):
    """Additive outer-product memory with exact scalar-per-head online traces."""

    online_trace = True

    def __init__(self, input_dim: int, dim: int, heads: int = 4, key_dim: int = 8,
                 value_dim: int = 12):
        super().__init__()
        if min(heads, key_dim, value_dim) < 1:
            raise ValueError("fast dimensions must be positive")
        self.input_dim, self.dim = input_dim, dim
        self.heads, self.key_dim, self.value_dim = heads, key_dim, value_dim
        self.key_scale = key_dim ** -0.5
        self.decay_logit = nn.Parameter(spread_decay_logits(heads, 8.0, 512.0))
        self.key = nn.Linear(input_dim, heads * key_dim, bias=False)
        self.value = nn.Linear(input_dim, heads * value_dim, bias=False)
        self.query = nn.Linear(input_dim, heads * key_dim, bias=False)
        self.out_norm = nn.RMSNorm(heads * value_dim)
        self.out = nn.Linear(heads * value_dim, dim, bias=False)
        self.skip = _skip(input_dim, dim)

    def zero_state(self, batch: int, *, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch, self.heads, self.key_dim, self.value_dim,
                           device=device, dtype=dtype)

    def write_parts(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch = x.shape[0]
        key = self.key(x).view(batch, self.heads, self.key_dim) * self.key_scale
        value = self.value(x).view(batch, self.heads, self.value_dim)
        return key, value

    def read_state(self, x: Tensor, state: Tensor) -> Tensor:
        batch = x.shape[0]
        query = self.query(x).view(batch, self.heads, self.key_dim) * self.key_scale
        read = torch.einsum("bhkv,bhk->bhv", state, query).reshape(batch, -1)
        return self.skip(x) + self.out(self.out_norm(read))

    def step(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        key, value = self.write_parts(x)
        decay = self.decay_logit.sigmoid().view(1, self.heads, 1, 1)
        new_state = decay * state + torch.einsum("bhk,bhv->bhkv", key, value)
        return self.read_state(x, new_state), new_state


class DeltaCell(FastCell):
    """Erase-then-write fast memory. Kept out of the diagonal online trace claim."""

    online_trace = False

    def __init__(self, input_dim: int, dim: int, heads: int = 4, key_dim: int = 8,
                 value_dim: int = 12):
        super().__init__(input_dim, dim, heads, key_dim, value_dim)
        self.rate = nn.Linear(input_dim, heads, bias=False)

    def write_parts(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        key, value = super().write_parts(x)
        return F.normalize(key, dim=-1), value

    def read_state(self, x: Tensor, state: Tensor) -> Tensor:
        batch = x.shape[0]
        query = F.normalize(self.query(x).view(batch, self.heads, self.key_dim), dim=-1)
        read = torch.einsum("bhkv,bhk->bhv", state, query).reshape(batch, -1)
        return self.skip(x) + self.out(self.out_norm(read))

    def step(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        key, value = self.write_parts(x)
        predicted = torch.einsum("bhkv,bhk->bhv", state, key)
        rate = self.rate(x).sigmoid().view(x.shape[0], self.heads, 1)
        error = rate * (value - predicted)
        decay = self.decay_logit.sigmoid().view(1, self.heads, 1, 1)
        new_state = decay * state + torch.einsum("bhk,bhv->bhkv", key, error)
        return self.read_state(x, new_state), new_state


class GRUCell(nn.Module):
    """Textbook recurrent control, trained only through TBPTT."""

    online_trace = False

    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.input_dim, self.dim = input_dim, dim
        self.cell = nn.GRUCell(input_dim, dim)

    def zero_state(self, batch: int, *, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch, self.dim, device=device, dtype=dtype)

    def step(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        new_state = self.cell(x, state)
        return new_state, new_state


class RTSModel(nn.Module):
    """A byte model whose recurrent state is always explicit."""

    def __init__(self, cell_type: str, dim: int = 384, layers: int = 1, *, heads: int = 4,
                 key_dim: int = 8, value_dim: int = 12):
        super().__init__()
        if cell_type not in CELL_TYPES:
            raise ValueError("unknown cell %r; choose from %s" % (cell_type, ", ".join(CELL_TYPES)))
        if dim < 1 or layers < 1:
            raise ValueError("dim and layers must be positive")
        self.cell_type, self.dim, self.layers_count = cell_type, dim, layers
        self.heads, self.key_dim, self.value_dim = heads, key_dim, value_dim

        if cell_type == "leaky":
            self.embedding = nn.Embedding(256, dim)
            cells: List[nn.Module] = [LeakyCell(dim) for _ in range(layers)]
        else:
            self.embedding = None
            cells = []
            for index in range(layers):
                input_dim = 256 if index == 0 else dim
                if cell_type == "gated":
                    cell = GatedCell(input_dim, dim)
                elif cell_type == "fast":
                    cell = FastCell(input_dim, dim, heads, key_dim, value_dim)
                elif cell_type == "delta":
                    cell = DeltaCell(input_dim, dim, heads, key_dim, value_dim)
                else:
                    cell = GRUCell(input_dim, dim)
                cells.append(cell)
        self.cells = nn.ModuleList(cells)
        self.decoder = nn.Linear(dim, 256, bias=False)

    @property
    def supports_online(self) -> bool:
        return self.cell_type in ONLINE_CELLS

    def encode(self, byte_ids: Tensor) -> Tensor:
        if byte_ids.dtype != torch.long:
            byte_ids = byte_ids.long()
        if self.embedding is not None:
            return self.embedding(byte_ids)
        return F.one_hot(byte_ids, 256).to(dtype=self.decoder.weight.dtype)

    def zero_state(self, batch: int, *, device=None, dtype=None) -> List[Tensor]:
        if device is None:
            device = self.decoder.weight.device
        if dtype is None:
            dtype = self.decoder.weight.dtype
        return [cell.zero_state(batch, device=device, dtype=dtype) for cell in self.cells]

    def step(self, byte_ids: Tensor, states: Sequence[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        if len(states) != len(self.cells):
            raise ValueError("wanted %d states, got %d" % (len(self.cells), len(states)))
        x = self.encode(byte_ids)
        next_states = []
        for cell, state in zip(self.cells, states):
            x, state = cell.step(x, state)
            next_states.append(state)
        return self.decoder(x), next_states

    def forward(self, byte_ids: Tensor, states: Sequence[Tensor] | None = None
                ) -> Tuple[Tensor, List[Tensor]]:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, time]")
        if states is None:
            states = self.zero_state(byte_ids.shape[0])
        logits = []
        current = list(states)
        for index in range(byte_ids.shape[1]):
            step_logits, current = self.step(byte_ids[:, index], current)
            logits.append(step_logits)
        return torch.stack(logits, dim=1), current

    def set_decay(self, value: float) -> None:
        """Set every diagonal decay to an exact value, mainly for gradient checks."""
        if not 0.0 < value < 1.0:
            raise ValueError("decay must be between zero and one")
        logit = math.log(value / (1.0 - value))
        with torch.no_grad():
            for cell in self.cells:
                if hasattr(cell, "decay_logit"):
                    cell.decay_logit.fill_(logit)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def state_scalar_count(self) -> int:
        return sum(state.numel() for state in self.zero_state(1))

    def half_lives(self) -> List[Tensor]:
        return [decay_half_life(cell.decay_logit.detach()) for cell in self.cells
                if hasattr(cell, "decay_logit")]
