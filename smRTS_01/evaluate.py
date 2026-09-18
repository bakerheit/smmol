"""Phase 2 evaluation: a model as an evaluation arm, scored on world.py's paired episodes.

Everything here runs with learning frozen. `world.run_plan` already refuses an evaluation that
mutated parameters; this module additionally fingerprints the model around the batched forward pass
itself, because the arms handed to `run_plan` are lookups into a precomputed table.

The batching is only an optimisation and only valid under the RESET protocol, where episodes are
independent by construction. Anything that carries state runs one episode at a time.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

import world
from cells import RTSModel, decay_half_life


CHUNK = 250


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def answer_logprobs(model: torch.nn.Module, batch: Sequence[world.Episode],
                    chunk: int = CHUNK) -> List[torch.Tensor]:
    """One 256-vector of log-probabilities per episode, read at the answer position.

    State is zeroed per chunk row, so this is exactly the RESET protocol. Padding sits strictly
    after the scored position and a recurrent state only flows forward, so it cannot leak in.
    """
    device = _device_of(model)
    was_training = model.training
    model.eval()
    out: List[torch.Tensor] = []
    try:
        for start in range(0, len(batch), chunk):
            piece = batch[start:start + chunk]
            inputs, _, mask, _ = world.batch_tensors(piece)
            logits, _ = model(inputs.to(device))
            picked = logits[mask.to(device)]
            out.extend(F.log_softmax(picked.float(), dim=-1).cpu().unbind(0))
    finally:
        model.train(was_training)
    if len(out) != len(batch):
        raise RuntimeError("scored %d rows for %d episodes" % (len(out), len(batch)))
    return out


@torch.no_grad()
def answer_logprobs_sequential(model: torch.nn.Module, batch: Sequence[world.Episode],
                               resets: Sequence[bool]) -> List[torch.Tensor]:
    """The carry-capable path: one episode at a time, state reset only where the protocol says."""
    device = _device_of(model)
    was_training = model.training
    model.eval()
    states = model.zero_state(1, device=device)
    out: List[torch.Tensor] = []
    try:
        for episode, reset in zip(batch, resets):
            if reset:
                states = model.zero_state(1, device=device)
            stream = torch.tensor([list(episode.data[:-1])], dtype=torch.long, device=device)
            logits, states = model(stream, states)
            states = [s.detach() for s in states]
            out.append(F.log_softmax(logits[0, episode.answer_index - 1].float(), dim=-1).cpu())
    finally:
        model.train(was_training)
    return out


def table_arm(logprobs: Sequence[torch.Tensor]) -> Callable:
    """Wrap precomputed rows as a `run_plan` arm. Order is the plan's episode order."""
    counter = {"i": 0}

    def arm(episode: world.Episode, reset: bool):
        row = logprobs[counter["i"]]
        counter["i"] += 1
        return row

    return arm


def evaluate_plan(model: torch.nn.Module, plan: world.EvalPlan, *,
                  with_lazy: bool = True, model_name: str = "model") -> Dict[str, Dict[str, object]]:
    """One (pairs, gap) cell: the model and every lazy arm on one identical episode set."""
    batch = plan.episodes()
    before = world.parameter_fingerprint(model)
    if plan.protocol.name == world.RESET:
        rows = answer_logprobs(model, batch)
    else:
        rows = answer_logprobs_sequential(model, batch, plan.protocol.resets(len(batch), plan.seed))
    after = world.parameter_fingerprint(model)
    if before != after:
        raise RuntimeError("the frozen forward pass mutated model parameters")

    arms: Dict[str, Callable] = {model_name: table_arm(rows)}
    if with_lazy:
        arms.update(world.lazy_arms(plan.pairs))
    summaries = world.run_plan(plan, arms, parameter_modules={model_name: model})
    if world.parameter_fingerprint(model) != before:
        raise RuntimeError("evaluation mutated model parameters")
    return summaries


def selection_score(model: torch.nn.Module, pairs: int = 16, gap: int = 128,
                    trials: int = world.SELECT_TRIALS) -> Dict[str, float]:
    """The independent selection metric. Uses SELECT_SEED, never EVAL_SEED."""
    plan = world.EvalPlan(pairs=pairs, gap=gap, trials=trials, seed=world.SELECT_SEED)
    batch = plan.episodes()
    rows = answer_logprobs(model, batch)
    hits = nll = rank = 0.0
    for row, episode in zip(rows, batch):
        hits += float(int(torch.argmax(row)) == episode.answer)
        nll += -float(row[episode.answer])
        rank += 1 + int((row > row[episode.answer]).sum())
    n = len(batch)
    return {"accuracy": hits / n, "nll": nll / n, "rank": rank / n, "trials": n,
            "episode_set_hash": world.episode_set_hash(batch)}


def futility_reference(pairs: int = 16, gap: int = 32,
                       trials: int = world.SELECT_TRIALS) -> Dict[str, float]:
    """The NLL floor the futility rule watches, on the selection episodes.

    predictions.md phrased this as "0.3 nats below the best lazy arm's NLL". A lazy arm emits one
    byte, not a distribution, so its NLL is undefined without a smoothing convention and the
    convention would dominate the number. Two honest distributional floors are used instead:

    `alphabet` - uniform over the 33 value bytes, ln 33. A model at or above this has not even
    learned that the answer is a value byte, which is the weakest possible sign of life. This is
    the one the futility rule uses.

    `in_episode` - uniform over the distinct values actually written in each episode. A model that
    reaches this has learned "the answer is one of the values on screen" but not which one. It is
    reported, never used as a kill threshold.
    """
    import math

    batch = world.episodes(trials, pairs, gap, seed=world.SELECT_SEED)
    total = 0.0
    for episode in batch:
        distinct = len({value for _, value in episode.pairs})
        total += math.log(distinct)
    return {
        "alphabet_nll": math.log(len(world.VALUE_BYTES)),
        "in_episode_nll": total / len(batch),
        "trials": len(batch),
        "pairs": pairs, "gap": gap,
        "rule": "futility fires if selection NLL has not fallen 0.3 nats below alphabet_nll",
    }


def half_life_summary(model: RTSModel) -> Dict[str, object]:
    out = []
    for index, values in enumerate(model.half_lives()):
        flat = values.detach().float().flatten()
        out.append({
            "layer": index,
            "count": int(flat.numel()),
            "min": float(flat.min()),
            "median": float(flat.median()),
            "p90": float(flat.quantile(0.9)),
            "max": float(flat.max()),
        })
    return {"per_layer": out}


def full_grid(model: torch.nn.Module, *, trials: int = world.EVAL_TRIALS,
              seed: int = world.EVAL_SEED,
              cells: Optional[Sequence[tuple]] = None) -> Dict[str, object]:
    """Accuracy / NLL / rank over the pair-count x gap grid, with every lazy arm alongside.

    Gap 8 is flagged position-confounded wherever it appears; gap 512 is flagged extrapolation
    because the Phase 2 training mix is gaps 32 and 128 only.
    """
    grid = cells or [(p, g) for p in world.PAIR_COUNTS for g in world.GAPS]
    rows = {}
    for pairs, gap in grid:
        plan = world.EvalPlan(pairs=pairs, gap=gap, trials=trials, seed=seed)
        summaries = evaluate_plan(model, plan)
        for name, row in summaries.items():
            row.pop("accuracy_by_episodes_since_reset", None)
            row["feasible_positions"] = list(row.get("feasible_positions", ()))
        rows["%d,%d" % (pairs, gap)] = {
            "pairs": pairs,
            "gap": gap,
            "position_confounded": gap == 8,
            "extrapolation": gap == 512,
            "arms": summaries,
        }
    return rows
