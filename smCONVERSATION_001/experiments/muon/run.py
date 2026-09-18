"""Run controlled AdamW versus Muon experiments for smCONVERSATION_001."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parents[1]
REPOSITORY_ROOT = HERE.parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from model import Config, LanguageModel  # noqa: E402
from train import initialize_from_grade_one, resolve_device  # noqa: E402
from training_data import build_sets, encode_supervised, read_school  # noqa: E402
from muon import SingleDeviceMuon  # noqa: E402


def synchronize(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def frozen_conversations(path, count):
    lines = []
    conversations = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            lines.append(line)
            conversations.append(json.loads(line))
            if len(conversations) == count:
                break
    if len(conversations) != count:
        raise ValueError("wanted %d conversations but found %d" % (count, len(conversations)))
    return conversations, hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def encode_all(examples, context):
    encoded = [encode_supervised(example, context) for example in examples]
    return [
        (torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long))
        for inputs, targets in encoded
    ]


def batch_from_indexes(encoded, indexes, device):
    x = torch.stack([encoded[index][0] for index in indexes]).to(device)
    y = torch.stack([encoded[index][1] for index in indexes]).to(device)
    return x, y


def full_evaluation(model, encoded, batch_size, device):
    model.eval()
    loss_sum = 0.0
    target_count = 0
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            indexes = list(range(start, min(start + batch_size, len(encoded))))
            x, y = batch_from_indexes(encoded, indexes, device)
            logits, _ = model(x)
            loss_sum += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"
            ).item()
            target_count += int((y != -100).sum().item())
    model.train()
    return loss_sum / target_count, target_count


def parameter_groups(model):
    muon_parameters = []
    auxiliary_decay = []
    auxiliary_no_decay = []
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    block_ids = {id(parameter) for parameter in model.blocks.parameters()}
    for parameter in model.parameters():
        if id(parameter) in block_ids and parameter.ndim == 2:
            muon_parameters.append(parameter)
        elif parameter.ndim >= 2:
            auxiliary_decay.append(parameter)
        else:
            auxiliary_no_decay.append(parameter)
    all_parameters = muon_parameters + auxiliary_decay + auxiliary_no_decay
    if len({id(parameter) for parameter in all_parameters}) != len(list(model.parameters())):
        raise RuntimeError("optimizer groups do not cover each parameter exactly once")
    return {
        "muon": muon_parameters,
        "aux_decay": auxiliary_decay,
        "aux_no_decay": auxiliary_no_decay,
        "muon_names": [names[id(parameter)] for parameter in muon_parameters],
        "aux_decay_names": [names[id(parameter)] for parameter in auxiliary_decay],
        "aux_no_decay_names": [names[id(parameter)] for parameter in auxiliary_no_decay],
    }


def make_optimizers(model, variant, common):
    groups = parameter_groups(model)
    adam_lr = common["adamw_learning_rate"]
    weight_decay = common["weight_decay"]
    if variant["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            [
                {"params": groups["muon"] + groups["aux_decay"], "weight_decay": weight_decay},
                {"params": groups["aux_no_decay"], "weight_decay": 0.0},
            ],
            lr=adam_lr,
            betas=(0.9, 0.95),
        )
        return [optimizer], groups
    if variant["optimizer"] != "hybrid_muon":
        raise ValueError("unknown optimizer %r" % variant["optimizer"])
    muon_lr = variant["muon_learning_rate"]
    muon_decay = weight_decay
    if variant.get("match_adamw_decay_per_step"):
        muon_decay = weight_decay * adam_lr / muon_lr
    muon = SingleDeviceMuon(
        groups["muon"],
        lr=muon_lr,
        weight_decay=muon_decay,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust=variant["muon_adjustment"],
    )
    auxiliary = torch.optim.AdamW(
        [
            {"params": groups["aux_decay"], "weight_decay": weight_decay},
            {"params": groups["aux_no_decay"], "weight_decay": 0.0},
        ],
        lr=adam_lr,
        betas=(0.9, 0.95),
    )
    return [muon, auxiliary], groups


def learning_rate_multiplier(step, steps, warmup):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def set_learning_rates(optimizers, multiplier):
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"]
            group["lr"] = group["initial_lr"] * multiplier


def run_one(name, variant, seed, settings, model_config, base, encoded_train, encoded_validation, device):
    torch.manual_seed(seed)
    model = LanguageModel(model_config)
    initialize_from_grade_one(model, base)
    model.to(device)
    optimizers, groups = make_optimizers(model, variant, settings)
    batch_generator = torch.Generator().manual_seed(seed + 700000)
    schedule = torch.randint(
        len(encoded_train),
        (settings["steps"], settings["batch_size"]),
        generator=batch_generator,
    ).tolist()
    eval_steps = set(settings["eval_steps"])
    records = []
    initial_loss, validation_targets = full_evaluation(
        model, encoded_validation, settings["batch_size"], device
    )
    records.append({"step": 0, "validation_loss": initial_loss, "bits_per_byte": initial_loss / math.log(2)})
    synchronize(device)
    training_started = time.perf_counter()
    clipped_steps = 0
    gradient_norm_sum = 0.0
    last_loss = None
    for step, indexes in enumerate(schedule, 1):
        multiplier = learning_rate_multiplier(step - 1, settings["steps"], settings["warmup_steps"])
        set_learning_rates(optimizers, multiplier)
        x, y = batch_from_indexes(encoded_train, indexes, device)
        _, loss = model(x, y)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
        gradient_norm_sum += gradient_norm
        clipped_steps += int(gradient_norm > settings["gradient_clip"])
        for optimizer in optimizers:
            optimizer.step()
        last_loss = float(loss.item())
        if step in eval_steps:
            synchronize(device)
            elapsed_before_eval = time.perf_counter() - training_started
            validation_loss, _ = full_evaluation(
                model, encoded_validation, settings["batch_size"], device
            )
            synchronize(device)
            records.append({
                "step": step,
                "train_loss": last_loss,
                "validation_loss": validation_loss,
                "bits_per_byte": validation_loss / math.log(2),
                "training_seconds_before_eval": elapsed_before_eval,
            })
    synchronize(device)
    training_seconds = time.perf_counter() - training_started
    best = min(records, key=lambda item: item["validation_loss"])
    return {
        "variant": name,
        "seed": seed,
        "records": records,
        "best_step": best["step"],
        "best_validation_loss": best["validation_loss"],
        "best_bits_per_byte": best["bits_per_byte"],
        "final_train_loss": last_loss,
        "training_seconds_including_eval": training_seconds,
        "steps_per_second_including_eval": settings["steps"] / training_seconds,
        "mean_gradient_norm_before_clip": gradient_norm_sum / settings["steps"],
        "clipped_step_fraction": clipped_steps / settings["steps"],
        "validation_target_bytes": validation_targets,
        "parameter_groups": {
            "muon": groups["muon_names"],
            "auxiliary_decay": groups["aux_decay_names"],
            "auxiliary_no_decay": groups["aux_no_decay_names"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "full.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("output already exists: %s" % args.output)

    settings = json.loads(args.config.read_text())
    if args.steps is not None:
        settings["steps"] = args.steps
        settings["warmup_steps"] = min(settings["warmup_steps"], max(1, args.steps // 10))
        settings["eval_steps"] = sorted(set(
            [0, args.steps] + [step for step in settings["eval_steps"] if step <= args.steps]
        ))
    if args.gradient_clip is not None:
        if args.gradient_clip <= 0:
            parser.error("--gradient-clip must be positive")
        settings["gradient_clip"] = args.gradient_clip
    if args.seeds:
        settings["seeds"] = args.seeds
    variant_names = args.variants or list(settings["variants"])
    unknown = set(variant_names) - set(settings["variants"])
    if unknown:
        parser.error("unknown variants: %s" % ", ".join(sorted(unknown)))

    model_config = Config.from_file(MODEL_ROOT / "model_config.json")
    conversations, corpus_hash = frozen_conversations(
        MODEL_ROOT / "data" / "accepted" / "conversations.jsonl",
        settings["corpus_conversations"],
    )
    school = read_school(REPOSITORY_ROOT / "smLANGUAGE_en_SCH_001" / "data" / "curriculum.jsonl")
    train_examples, validation_examples, data_meta = build_sets(
        conversations,
        school,
        model_config.context_length,
        16001,
        0.25,
        6,
    )
    encoded_train = encode_all(train_examples, model_config.context_length)
    encoded_validation = encode_all(validation_examples, model_config.context_length)
    base_path = REPOSITORY_ROOT / "smLANGUAGE_en_SCH_001" / "checkpoints" / "grade_01" / "best.pt"
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")

    result = {
        "format_version": 1,
        "experiment": "smCONVERSATION_001-muon",
        "created_unix_seconds": time.time(),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
        },
        "settings": settings,
        "selected_variants": variant_names,
        "corpus_sha256": corpus_hash,
        "data": data_meta,
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in settings["seeds"]:
        for variant_name in variant_names:
            print("run | %s | seed %d" % (variant_name, seed), flush=True)
            run = run_one(
                variant_name,
                settings["variants"][variant_name],
                seed,
                settings,
                model_config,
                base,
                encoded_train,
                encoded_validation,
                device,
            )
            result["runs"].append(run)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(
                "done | %s | seed %d | best val %.5f | %.2f steps/s" % (
                    variant_name,
                    seed,
                    run["best_validation_loss"],
                    run["steps_per_second_including_eval"],
                ),
                flush=True,
            )
    print("wrote %s" % args.output, flush=True)


if __name__ == "__main__":
    main()
