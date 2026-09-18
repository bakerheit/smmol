"""Train one cumulative school stage and keep both resumable and open inference weights.

`--precision {fp32,bf16,fp16}` autocasts the forward pass and loss (default bf16 on mps/cuda, else
fp32); validation always runs in fp32 so the reported number stays comparable. `--patience N` stops
training after N evals with no validation improvement (0 disables it). `--precision fp32 --patience 0`
reproduces the previous behaviour exactly.
"""

import argparse
import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
import shutil
import time

import torch

from model import Config, LanguageModel
from safetensors_io import save_file
from training_data import build_splits, latest_stage, read_items


HERE = Path(__file__).resolve().parent
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_config_document(path):
    document = json.loads(Path(path).read_text())
    model_config = Config.from_dict(document)
    tokenizer = document.get("tokenizer") or {}
    if tokenizer.get("kind") != "utf8_bytes" or tokenizer.get("vocab_size") != model_config.vocab_size:
        raise ValueError("tokenizer must be utf8_bytes with the same vocab_size as the model")
    return document, model_config


def sample_batch(source, batch_size, context, generator, device):
    if len(source) <= context + 1:
        raise ValueError("split has %d bytes but context_length is %d" % (len(source), context))
    starts = torch.randint(len(source) - context - 1, (batch_size,), generator=generator)
    offsets = torch.arange(context)
    x = source[starts[:, None] + offsets].long()
    y = source[starts[:, None] + offsets + 1].long()
    return x.to(device), y.to(device)


def checkpoint(model, optimizer, config_document, stage, data_meta, corpus_sha256, step, best_validation, generator):
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config_document,
        "stage": stage,
        "data": data_meta,
        "corpus_sha256": corpus_sha256,
        "step": step,
        "best_validation": best_validation,
        "batch_generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }


def write_bundle(payload, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "smLANGUAGE_en_SCH_001",
        "stage": payload["stage"],
        "items": payload["data"]["items"],
        "corpus_sha256": payload["corpus_sha256"],
        "step": payload["step"],
        "format": "safetensors",
    }
    save_file(payload["model"], folder / "weights.safetensors", metadata)
    write_json(payload["config"], folder / "model_config.json")
    write_json(metadata, folder / "metadata.json")


def promote_bundle(stage_folder, out_folder, checkpoint_path):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (Path(stage_folder) / "weights.safetensors", "latest.safetensors"),
        (Path(stage_folder) / "model_config.json", "model_config.json"),
        (Path(stage_folder) / "metadata.json", "metadata.json"),
        (Path(checkpoint_path), "latest.pt"),
    ):
        temporary = out_folder / (name + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, out_folder / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "model_config.json")
    parser.add_argument("--corpus", type=Path, default=HERE / "data" / "curriculum.jsonl")
    parser.add_argument("--stage", help="last accepted school level to include; default is latest in compiled corpus")
    parser.add_argument("--out", type=Path, help="stage checkpoint folder; default checkpoints/<stage>")
    parser.add_argument("--promote-to", type=Path, default=HERE / "out", help="stable bundle used by the harness")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--steps", type=int, help="total step target for this stage")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init", type=Path, help="initialize weights from another full .pt checkpoint")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--precision", choices=sorted(DTYPES), default=None,
                         help="autocast dtype for the forward pass; defaults to bf16 on mps, else fp32")
    parser.add_argument("--patience", type=int, default=3,
                         help="stop after this many evals with no validation improvement; 0 disables")
    args = parser.parse_args()
    if args.resume and args.init:
        parser.error("use either --resume or --init, not both")

    config_document, config = load_config_document(args.config)
    training = config_document.get("training") or {}
    items = read_items(args.corpus)
    stage = args.stage or latest_stage(items)
    seed = int(training.get("seed", 13001))
    train_bytes, validation_bytes, data_meta = build_splits(items, stage, seed)
    train_data = torch.frombuffer(bytearray(train_bytes), dtype=torch.uint8)
    validation_data = torch.frombuffer(bytearray(validation_bytes), dtype=torch.uint8)
    corpus_sha256 = file_hash(args.corpus)

    target_steps = args.steps if args.steps is not None else int(training.get("steps_per_stage", 1200))
    batch_size = args.batch if args.batch is not None else int(training.get("batch_size", 32))
    learning_rate = args.lr if args.lr is not None else float(training.get("learning_rate", 6e-4))
    warmup = int(training.get("warmup_steps", 60))
    eval_every = args.eval_every if args.eval_every is not None else int(training.get("eval_every", 100))
    eval_batches = int(training.get("eval_batches", 12))
    weight_decay = float(training.get("weight_decay", 0.1))
    if min(target_steps, batch_size, eval_every, eval_batches) < 1 or learning_rate <= 0:
        parser.error("steps, batch, eval settings and learning rate must be positive")

    stage_folder = args.out or HERE / "checkpoints" / stage
    latest_path = stage_folder / "latest.pt"
    best_path = stage_folder / "best.pt"
    resume = None
    if args.resume:
        if not latest_path.exists():
            parser.error("cannot resume because %s does not exist" % latest_path)
        resume = torch.load(latest_path, map_location="cpu", weights_only=False)
        if Config.from_dict(resume["config"]) != config:
            parser.error("checkpoint architecture does not match model_config.json")
        if resume.get("stage") != stage or resume.get("corpus_sha256") != corpus_sha256:
            parser.error("checkpoint stage or curriculum hash does not match this run")

    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if args.precision is None:
        args.precision = "bf16" if device == "mps" else "fp32"
    dtype = DTYPES[args.precision]
    if args.precision == "fp16":
        print("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.", flush=True)
    amp = (lambda: torch.autocast(device_type=device, dtype=dtype)) if dtype is not None else nullcontext
    model = LanguageModel(config)
    if resume:
        model.load_state_dict(resume["model"])
    elif args.init:
        initialized = torch.load(args.init, map_location="cpu", weights_only=False)
        if Config.from_dict(initialized["config"]) != config:
            parser.error("initial checkpoint architecture does not match model_config.json")
        model.load_state_dict(initialized["model"])
    model.to(device)

    decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    step = int(resume["step"]) if resume else 0
    best_validation = float(resume["best_validation"]) if resume else float("inf")
    if step >= target_steps:
        parser.error("checkpoint is already at step %d; target is %d" % (step, target_steps))
    generator = torch.Generator()
    if resume:
        generator.set_state(resume["batch_generator_state"])
        torch.set_rng_state(resume["torch_rng_state"])
    else:
        generator.manual_seed(seed)

    @torch.no_grad()
    def evaluate():
        model.eval()
        fixed = torch.Generator().manual_seed(seed + 100000)
        losses = []
        for _ in range(eval_batches):
            x, y = sample_batch(validation_data, batch_size, config.context_length, fixed, device)
            losses.append(model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

    def rate(current):
        if current < warmup:
            return learning_rate * (current + 1) / max(warmup, 1)
        progress = (current - warmup) / max(target_steps - warmup, 1)
        return learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    stage_folder.mkdir(parents=True, exist_ok=True)
    log_path = stage_folder / "train.jsonl"
    started = time.time()
    print(
        "smLANGUAGE_en_SCH_001 | %s | %.2fM params | %s | %d items | steps %d->%d | precision %s | patience %d"
        % (stage, model.parameter_count() / 1e6, device, data_meta["items"], step, target_steps,
           args.precision, args.patience),
        flush=True,
    )
    model.train()
    stall = 0
    while step < target_steps:
        lr = rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = sample_batch(train_data, batch_size, config.context_length, generator, device)
        with amp():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step != 1 and step % eval_every and step != target_steps:
            continue
        # validation is always plain fp32 (no autocast), so the number stays comparable across runs
        validation_loss = evaluate()
        improved = validation_loss < best_validation
        if improved:
            best_validation = validation_loss
            stall = 0
        else:
            stall += 1
        event = {
            "stage": stage,
            "step": step,
            "train_loss": round(loss.item(), 6),
            "validation_loss": round(validation_loss, 6),
            "validation_bits_per_byte": round(validation_loss / math.log(2), 6),
            "learning_rate": lr,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        payload = checkpoint(
            model, optimizer, config_document, stage, data_meta, corpus_sha256, step, best_validation, generator
        )
        atomic_torch_save(payload, latest_path)
        if improved:
            atomic_torch_save(payload, best_path)
        print(
            "step %4d | train %.3f | val %.3f (%.2f bpb)%s"
            % (step, loss.item(), validation_loss, validation_loss / math.log(2), " | best" if improved else ""),
            flush=True,
        )
        if args.patience and stall >= args.patience:
            print(
                "early stop | no validation improvement for %d evals | stopped at step %d | best val %.4f"
                % (args.patience, step, best_validation),
                flush=True,
            )
            break

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    write_bundle(best, stage_folder)
    if not args.no_promote:
        promote_bundle(stage_folder, args.promote_to, best_path)
    print("done | best validation %.4f | %s" % (best_validation, stage_folder / "weights.safetensors"), flush=True)


if __name__ == "__main__":
    main()
