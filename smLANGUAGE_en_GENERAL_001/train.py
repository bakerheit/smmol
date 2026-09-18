"""Fine-tune the first-grade school checkpoint on short conversations.

`--precision` (fp32/bf16/fp16, defaults to bf16 on mps/cuda) autocasts the forward pass and loss;
`--patience` (default 3, 0 disables) stops training when validation loss hasn't improved for that
many evals in a row. `--precision fp32 --patience 0` reproduces the previous behaviour exactly.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from model import Config, LanguageModel
from safetensors_io import save_file
from training_data import build_supervised_sets, read_examples


HERE = Path(__file__).resolve().parent
DEFAULT_INIT = HERE.parent / "smLANGUAGE_en_SCH_001" / "checkpoints" / "grade_01" / "best.pt"
STAGE = "general_conversation_v1"
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sample_batch(source, batch_size, generator, device):
    indexes = torch.randint(len(source), (batch_size,), generator=generator).tolist()
    x = torch.tensor([source[index][0] for index in indexes], dtype=torch.long, device=device)
    y = torch.tensor([source[index][1] for index in indexes], dtype=torch.long, device=device)
    return x, y


def checkpoint(model, optimizer, config, data, corpus_hash, step, best_validation, generator):
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "stage": STAGE,
        "data": data,
        "corpus_sha256": corpus_hash,
        "base_checkpoint": str(DEFAULT_INIT),
        "step": step,
        "best_validation": best_validation,
        "batch_generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }


def write_bundle(payload, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "smLANGUAGE_en_GENERAL_001",
        "stage": payload["stage"],
        "items": payload["data"]["items"],
        "corpus_sha256": payload["corpus_sha256"],
        "step": payload["step"],
        "format": "safetensors",
        "base": "smLANGUAGE_en_SCH_001-grade-01",
        "selection": payload.get("selection", "best"),
    }
    save_file(payload["model"], folder / "weights.safetensors", metadata)
    write_json(payload["config"], folder / "model_config.json")
    write_json(metadata, folder / "metadata.json")


def promote(folder, out, checkpoint_path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (Path(folder) / "weights.safetensors", "latest.safetensors"),
        (Path(folder) / "model_config.json", "model_config.json"),
        (Path(folder) / "metadata.json", "metadata.json"),
        (Path(checkpoint_path), "latest.pt"),
    ):
        temporary = out / (name + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, out / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "model_config.json")
    parser.add_argument("--corpus", type=Path, default=HERE / "data" / "conversations.jsonl")
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--out", type=Path, default=HERE / "checkpoints" / STAGE)
    parser.add_argument("--promote-to", type=Path, default=HERE / "out")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deploy", choices=("best", "latest"), default="latest",
                        help="checkpoint to export; chat behavior currently favors the fully tuned checkpoint")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--precision", choices=sorted(DTYPES), default=None,
                        help="defaults to bf16 on mps/cuda, else fp32")
    parser.add_argument("--patience", type=int, default=3,
                        help="stop after this many evals with no val improvement; 0 disables")
    args = parser.parse_args()

    document = json.loads(args.config.read_text())
    config = Config.from_dict(document)
    tokenizer = document.get("tokenizer") or {}
    if tokenizer.get("kind") != "utf8_bytes" or tokenizer.get("vocab_size") != config.vocab_size:
        parser.error("tokenizer must be utf8_bytes with the same vocabulary size")
    training = document.get("training") or {}
    seed = int(training.get("seed", 14001))
    examples = read_examples(args.corpus)
    train_data, validation_data, data_meta = build_supervised_sets(examples, config.context_length, seed)
    corpus_hash = file_hash(args.corpus)

    target_steps = args.steps if args.steps is not None else int(training.get("steps", 1600))
    batch_size = args.batch if args.batch is not None else int(training.get("batch_size", 32))
    learning_rate = args.lr if args.lr is not None else float(training.get("learning_rate", 3e-4))
    warmup = int(training.get("warmup_steps", 50))
    eval_every = args.eval_every if args.eval_every is not None else int(training.get("eval_every", 100))
    eval_batches = int(training.get("eval_batches", 12))
    weight_decay = float(training.get("weight_decay", 0.05))
    if min(target_steps, batch_size, eval_every, eval_batches) < 1 or learning_rate <= 0:
        parser.error("training values must be positive")

    latest_path = args.out / "latest.pt"
    best_path = args.out / "best.pt"
    saved = None
    if args.resume:
        if not latest_path.exists():
            parser.error("no checkpoint to resume at %s" % latest_path)
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved.get("stage") != STAGE or saved.get("corpus_sha256") != corpus_hash:
            parser.error("checkpoint stage or corpus hash does not match")

    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if args.precision is None:
        args.precision = "bf16" if device in ("mps", "cuda") else "fp32"
    dtype = DTYPES[args.precision]
    amp = (lambda: torch.autocast(device_type=device, dtype=dtype)) if dtype is not None else nullcontext
    if args.precision == "fp16":
        print("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.", flush=True)
    model = LanguageModel(config)
    if saved:
        if Config.from_dict(saved["config"]) != config:
            parser.error("checkpoint architecture does not match")
        model.load_state_dict(saved["model"])
    else:
        initialized = torch.load(args.init, map_location="cpu", weights_only=False)
        if Config.from_dict(initialized["config"]) != config:
            parser.error("base checkpoint architecture does not match")
        model.load_state_dict(initialized["model"])
    model.to(device)

    decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    step = int(saved["step"]) if saved else 0
    best_validation = float(saved["best_validation"]) if saved else float("inf")
    if step >= target_steps:
        parser.error("checkpoint is already at step %d" % step)
    generator = torch.Generator()
    if saved:
        generator.set_state(saved["batch_generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
    else:
        generator.manual_seed(seed)

    @torch.no_grad()
    def evaluate():
        model.eval()
        fixed = torch.Generator().manual_seed(seed + 100000)
        losses = []
        for _ in range(eval_batches):
            x, y = sample_batch(validation_data, batch_size, fixed, device)
            losses.append(model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

    def rate(current):
        if current < warmup:
            return learning_rate * (current + 1) / max(warmup, 1)
        progress = (current - warmup) / max(target_steps - warmup, 1)
        return learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train.jsonl"
    if not saved:
        log_path.write_text("")
    started = time.time()
    print("smLANGUAGE_en_GENERAL_001 | %.2fM params | %s | precision %s | %d examples | steps %d->%d" %
          (model.parameter_count() / 1e6, device, args.precision, data_meta["items"], step, target_steps),
          flush=True)
    model.train()
    evals_since_improved = 0
    while step < target_steps:
        lr = rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = sample_batch(train_data, batch_size, generator, device)
        with amp():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step != 1 and step % eval_every and step != target_steps:
            continue
        validation_loss = evaluate()
        improved = validation_loss < best_validation
        if improved:
            best_validation = validation_loss
            evals_since_improved = 0
        else:
            evals_since_improved += 1
        event = {
            "stage": STAGE,
            "step": step,
            "train_loss": round(loss.item(), 6),
            "validation_loss": round(validation_loss, 6),
            "validation_bits_per_byte": round(validation_loss / math.log(2), 6),
            "learning_rate": lr,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        payload = checkpoint(model, optimizer, document, data_meta, corpus_hash, step,
                             best_validation, generator)
        atomic_torch_save(payload, latest_path)
        if improved:
            atomic_torch_save(payload, best_path)
        print("step %4d | train %.3f | val %.3f (%.2f bpb)%s" %
              (step, loss.item(), validation_loss, validation_loss / math.log(2),
               " | best" if improved else ""), flush=True)
        if args.patience and evals_since_improved >= args.patience:
            print("early stopping: val loss has not improved for %d evals, stopping at step %d, "
                  "best val %.4f" % (args.patience, step, best_validation), flush=True)
            break

    deployed_path = best_path if args.deploy == "best" else latest_path
    deployed = torch.load(deployed_path, map_location="cpu", weights_only=False)
    deployed["selection"] = args.deploy
    write_bundle(deployed, args.out)
    promote(args.out, args.promote_to, deployed_path)
    print("done | best validation %.4f | deployed %s step %d | %s" %
          (best_validation, args.deploy, deployed["step"], args.out / "weights.safetensors"), flush=True)


if __name__ == "__main__":
    main()
