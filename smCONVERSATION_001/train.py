"""Train the longer-context conversation model from the grade-one checkpoint.

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
from training_data import build_sets, encode_supervised, read_conversations, read_school


HERE = Path(__file__).resolve().parent
DEFAULT_INIT = HERE.parent / "smLANGUAGE_en_SCH_001" / "checkpoints" / "grade_01" / "best.pt"
DEFAULT_SCHOOL = HERE.parent / "smLANGUAGE_en_SCH_001" / "data" / "curriculum.jsonl"
STAGE = "conversation_v1_1k_context"
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_device(requested):
    if requested == "dml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("--device dml requires the torch-directml package") from exc
        return torch_directml.device()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    return requested


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


def data_hash(*paths, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def initialize_from_grade_one(model, base):
    source_config = Config.from_dict(base["config"])
    target_config = model.config
    for key in ("vocab_size", "width", "layers", "heads", "tie_embeddings"):
        if getattr(source_config, key) != getattr(target_config, key):
            raise ValueError("base and target %s do not match" % key)
    source = base["model"]
    target = model.state_dict()
    copied = []
    expanded = []
    for name, value in target.items():
        if name in source and source[name].shape == value.shape:
            target[name] = source[name]
            copied.append(name)
        elif name == "position_embedding.weight" and name in source and source[name].shape[1] == value.shape[1]:
            count = min(source[name].shape[0], value.shape[0])
            value[:count].copy_(source[name][:count])
            target[name] = value
            expanded.append({"tensor": name, "copied_positions": count, "new_positions": value.shape[0] - count})
        else:
            raise ValueError("cannot transfer base tensor %s" % name)
    model.load_state_dict(target)
    return {"copied_tensors": len(copied), "expanded": expanded}


def sample_batch(examples, batch_size, context, generator, device):
    indexes = torch.randint(len(examples), (batch_size,), generator=generator).tolist()
    encoded = [encode_supervised(examples[index], context) for index in indexes]
    x = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
    y = torch.tensor([item[1] for item in encoded], dtype=torch.long, device=device)
    return x, y


def write_bundle(payload, folder):
    folder = Path(folder)
    metadata = {
        "model": "smCONVERSATION_001",
        "stage": payload["stage"],
        "conversations": payload["data"]["conversations"],
        "assistant_targets": payload["data"]["assistant_targets"],
        "validation_domains": ",".join(payload["data"]["validation_domains"]),
        "corpus_sha256": payload["corpus_sha256"],
        "step": payload["step"],
        "format": "safetensors",
        "base": "smLANGUAGE_en_SCH_001-grade-01",
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
    parser.add_argument("--corpus", type=Path, default=HERE / "data" / "accepted" / "conversations.jsonl")
    parser.add_argument("--school-corpus", type=Path, default=DEFAULT_SCHOOL)
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--out", type=Path, default=HERE / "checkpoints" / STAGE)
    parser.add_argument("--promote-to", type=Path, default=HERE / "out")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="permit fewer than 1,000 conversations for a smoke run")
    parser.add_argument("--device", choices=("auto", "cuda", "dml", "mps", "cpu"), default="auto")
    parser.add_argument("--precision", choices=sorted(DTYPES), default=None,
                         help="autocast dtype for the forward pass; defaults to bf16 on mps/cuda, else fp32")
    parser.add_argument("--patience", type=int, default=3,
                         help="stop after this many evals with no validation improvement; 0 disables")
    args = parser.parse_args()

    document = json.loads(args.config.read_text())
    config = Config.from_dict(document)
    training = document["training"]
    conversations = read_conversations(args.corpus)
    if len(conversations) < 1000 and not args.allow_partial:
        parser.error("refusing to train the final model on %d/1000 conversations; use --allow-partial for a smoke run" % len(conversations))
    school = read_school(args.school_corpus)
    train_examples, validation_examples, data_meta = build_sets(
        conversations, school, config.context_length, training["seed"],
        training["school_replay_fraction"], training["max_history_turns"],
    )
    settings = {
        "seed": training["seed"], "replay": training["school_replay_fraction"],
        "history": training["max_history_turns"], "context": config.context_length,
    }
    fingerprint = data_hash(args.corpus, args.school_corpus, settings=settings)

    target_steps = args.steps if args.steps is not None else training["steps"]
    batch_size = args.batch if args.batch is not None else training["batch_size"]
    eval_every = args.eval_every if args.eval_every is not None else training["eval_every"]
    learning_rate = training["learning_rate"]
    eval_batches = training["eval_batches"]
    warmup = training["warmup_steps"]
    if min(target_steps, batch_size, eval_every, eval_batches) < 1:
        parser.error("training values must be positive")

    latest_path = args.out / "latest.pt"
    best_path = args.out / "best.pt"
    saved = None
    if args.resume:
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved.get("corpus_sha256") != fingerprint or saved.get("stage") != STAGE:
            parser.error("resume checkpoint does not match this corpus and stage")

    torch.manual_seed(training["seed"])
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    if args.precision is None:
        args.precision = "bf16" if device in ("mps", "cuda") else "fp32"
    dtype = DTYPES[args.precision]
    if args.precision == "fp16":
        print("note: fp16 has no gradient scaler here; bf16 is the safe choice on this stack.", flush=True)
    amp = (lambda: torch.autocast(device_type=device, dtype=dtype)) if dtype is not None else nullcontext
    model = LanguageModel(config)
    if saved:
        model.load_state_dict(saved["model"])
        transfer = saved["transfer"]
    else:
        base = torch.load(args.init, map_location="cpu", weights_only=False)
        transfer = initialize_from_grade_one(model, base)
    model.to(device)

    decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": training["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate, betas=(0.9, 0.95),
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
        generator.manual_seed(training["seed"])

    @torch.no_grad()
    def evaluate():
        model.eval()
        fixed = torch.Generator().manual_seed(training["seed"] + 100000)
        losses = []
        for _ in range(eval_batches):
            x, y = sample_batch(validation_examples, batch_size, config.context_length, fixed, device)
            losses.append(model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

    def rate(current):
        if current < warmup:
            return learning_rate * (current + 1) / warmup
        progress = (current - warmup) / max(target_steps - warmup, 1)
        return learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1))))

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train.jsonl"
    if not saved:
        log_path.write_text("")
    started = time.time()
    print("smCONVERSATION_001 | %.2fM params | %s | %d conversations | %d targets | steps %d->%d | "
          "precision %s | patience %d" % (
        model.parameter_count() / 1e6, device, len(conversations), data_meta["assistant_targets"], step,
        target_steps, args.precision, args.patience,
    ), flush=True)
    model.train()
    stall = 0
    while step < target_steps:
        lr = rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = sample_batch(train_examples, batch_size, config.context_length, generator, device)
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
        best_validation = min(best_validation, validation_loss)
        stall = 0 if improved else stall + 1
        event = {
            "step": step, "train_loss": round(loss.item(), 6),
            "validation_loss": round(validation_loss, 6),
            "validation_bits_per_byte": round(validation_loss / math.log(2), 6),
            "learning_rate": lr, "elapsed_seconds": round(time.time() - started, 3),
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        payload = {
            "format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": document, "stage": STAGE, "data": data_meta, "corpus_sha256": fingerprint,
            "base_checkpoint": str(args.init), "transfer": transfer, "step": step,
            "best_validation": best_validation, "batch_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        atomic_torch_save(payload, latest_path)
        if improved:
            atomic_torch_save(payload, best_path)
        print("step %4d | train %.3f | val %.3f%s" % (
            step, loss.item(), validation_loss, " | best" if improved else ""
        ), flush=True)
        if args.patience and stall >= args.patience:
            print(
                "early stop | no validation improvement for %d evals | stopped at step %d | best val %.4f"
                % (args.patience, step, best_validation),
                flush=True,
            )
            break

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    write_bundle(best, args.out)
    promote(args.out, args.promote_to, best_path)
    print("done | best validation %.4f" % best_validation, flush=True)


if __name__ == "__main__":
    main()
