"""In-process Paratroop Harness adapter for smLANGUAGE_en_SCH_001."""

import json
from pathlib import Path
import threading

import torch

from model import Config, LanguageModel
from safetensors_io import load_file


class Generator:
    def __init__(self, checkpoint):
        path = Path(checkpoint)
        if path.suffix == ".safetensors":
            document = json.loads(path.with_name("model_config.json").read_text())
            state, self.metadata = load_file(path)
        else:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            document = saved["config"]
            state = saved["model"]
            self.metadata = {"stage": saved.get("stage", "unknown"), "step": str(saved.get("step", 0))}
        self.config_document = document
        self.config = Config.from_dict(document)
        tokenizer = document.get("tokenizer") or {}
        if tokenizer.get("kind") != "utf8_bytes" or self.config.vocab_size != 256:
            raise ValueError("the harness adapter needs the utf8_bytes tokenizer with vocab_size 256")
        self.model = LanguageModel(self.config)
        self.model.load_state_dict(state)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.lock = threading.Lock()

    @staticmethod
    def _payload(user):
        try:
            return json.loads(user.split("Input:\n", 1)[1].split("\n\nYour last reply", 1)[0])
        except (IndexError, TypeError, ValueError):
            return {"message": str(user)}

    @classmethod
    def prompt(cls, user):
        payload = cls._payload(user)
        message = str(payload.get("message") or "").strip()
        lines = ["Question: " + message]
        known = []
        for result in payload.get("tool_results") or []:
            if result.get("result"):
                known.append(str(result["result"]))
        for problem in payload.get("problems") or []:
            if problem.get("answer") is not None:
                known.append("%s = %s" % (problem.get("expression", "answer"), problem["answer"]))
        if known:
            lines.append("Known information: " + "; ".join(known))
        question = str(payload.get("question_to_ask") or "").strip()
        if question:
            lines.append("Question to ask: " + question)
        lines.append("Helpful English reply:")
        return "\n".join(lines) + " "

    @torch.no_grad()
    def complete(
        self,
        system,
        user,
        max_tokens=240,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.05,
    ):
        del system
        prompt = self.prompt(user)
        tokens = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=self.device)
        limit = min(max(int(max_tokens), 1), 400)
        with self.lock:
            generated = self.model.generate(
                tokens,
                limit,
                temperature=float(temperature),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
            )
        new_bytes = bytes(generated[0, tokens.shape[1] :].tolist())
        text = new_bytes.decode("utf-8", errors="replace").strip()
        return text.split("\n\n", 1)[0].strip()
