"""Paratroop Harness adapter for smLANGUAGE_en_GENERAL_001."""

import json
from pathlib import Path
import threading

import torch

from model import Config, LanguageModel
from safetensors_io import load_file


class Generator:
    def __init__(self, checkpoint):
        path = Path(checkpoint)
        document = json.loads(path.with_name("model_config.json").read_text())
        state, self.metadata = load_file(path)
        self.config_document = document
        self.config = Config.from_dict(document)
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
        lines = ["Question: " + str(payload.get("message") or "").strip()]
        known = []
        for result in payload.get("tool_results") or []:
            if result.get("result"):
                known.append(str(result["result"]))
        for problem in payload.get("problems") or []:
            if problem.get("answer") is not None:
                known.append("%s = %s" % (problem.get("expression", "answer"), problem["answer"]))
        if known:
            lines.append("Known information: " + "; ".join(known))
        lines.append("Helpful English reply:")
        return "\n".join(lines) + " "

    @torch.no_grad()
    def complete(self, system, user, max_tokens=160, temperature=0.5, top_k=30, repetition_penalty=1.1):
        del system
        prompt = self.prompt(user)
        tokens = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=self.device)
        with self.lock:
            generated = self.model.generate(
                tokens,
                min(max(int(max_tokens), 1), 300),
                temperature=float(temperature),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
            )
        new_bytes = bytes(generated[0, tokens.shape[1] :].tolist())
        return new_bytes.decode("utf-8", errors="replace").split("\n\n", 1)[0].strip()

