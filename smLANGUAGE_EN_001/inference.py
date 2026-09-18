"""In-process adapter used by paratroop_harness_02's Language module."""

import json
import threading

import torch

from model import Config, LanguageModel


class Generator:
    """Load one checkpoint and turn a harness language payload into text completion."""

    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = LanguageModel(Config(**saved["config"]))
        self.model.load_state_dict(saved["model"])
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
        lines = ["Question: " + str(payload.get("message") or "")]
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
            lines.append("The question to ask: " + question)
        else:
            summary = str((payload.get("decision") or {}).get("summary") or "").strip()
            if summary:
                lines.append("Goal: " + summary)
        lines.append('A helpful reply: "')
        return "\n".join(lines)

    @torch.no_grad()
    def complete(self, system, user, max_tokens=300, temperature=0.4):
        """Generate only; the adapter does not fake instruction-following or tool use."""
        del system
        prompt = self.prompt(user)
        tokens = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=self.device)
        # This model has byte tokens. A short cap keeps one harness reply from rambling for pages.
        limit = min(max(int(max_tokens), 1), 400)
        with self.lock:
            generated = self.model.generate(tokens, limit, temperature=max(float(temperature), 0.1), top_k=40)
        new_bytes = bytes(generated[0, tokens.shape[1] :].tolist())
        text = new_bytes.decode("utf-8", errors="replace").strip()
        # TinyStories often closes a spoken line with a quote. Stop there, or at the next paragraph.
        text = text.split('"', 1)[0].split("\n\n", 1)[0].strip()
        return text
