"""OpenAI-compatible Ministral client with strict JSON schemas and durable call logs."""

import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from conversation_data import append_jsonl


THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def post_json(url, body, timeout=900):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class Teacher:
    def __init__(self, endpoint, model, log_path, retries=3):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.log_path = Path(log_path)
        self.retries = retries

    def ask(self, kind, system, prompt, schema, seed, temperature, max_tokens, metadata=None):
        body = {
            "model": self.model,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": kind, "strict": True, "schema": schema},
            },
        }
        last_error = None
        for attempt in range(1, self.retries + 1):
            started = time.time()
            try:
                data = post_json(self.endpoint + "/v1/chat/completions", body)
                text = str(data["choices"][0]["message"]["content"])
                text = THINK.sub("", text).strip()
                try:
                    parsed = json.loads(text)
                except ValueError:
                    start = text.find("{")
                    parsed, _ = json.JSONDecoder().raw_decode(text[start:])
                append_jsonl(self.log_path, {
                    "kind": kind,
                    "seed": seed,
                    "attempt": attempt,
                    "seconds": round(time.time() - started, 3),
                    "metadata": metadata or {},
                    "usage": data.get("usage") or {},
                    "ok": True,
                })
                return parsed
            except (KeyError, IndexError, TypeError, ValueError, urllib.error.URLError) as exc:
                last_error = exc
                append_jsonl(self.log_path, {
                    "kind": kind,
                    "seed": seed,
                    "attempt": attempt,
                    "seconds": round(time.time() - started, 3),
                    "metadata": metadata or {},
                    "ok": False,
                    "error": str(exc)[:500],
                })
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("teacher failed after %d attempts: %s" % (self.retries, last_error))
