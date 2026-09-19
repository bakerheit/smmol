"""Validate accepted conversations and print a corpus quality report."""

import argparse
import hashlib
import json
from pathlib import Path

from conversation_data import HERE, corpus_report, load_config, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", action="store_true", help="require the configured 1,000 accepted conversations")
    args = parser.parse_args()
    config = load_config()
    path = HERE / "data" / "accepted" / "conversations.jsonl"
    conversations = read_jsonl(path)
    report = corpus_report(conversations, config)
    manifest_path = HERE / "data" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        digest = hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()
        if digest != manifest.get("sha256"):
            raise ValueError("accepted corpus hash does not match the manifest")
    if report["structural_failures"]:
        raise ValueError("accepted corpus has structural failures: %r" % report["structural_failures"][:3])
    if report["duplicate_conversations"]:
        raise ValueError("accepted corpus contains duplicate conversations")
    required = config["target_accepted"] if args.complete else min(config["pilot_accepted"], len(conversations))
    if args.complete and len(conversations) != required:
        raise ValueError("expected %d conversations, got %d" % (required, len(conversations)))
    print(json.dumps(report, indent=2, sort_keys=True))
    print("valid")


if __name__ == "__main__":
    main()
