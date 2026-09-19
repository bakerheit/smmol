"""Collect a bounded, license-gated source corpus through the GitHub REST API."""

import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "sources.json"
DEFAULT_OUTPUT = HERE / "data" / "raw" / "records.jsonl"
IGNORED_PARTS = {
    ".git", ".github", ".mypy_cache", ".pytest_cache", ".tox", ".venv", "__pycache__",
    "build", "coverage", "dist", "docs/_build", "generated", "node_modules", "vendor", "venv",
}
GENERATED_MARKERS = (
    "auto-generated", "automatically generated", "code generated", "do not edit",
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key)\s*=\s*['\"][^'\"]{8,}['\"]"),
)


class ApiError(RuntimeError):
    pass


class GitHubClient:
    """Small serial client that delegates auth and rate-limit handling to `gh`."""

    def __init__(self, executable="gh"):
        self.executable = executable

    def json(self, endpoint):
        command = [
            self.executable, "api", "--method", "GET",
            "-H", "Accept: application/vnd.github+json", endpoint,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip() or "unknown GitHub API error"
            raise ApiError("%s: %s" % (endpoint, message))
        try:
            return json.loads(result.stdout)
        except ValueError as exc:
            raise ApiError("%s did not return JSON" % endpoint) from exc


def normalized_parts(path):
    return [part.lower() for part in PurePosixPath(path).parts]


def path_allowed(path, extensions):
    candidate = PurePosixPath(path)
    parts = normalized_parts(path)
    if candidate.suffix.lower() not in extensions:
        return False
    for ignored in IGNORED_PARTS:
        ignored_parts = ignored.split("/")
        if len(ignored_parts) == 1 and ignored_parts[0] in parts:
            return False
        if len(ignored_parts) > 1 and "/".join(parts).find(ignored) >= 0:
            return False
    name = candidate.name.lower()
    return not name.endswith(".min.py")


def decode_source(raw, min_bytes, max_bytes):
    if not min_bytes <= len(raw) <= max_bytes or b"\x00" in raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    head = text[:2000].lower()
    if any(marker in head for marker in GENERATED_MARKERS):
        return None
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return None
    lines = text.splitlines()
    if lines and max(map(len, lines)) > 1000:
        return None
    printable = sum(character.isprintable() or character in "\n\r\t" for character in text)
    if text and printable / len(text) < 0.98:
        return None
    return text


def decode_blob(payload):
    if payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
        raise ApiError("GitHub blob was not base64 encoded")
    try:
        encoded = "".join(payload["content"].split())
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ApiError("GitHub blob contained invalid base64") from exc


def repository_candidates(client, repositories, query, max_repositories):
    names = list(repositories)
    if query:
        safe_query = "%s is:public archived:false fork:false" % query
        endpoint = "search/repositories?q=%s&sort=stars&order=desc&per_page=%d" % (
            quote(safe_query), min(max_repositories, 100),
        )
        for item in client.json(endpoint).get("items", []):
            names.append(item["full_name"])
    deduplicated = []
    for name in names:
        if name not in deduplicated:
            deduplicated.append(name)
    return deduplicated[:max_repositories]


def inspect_repository(client, name, allowed_licenses):
    metadata = client.json("repos/%s" % name)
    if metadata.get("visibility") != "public" or metadata.get("private"):
        raise ValueError("repository is not public")
    if metadata.get("archived") or metadata.get("fork"):
        raise ValueError("repository is archived or a fork")
    license_id = (metadata.get("license") or {}).get("spdx_id")
    if license_id not in allowed_licenses:
        raise ValueError("license %r is not allowlisted" % license_id)
    branch = metadata.get("default_branch")
    commit = client.json("repos/%s/commits/%s" % (name, quote(branch, safe="")))
    commit_sha = commit["sha"]
    tree_sha = commit["commit"]["tree"]["sha"]
    license_payload = client.json("repos/%s/license" % name)
    license_text = decode_blob(license_payload).decode("utf-8", errors="replace")
    return {
        "repository": metadata["full_name"],
        "repository_url": metadata["html_url"],
        "default_branch": branch,
        "commit": commit_sha,
        "tree": tree_sha,
        "license": license_id,
        "license_path": license_payload.get("path", "LICENSE"),
        "license_url": license_payload.get("html_url"),
        "license_text": license_text,
    }


def collect_repository(client, source, settings, seen_hashes, remaining_bytes):
    name = source["repository"]
    tree = client.json("repos/%s/git/trees/%s?recursive=1" % (name, source["tree"]))
    if tree.get("truncated"):
        raise ValueError("recursive Git tree was truncated")
    extensions = {value.lower() for value in settings["extensions"]}
    blobs = [
        item for item in tree.get("tree", [])
        if item.get("type") == "blob"
        and settings["min_file_bytes"] <= int(item.get("size", 0)) <= settings["max_file_bytes"]
        and path_allowed(item.get("path", ""), extensions)
    ]
    seed = settings["selection_seed"]
    blobs.sort(key=lambda item: hashlib.sha256(
        ("%s:%s:%s" % (seed, name, item["path"])).encode("utf-8")
    ).digest())
    records = []
    used_bytes = 0
    limit = min(settings["max_bytes_per_repository"], remaining_bytes)
    for item in blobs:
        if len(records) >= settings["max_files_per_repository"]:
            break
        if used_bytes + item["size"] > limit:
            continue
        payload = client.json("repos/%s/git/blobs/%s" % (name, item["sha"]))
        raw = decode_blob(payload)
        text = decode_source(raw, settings["min_file_bytes"], settings["max_file_bytes"])
        if text is None:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        record_id = hashlib.sha256(
            ("%s@%s:%s" % (name, source["commit"], item["path"])).encode("utf-8")
        ).hexdigest()[:24]
        records.append({
            "id": record_id,
            "repository": name,
            "repository_url": source["repository_url"],
            "commit": source["commit"],
            "license": source["license"],
            "license_url": source["license_url"],
            "path": item["path"],
            "blob_sha": item["sha"],
            "sha256": digest,
            "bytes": len(raw),
            "content": text,
        })
        used_bytes += len(raw)
    return records


def write_corpus(records, repositories, settings, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    output.write_text(payload)
    license_folder = output.parent / "licenses"
    license_folder.mkdir(parents=True, exist_ok=True)
    for repository in repositories:
        filename = repository["repository"].replace("/", "__") + ".txt"
        (license_folder / filename).write_text(repository.pop("license_text"))
        repository["license_file"] = str((license_folder / filename).relative_to(output.parent))
    manifest = {
        "format_version": 1,
        "source": "GitHub REST API",
        "records": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "licenses": dict(sorted(Counter(record["license"] for record in records).items())),
        "settings": settings,
        "repositories": repositories,
    }
    output.with_name("manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo", action="append", default=[], help="public owner/repo; may be repeated")
    parser.add_argument("--query", help="optional GitHub repository-search query")
    parser.add_argument("--max-repositories", type=int, default=20)
    args = parser.parse_args(argv)
    document = json.loads(args.config.read_text())
    repositories = args.repo or document.get("repositories", [])
    if not repositories and not args.query:
        parser.error("configure at least one repository or query")
    settings = {key: value for key, value in document.items() if key not in {"format_version", "repositories"}}
    client = GitHubClient()
    names = repository_candidates(client, repositories, args.query, args.max_repositories)
    accepted = []
    records = []
    seen_hashes = set()
    for name in names:
        if sum(record["bytes"] for record in records) >= settings["max_total_bytes"]:
            break
        try:
            source = inspect_repository(client, name, set(settings["allowed_licenses"]))
            remaining = settings["max_total_bytes"] - sum(record["bytes"] for record in records)
            selected = collect_repository(client, source, settings, seen_hashes, remaining)
            if not selected:
                raise ValueError("no eligible source files")
            accepted.append(source)
            records.extend(selected)
            print("accepted %s @ %.12s | %s | %d files | %d bytes" % (
                name, source["commit"], source["license"], len(selected),
                sum(record["bytes"] for record in selected),
            ), file=sys.stderr, flush=True)
        except (ApiError, KeyError, ValueError) as exc:
            print("skipped %s: %s" % (name, exc), file=sys.stderr, flush=True)
    if len(accepted) < 2:
        parser.error("need at least two accepted repositories for a repository-held-out split")
    manifest = write_corpus(records, accepted, settings, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
