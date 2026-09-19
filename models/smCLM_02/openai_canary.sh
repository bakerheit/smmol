#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ -z "${OPENAI_API_KEY:-}" ] && command -v security >/dev/null 2>&1; then
    OPENAI_API_KEY=$(security find-generic-password \
        -a "${USER:-}" -s SMMOL_OPENAI_API_KEY -w 2>/dev/null || true)
    export OPENAI_API_KEY
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY is missing." >&2
    echo "Install a key in the environment or the macOS Keychain service" >&2
    echo "SMMOL_OPENAI_API_KEY. Use a fresh key scoped to this project, and never one" >&2
    echo "that has been pasted anywhere outside a password manager." >&2
    exit 2
fi

exec python3 "$SCRIPT_DIR/propose_ideas.py" \
    --provider openai \
    --model gpt-5.6-luna \
    --max-budget-usd 0.25 \
    --batch-size 20 \
    --limit 2 \
    "$@"
