#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${FINETUNED_MODEL_PATH:?Set FINETUNED_MODEL_PATH to the exported .nemo model}"
exec uvicorn backend.server:app --host 0.0.0.0 --port "${BACKEND_PORT:-8001}" --workers 1
