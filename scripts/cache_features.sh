#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || { echo "Missing ${PY}; run scripts/setup_env.sh" >&2; exit 2; }
CONFIG=""; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$(realpath "$2")"; shift 2 ;;
    --data-root|--output-root|--model-path) EXTRA+=("$1" "$(realpath -m "$2")"); shift 2 ;;
    --offline) EXTRA+=("--offline"); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${CONFIG}" ]] || { echo "--config is required" >&2; exit 2; }
echo "Repository=${ROOT}"; echo "Config=${CONFIG}"; printf 'Overrides='; printf '%q ' "${EXTRA[@]}"; echo
exec "${PY}" -m textvqa_feature_restore.cli cache-features --config "${CONFIG}" "${EXTRA[@]}"

