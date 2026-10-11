#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || { echo "Missing ${PY}; run scripts/setup_env.sh" >&2; exit 2; }
CONFIG=""; EXTRA=(); RESUME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$(realpath "$2")"; shift 2 ;;
    --data-root|--output-root|--model-path) EXTRA+=("$1" "$(realpath -m "$2")"); shift 2 ;;
    --offline) EXTRA+=("--offline"); shift ;;
    --resume) RESUME="$(realpath "$2")"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${CONFIG}" ]] || { echo "--config is required" >&2; exit 2; }
echo "Repository=${ROOT}"; echo "Config=${CONFIG}"; echo "Resume=${RESUME:-none}"; printf 'Overrides='; printf '%q ' "${EXTRA[@]}"; echo
COMMAND=("${PY}" -m textvqa_feature_restore.cli train --config "${CONFIG}" "${EXTRA[@]}")
[[ -z "${RESUME}" ]] || COMMAND+=(--resume "${RESUME}")
exec "${COMMAND[@]}"

