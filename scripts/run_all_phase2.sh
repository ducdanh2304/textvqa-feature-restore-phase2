#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec > >(tee -a "${ROOT}/run_all_phase2.log") 2>&1
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
"${PY}" -m textvqa_feature_restore.cli preflight --config "${CONFIG}" "${EXTRA[@]}"
"${PY}" -m textvqa_feature_restore.cli cache-features --config "${CONFIG}" "${EXTRA[@]}"
"${PY}" -m textvqa_feature_restore.cli train --config "${CONFIG}" "${EXTRA[@]}"
OUTPUT_ROOT="$("${PY}" - "${CONFIG}" "${EXTRA[@]}" <<'PY'
import sys
from textvqa_feature_restore.cli import build_parser, _config
args=build_parser().parse_args(['verify','--checkpoint','placeholder',*sys.argv[2:],'--config',sys.argv[1]])
print(_config(args)['paths']['output_root'])
PY
)"
CHECKPOINT="${OUTPUT_ROOT}/training/best_adapter.pt"
"${PY}" -m textvqa_feature_restore.cli verify --config "${CONFIG}" --checkpoint "${CHECKPOINT}" "${EXTRA[@]}"
"${PY}" -m textvqa_feature_restore.cli bundle --config "${CONFIG}" "${EXTRA[@]}"
echo "Phase 2 artifacts: ${OUTPUT_ROOT}"
echo "Phase 2 archive: ${OUTPUT_ROOT}.zip"

