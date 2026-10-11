#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec > >(tee -a "${ROOT}/setup_env.log") 2>&1
VENV="${ROOT}/.venv"
OFFLINE=0
WHEEL_CACHE=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offline) OFFLINE=1; shift ;;
    --wheel-cache) WHEEL_CACHE="$(realpath "$2")"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "Unknown setup argument: $1" >&2; exit 2 ;;
  esac
done

echo "Repository: ${ROOT}"
echo "Virtual environment: ${VENV}"
"${PYTHON_BIN}" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,12), sys.version'
"${PYTHON_BIN}" -m venv "${VENV}"
PY="${VENV}/bin/python"

if [[ "${OFFLINE}" -eq 1 ]]; then
  [[ -n "${WHEEL_CACHE}" && -d "${WHEEL_CACHE}" ]] || { echo "Offline setup requires --wheel-cache DIR" >&2; exit 2; }
  "${PY}" -m pip install --no-index --find-links "${WHEEL_CACHE}" -r "${ROOT}/requirements-lock.txt"
  "${PY}" -m pip install --no-index --no-deps -e "${ROOT}"
else
  "${PY}" -m pip install --upgrade pip==25.0 setuptools==75.8.0 wheel==0.45.1
  "${PY}" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
  "${PY}" -m pip install -r "${ROOT}/requirements-lock.txt"
  "${PY}" -m pip install --no-deps -e "${ROOT}"
fi

"${PY}" -c 'import torch, textvqa_feature_restore; print("torch", torch.__version__, "package", textvqa_feature_restore.__version__)'
echo "Setup complete. Activate with: source ${VENV}/bin/activate"

