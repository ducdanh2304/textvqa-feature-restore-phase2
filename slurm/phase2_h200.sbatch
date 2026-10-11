#!/usr/bin/env bash
# Edit DATA_ROOT, OUTPUT_ROOT, and REPO_ROOT before submission.
#SBATCH --job-name=textvqa_phase2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=<OUTPUT_ROOT>/slurm_%j.out
#SBATCH --error=<OUTPUT_ROOT>/slurm_%j.err

set -euo pipefail

REPO_ROOT="/path/to/textvqa-feature-restore-phase2"
DATA_ROOT="/path/to/textvqa_phase2_data"
OUTPUT_ROOT="/path/to/textvqa_phase2_output"
CONFIG="${REPO_ROOT}/configs/phase2_h200.yaml"

[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || { echo "Run scripts/setup_env.sh before sbatch" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}/hf_home" "${OUTPUT_ROOT}/hf_cache"
export HF_HOME="${OUTPUT_ROOT}/hf_home"
export HF_HUB_CACHE="${OUTPUT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Job=${SLURM_JOB_ID:-interactive} host=$(hostname) repo=${REPO_ROOT} data=${DATA_ROOT} output=${OUTPUT_ROOT}"
nvidia-smi
"${REPO_ROOT}/scripts/preflight.sh" --config "${CONFIG}" --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}"
"${REPO_ROOT}/scripts/cache_features.sh" --config "${CONFIG}" --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}"
"${REPO_ROOT}/scripts/train_adapter.sh" --config "${CONFIG}" --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}"
"${REPO_ROOT}/scripts/verify_checkpoint.sh" --config "${CONFIG}" --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" --checkpoint "${OUTPUT_ROOT}/training/best_adapter.pt"
"${REPO_ROOT}/scripts/make_bundle.sh" --config "${CONFIG}" --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}"
nvidia-smi
echo "Artifacts=${OUTPUT_ROOT}"
echo "Bundle=${OUTPUT_ROOT}.zip"

