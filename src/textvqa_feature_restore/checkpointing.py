from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .io_utils import sha256_file


REQUIRED_CHECKPOINT_FIELDS = {
    "adapter_state_dict", "optimizer_state_dict", "epoch", "global_step", "best_dev_loss",
    "current_dev_loss", "early_stopping_counter", "adapter_config", "model_id", "model_revision",
    "train_manifest_sha256", "dev_manifest_sha256", "benchmark_excluded_ids_sha256",
    "cache_manifest_sha256", "training_config_sha256", "random_state", "numpy_random_state",
    "torch_rng_state", "cuda_rng_state_all", "batch_size", "cache_storage_dtype",
    "cache_implementation_version",
}


def assert_checkpoint_fields(payload: dict[str, Any]) -> None:
    missing = REQUIRED_CHECKPOINT_FIELDS - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")


def expected_provenance(
    *,
    model_id: str,
    model_revision: str,
    train_manifest: str | Path,
    dev_manifest: str | Path,
    benchmark_ids: str | Path,
    cache_manifest: str | Path,
    resolved_config: str | Path,
    batch_size: int,
    cache_storage_dtype: str,
    cache_implementation_version: str,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "train_manifest_sha256": sha256_file(train_manifest),
        "dev_manifest_sha256": sha256_file(dev_manifest),
        "benchmark_excluded_ids_sha256": sha256_file(benchmark_ids),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "training_config_sha256": sha256_file(resolved_config),
        "batch_size": int(batch_size),
        "cache_storage_dtype": cache_storage_dtype,
        "cache_implementation_version": cache_implementation_version,
    }


def validate_checkpoint_provenance(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    assert_checkpoint_fields(payload)
    mismatch = {key: {"expected": value, "actual": payload.get(key)} for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise ValueError(f"Checkpoint provenance mismatch: {mismatch}")


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint root must be a mapping")
    assert_checkpoint_fields(payload)
    return payload

