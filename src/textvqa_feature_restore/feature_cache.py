from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from . import CACHE_IMPLEMENTATION_VERSION, MODEL_ID, MODEL_REVISION
from .config import output_root, phase1_root
from .data import PairRecord, iter_conditions, validate_manifest
from .diagnostics import append_error, configure_logging, tensor_stats
from .io_utils import atomic_torch_save, atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_bytes, sha256_file
from .model import FrozenQwenVisualExtractor
from .reproducibility import set_reproducible


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def expected_cache_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    root = phase1_root(cfg)
    return {
        "cache_implementation_version": CACHE_IMPLEMENTATION_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "processor_use_fast": False,
        "min_pixels": int(cfg["processor"]["min_pixels"]),
        "max_pixels": int(cfg["processor"]["max_pixels"]),
        "attention_implementation": "sdpa",
        "visual_dtype": "torch.float32",
        "cache_storage_dtype": cfg["cache"]["storage_dtype"],
        "train_manifest_sha256": sha256_file(root / "train_manifest.csv"),
        "dev_manifest_sha256": sha256_file(root / "dev_manifest.csv"),
        "benchmark_excluded_ids_sha256": sha256_file(root / "benchmark_excluded_image_ids.csv"),
    }


def cache_filename(cache_files: Path, record: PairRecord, condition: str) -> Path:
    return cache_files / f"{record.split}__{record.image_id}__{condition}.pt"


def _metadata_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    bad = {key: {"expected": value, "actual": actual.get(key)} for key, value in expected.items() if actual.get(key) != value}
    if bad:
        raise ValueError(f"Stale/mismatched cache metadata: {bad}")


def validate_cache_file(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "feature" not in payload or "metadata" not in payload:
        raise ValueError(f"Malformed cache payload: {path}")
    metadata, feature = payload["metadata"], payload["feature"]
    _metadata_matches(metadata, expected)
    if list(feature.shape) != metadata.get("feature_shape") or str(feature.dtype) != metadata.get("feature_dtype"):
        raise ValueError(f"Cache tensor shape/dtype mismatch: {path}")
    if feature.ndim != 2 or feature.shape[-1] != 2048:
        raise ValueError(f"Cache feature must have shape [tokens, 2048]: {path}")
    if not torch.isfinite(feature).all():
        raise ValueError(f"Non-finite cached tensor: {path}")
    if sha256_bytes(tensor_bytes(feature)) != metadata.get("cached_feature_sha256"):
        raise ValueError(f"Cached tensor checksum mismatch: {path}")
    return metadata


def _record_metadata(
    cfg: dict[str, Any], record: PairRecord, condition: str, source: Path, source_sha: str,
    feature_fp32: torch.Tensor, stored: torch.Tensor, gpu_name: str,
) -> dict[str, Any]:
    import transformers
    import qwen_vl_utils

    provenance = expected_cache_provenance(cfg)
    stats = tensor_stats(feature_fp32)
    return {
        **provenance,
        "image_id": record.image_id, "question_id": record.question_id, "split": record.split,
        "condition": condition, "source_image_path": str(source), "source_image_sha256": source_sha,
        "source_width": record.source_width, "source_height": record.source_height,
        "degradation_type": record.degradation_type, "degradation_level": record.degradation_level,
        "transformers_version": transformers.__version__, "qwen_vl_utils_version": getattr(qwen_vl_utils, "__version__", "unknown"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "gpu_name": gpu_name,
        "feature_shape": list(stored.shape), "feature_dtype": str(stored.dtype),
        "fp32_feature_sha256": sha256_bytes(tensor_bytes(feature_fp32)),
        "cached_feature_sha256": sha256_bytes(tensor_bytes(stored)),
        "fp32_statistics": stats,
        "downcast_max_abs_error": float((stored.float() - feature_fp32).abs().max().item()),
    }


def _expected_for_record(cfg: dict[str, Any], record: PairRecord, condition: str, source: Path, source_sha: str) -> dict[str, Any]:
    return {
        **expected_cache_provenance(cfg),
        "image_id": record.image_id, "question_id": record.question_id, "split": record.split,
        "condition": condition, "source_image_path": str(source), "source_image_sha256": source_sha,
        "source_width": record.source_width, "source_height": record.source_height,
        "degradation_type": record.degradation_type, "degradation_level": record.degradation_level,
    }


def load_phase1_records(cfg: dict[str, Any]) -> tuple[list[PairRecord], list[PairRecord]]:
    import pandas as pd
    root = phase1_root(cfg)
    benchmark = pd.read_csv(root / "benchmark_excluded_image_ids.csv", dtype=str, keep_default_na=False)
    benchmark_ids = set(benchmark.image_id)
    _, train = validate_manifest(root / "train_manifest.csv", phase1_root=root, split="train", expected_rows=2000, benchmark_ids=benchmark_ids)
    _, dev = validate_manifest(root / "dev_manifest.csv", phase1_root=root, split="dev", expected_rows=200, benchmark_ids=benchmark_ids)
    return train, dev


def run_cache(cfg: dict[str, Any]) -> dict[str, Any]:
    out = output_root(cfg)
    if not (out / "preflight_report.json").is_file():
        raise RuntimeError("Run preflight successfully before cache-features")
    if json.loads((out / "preflight_report.json").read_text())["status"] != "READY":
        raise RuntimeError("Preflight status is not READY")
    cache_root, cache_files = out / "cache", out / "cache" / "features"
    cache_files.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(cache_root / "cache.log", "phase2.cache")
    errors = cache_root / "errors.jsonl"
    set_reproducible(int(cfg["reproducibility"]["seed"]), bool(cfg["hardware"]["allow_tf32"]))
    train, dev = load_phase1_records(cfg)
    jobs = list(iter_conditions(train)) + list(iter_conditions(dev))
    if len(jobs) != 4400:
        raise RuntimeError(f"Expected 4,400 cache jobs, found {len(jobs)}")
    batch_size = int(cfg["cache"]["batch_size"])
    if batch_size < 1:
        raise ValueError("cache.batch_size must be positive")
    extractor = FrozenQwenVisualExtractor(cfg)
    gpu_name = torch.cuda.get_device_name(0)
    manifest_records: list[dict[str, Any]] = []
    reused = newly_extracted = failures = 0
    extraction_times: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    try:
        pending: list[tuple[PairRecord, str, Path, str]] = []
        for job in jobs:
            record, condition, source, source_sha = job
            target = cache_filename(cache_files, record, condition)
            expected = _expected_for_record(cfg, record, condition, source, source_sha)
            if target.is_file():
                metadata = validate_cache_file(target, expected)
                manifest_records.append({**metadata, "cache_file": str(target), "cache_file_sha256": sha256_file(target)})
                reused += 1
            else:
                pending.append(job)
        for offset in range(0, len(pending), batch_size):
            batch_jobs = pending[offset:offset + batch_size]
            paths = [job[2] for job in batch_jobs]
            start = time.perf_counter()
            features = extractor.extract_batch(paths)
            if extractor.qwen_gradient_count() != 0:
                raise RuntimeError("Frozen Qwen received gradients during inference-mode cache extraction")
            elapsed = time.perf_counter() - start
            extraction_times.extend([elapsed / len(paths)] * len(paths))
            for job, feature_fp32 in zip(batch_jobs, features):
                record, condition, source, source_sha = job
                stored = feature_fp32.to(torch.float16 if cfg["cache"]["storage_dtype"] == "float16" else torch.bfloat16).contiguous()
                if not torch.isfinite(stored).all():
                    raise RuntimeError(f"Feature downcast became non-finite: {record.split}/{record.image_id}/{condition}")
                metadata = _record_metadata(cfg, record, condition, source, source_sha, feature_fp32, stored, gpu_name)
                target = cache_filename(cache_files, record, condition)
                atomic_torch_save({"metadata": metadata, "feature": stored}, target)
                validate_cache_file(target, _expected_for_record(cfg, record, condition, source, source_sha))
                manifest_records.append({**metadata, "cache_file": str(target), "cache_file_sha256": sha256_file(target)})
                newly_extracted += 1
                if (newly_extracted + reused) % 50 == 0:
                    logger.info("Cache progress %d/4400 (new=%d reused=%d)", newly_extracted + reused, newly_extracted, reused)
        manifest_records.sort(key=lambda item: (item["split"], item["image_id"], item["condition"]))
        if len(manifest_records) != 4400 or len({(x["split"], x["image_id"], x["condition"]) for x in manifest_records}) != 4400:
            raise RuntimeError("Cache manifest count or uniqueness failure")
        atomic_write_jsonl(cache_root / "manifest.jsonl", manifest_records)
        disk_usage = sum(path.stat().st_size for path in cache_root.rglob("*") if path.is_file())
        mean_seconds = float(sum(extraction_times) / len(extraction_times)) if extraction_times else 0.0
        equivalent_without_cache = mean_seconds * 4400 * max(int(cfg["training"]["max_epochs"]) - 1, 0)
        summary = {
            "status": "COMPLETE", "total_cached": len(manifest_records), "reused_cache": reused,
            "newly_extracted": newly_extracted, "failures": failures, "mean_extraction_seconds_per_image": mean_seconds,
            "cache_extraction_batch_size": batch_size,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "disk_usage_bytes": disk_usage,
            "estimated_equivalent_non_cache_training_seconds_avoided": equivalent_without_cache,
            "manifest_sha256": sha256_file(cache_root / "manifest.jsonl"), **expected_cache_provenance(cfg),
        }
        atomic_write_json(cache_root / "cache_summary.json", summary)
        atomic_write_json(out / "cache_report.json", summary)
        markdown = "# Feature cache — COMPLETE\n\n" + "\n".join(f"- **{key}**: `{value}`" for key, value in summary.items()) + "\n"
        from .io_utils import atomic_write_text
        atomic_write_text(out / "cache_report.md", markdown)
        return summary
    except Exception as error:
        failures += 1
        append_error(errors, "cache_features", error, newly_extracted=newly_extracted, reused=reused)
        raise
    finally:
        del extractor
        gc.collect()
        torch.cuda.empty_cache()

