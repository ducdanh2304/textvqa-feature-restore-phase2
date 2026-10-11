from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from . import MODEL_ID, MODEL_REVISION
from .config import output_root, phase1_root, write_resolved_config
from .data import assert_disjoint, validate_manifest
from .diagnostics import nvidia_smi_query, write_environment
from .io_utils import atomic_write_json, atomic_write_text, read_json, sha256_file


REQUIRED_PHASE1 = [
    "split_summary.json", "benchmark_excluded_image_ids.csv", "train_manifest.csv", "dev_manifest.csv",
    "architecture_report.json", "injection_point_report.json", "adapter_config.json",
    "trainable_parameters.json", "smoke_summary.json",
]


def _normalized(mapping: dict[str, Any]) -> dict[str, Any]:
    key = lambda value: "".join(char for char in str(value).lower() if char.isalnum())
    return {key(name): value for name, value in mapping.items()}


def _normalized_value(value: Any) -> Any:
    if isinstance(value, str):
        return "".join(char for char in value.upper() if char.isalnum())
    return value


def _phase1_metadata_checks(root: Path) -> dict[str, Any]:
    smoke = _normalized(read_json(root / "smoke_summary.json"))
    expected = {
        "status": "PHASE1SMOKECOMPLETE", "injectionstatus": "READY", "trainuniqueimages": 2000,
        "devuniqueimages": 200, "traindevoverlap": 0, "benchmarkoverlap": 0,
        "qwentrainableparameters": 0, "generationcompatibilitysuccesses": 16,
        "generationcompatibilitytotal": 16,
    }
    failures = {name: {"expected": value, "actual": smoke.get(name)} for name, value in expected.items()
                if _normalized_value(smoke.get(name)) != _normalized_value(value)}
    if failures:
        raise ValueError(f"Phase-1 smoke summary mismatch: {failures}")
    if int(smoke.get("adaptertrainableparameters", 0)) <= 0:
        raise ValueError("Phase-1 adapter trainable parameter count is not positive")
    architecture = read_json(root / "architecture_report.json")
    injection = read_json(root / "injection_point_report.json")
    adapter = read_json(root / "adapter_config.json")
    trainable = read_json(root / "trainable_parameters.json")
    if architecture.get("model_id") != MODEL_ID or architecture.get("model_revision") != MODEL_REVISION:
        raise ValueError("Phase-1 model identity mismatch")
    if injection.get("status") != "READY" or injection.get("candidate_module") != "model.model.visual":
        raise ValueError("Phase-1 injection point is not READY at model.model.visual")
    if int(injection.get("hidden_size", -1)) != 2048:
        raise ValueError("Phase-1 feature hidden size is not 2048")
    if adapter.get("class_name") != "ResidualFeatureRestorer" or int(adapter.get("hidden_size", -1)) != 2048 or int(adapter.get("bottleneck_dim", -1)) != 256:
        raise ValueError("Phase-1 adapter configuration mismatch")
    numerical = architecture.get("numerical_policy", {})
    if adapter.get("vision_dtype", numerical.get("vision_dtype")) != "torch.float32":
        raise ValueError("Phase-1 visual dtype is not float32")
    if adapter.get("language_dtype", numerical.get("language_dtype")) != "torch.float16":
        raise ValueError("Phase-1 language dtype is not float16")
    if int(trainable.get("qwen_trainable", -1)) != 0:
        raise ValueError("Phase-1 Qwen trainable count is not zero")
    return {"architecture": architecture, "injection": injection, "adapter": adapter, "trainable": trainable}


def run_preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    out, root = output_root(cfg), phase1_root(cfg)
    out.mkdir(parents=True, exist_ok=True)
    try:
        missing = [name for name in REQUIRED_PHASE1 if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Phase-1 files: {missing}")
        metadata = _phase1_metadata_checks(root)
        import pandas as pd
        benchmark = pd.read_csv(root / "benchmark_excluded_image_ids.csv", dtype=str, keep_default_na=False)
        if "image_id" not in benchmark or benchmark.image_id.nunique() != 1000 or len(benchmark) != 1000:
            raise ValueError("Benchmark exclusion file must contain exactly 1,000 unique image_id values")
        benchmark_ids = set(benchmark.image_id)
        train_frame, train_records = validate_manifest(root / "train_manifest.csv", phase1_root=root, split="train", expected_rows=2000, benchmark_ids=benchmark_ids)
        dev_frame, dev_records = validate_manifest(root / "dev_manifest.csv", phase1_root=root, split="dev", expected_rows=200, benchmark_ids=benchmark_ids)
        assert_disjoint(train_frame, dev_frame)

        gpu = nvidia_smi_query()
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("At least one CUDA GPU is required")
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if not cfg["hardware"]["allow_non_h200"] and not any("H200" in name.upper() for name in names):
            raise RuntimeError(f"No H200 is visible: {names}. Set allow_non_h200: true only for an explicit non-H200 test.")
        disk = shutil.disk_usage(out)
        max_tokens = int(cfg["processor"]["max_pixels"]) // (28 * 28) + 8
        bytes_per_value = 2
        estimated_cache = 4400 * max_tokens * 2048 * bytes_per_value + 4400 * 32_768
        required_disk = int(estimated_cache * 1.30)
        if disk.free < required_disk:
            raise RuntimeError(f"Insufficient output disk: free={disk.free}, required_with_30_percent_margin={required_disk}")
        if cfg["model"].get("offline"):
            model_path = Path(cfg["model"]["path"])
            if not (model_path / "config.json").is_file():
                raise FileNotFoundError(f"Offline model snapshot lacks config.json: {model_path}")
            revision_file = model_path / "REVISION"
            path_proves_revision = MODEL_REVISION in model_path.parts
            file_proves_revision = revision_file.is_file() and revision_file.read_text(encoding="utf-8").strip() == MODEL_REVISION
            if not (path_proves_revision or file_proves_revision):
                raise ValueError("Offline snapshot must be under a directory named by the pinned revision or contain REVISION with the exact hash")

        environment = write_environment(out / "environment.json")
        report = {
            "status": "READY", "phase1_root": str(root), "train_pairs": len(train_records), "dev_pairs": len(dev_records),
            "benchmark_unique_images": 1000, "train_dev_image_overlap": 0, "train_dev_question_overlap": 0,
            "benchmark_overlap": 0, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "injection_point": "model.model.visual", "hidden_size": 2048, "visual_dtype": "float32",
            "language_dtype": "float16", "visible_gpu_names": names, "nvidia_smi": gpu,
            "disk_free_bytes": disk.free, "estimated_cache_bytes": estimated_cache,
            "required_cache_disk_with_30_percent_margin": required_disk,
            "input_hashes": {name: sha256_file(root / name) for name in REQUIRED_PHASE1},
        }
        atomic_write_json(out / "preflight_report.json", report)
        markdown = "# Phase 2 preflight — READY\n\n" + "\n".join(f"- **{key}**: `{value}`" for key, value in report.items() if key not in {"input_hashes", "nvidia_smi"}) + "\n"
        atomic_write_text(out / "preflight_report.md", markdown)
        write_resolved_config(cfg, out / "phase2_config_resolved.yaml")
        return report
    except Exception as error:
        report = {"status": "NOT_READY", "reason": f"{type(error).__name__}: {error}", "phase1_root": str(root)}
        atomic_write_json(out / "preflight_report.json", report)
        atomic_write_text(out / "preflight_report.md", f"# Phase 2 preflight — NOT_READY\n\n{report['reason']}\n")
        raise

