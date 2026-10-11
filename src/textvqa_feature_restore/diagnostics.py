from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from .io_utils import atomic_write_json


def configure_logging(path: str | Path, name: str) -> logging.Logger:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def append_error(path: str | Path, stage: str, error: BaseException, **context: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "context": context,
        "traceback": traceback.format_exc(),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def nvidia_smi_query() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        rows = [line.strip().split(", ") for line in result.stdout.splitlines() if line.strip()]
        return {"available": True, "gpus": [dict(name=r[0], memory_total_mib=int(r[1]), memory_free_mib=int(r[2]), driver=r[3]) for r in rows]}
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}


def write_environment(path: str | Path) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "qwen-vl-utils", "accelerate", "huggingface-hub", "numpy", "pandas", "Pillow", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi": nvidia_smi_query(),
        "packages": packages,
    }
    atomic_write_json(path, report)
    return report


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    finite = torch.isfinite(value)
    return {
        "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
        "finite": bool(finite.all().item()), "nan_count": int(torch.isnan(value).sum().item()),
        "posinf_count": int(torch.isposinf(value).sum().item()), "neginf_count": int(torch.isneginf(value).sum().item()),
        "finite_min": float(value[finite].min().item()) if finite.any() else None,
        "finite_max": float(value[finite].max().item()) if finite.any() else None,
        "mean": float(value[finite].float().mean().item()) if finite.any() else None,
        "std": float(value[finite].float().std(unbiased=False).item()) if finite.any() else None,
        "max_abs": float(value[finite].float().abs().max().item()) if finite.any() else None,
    }

