from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _unique(items: Iterable[Path], label: str) -> Path:
    values = sorted({item.resolve() for item in items})
    if len(values) != 1:
        listing = "\n".join(f"  - {item}" for item in values) or "  (không tìm thấy)"
        raise RuntimeError(f"Cần đúng một {label}, nhưng tìm thấy {len(values)}:\n{listing}")
    return values[0]


def discover_inputs(input_root: Path) -> dict[str, Path]:
    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Thư mục đầu vào không tồn tại: {input_root}")

    phase1 = []
    for manifest in input_root.rglob("train_manifest.csv"):
        root = manifest.parent
        required = [root / "dev_manifest.csv", root / "benchmark_excluded_image_ids.csv",
                    root / "data/train/clean", root / "data/train/degraded",
                    root / "data/dev/clean", root / "data/dev/degraded"]
        if all(path.exists() for path in required):
            phase1.append(root)
    phase1_root = _unique(phase1, "bộ dữ liệu kết quả Giai đoạn 1")

    benchmarks = []
    for manifest in input_root.rglob("final_main_manifest.csv"):
        degraded = manifest.parent / "images/realistic_mix/L2"
        if degraded.is_dir():
            benchmarks.append(manifest)
    benchmark_manifest = _unique(benchmarks, "final_main_manifest.csv kèm images/realistic_mix/L2")
    degraded_root = benchmark_manifest.parent / "images/realistic_mix/L2"

    frame = pd.read_csv(benchmark_manifest, dtype=str, keep_default_na=False)
    if not {"image_id", "question_id", "question", "answers_json"}.issubset(frame.columns):
        raise RuntimeError(f"Manifest benchmark thiếu cột bắt buộc: {benchmark_manifest}")
    image_ids = set(frame.image_id.astype(str))
    candidates: dict[Path, int] = {}
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        for path in input_root.rglob(suffix):
            if path.parent == degraded_root or phase1_root in path.parents:
                continue
            if path.stem in image_ids:
                candidates[path.parent.resolve()] = candidates.get(path.parent.resolve(), 0) + 1
    complete = [root for root, count in candidates.items() if count == len(image_ids)]
    clean_root = _unique(complete, f"thư mục ảnh sạch chứa đủ {len(image_ids)} ảnh benchmark")

    if len(frame) != 1000 or frame.image_id.nunique() != 1000 or frame.question_id.nunique() != 1000:
        raise RuntimeError("Benchmark phải có đúng 1.000 image_id và 1.000 question_id duy nhất")
    return {
        "input_root": input_root,
        "phase1_root": phase1_root,
        "benchmark_manifest": benchmark_manifest,
        "clean_root": clean_root,
        "degraded_root": degraded_root,
    }


def describe_inputs(paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}
