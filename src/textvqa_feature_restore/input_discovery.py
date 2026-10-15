from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .io_utils import atomic_write_json, sha256_file

STVQA_HF_DATASET = "vikhyatk/st-vqa"
STVQA_HF_REVISION = "3453b8a0359fd2cea7f0c2c5602ba8618af59794"


def _unique(items: Iterable[Path], label: str) -> Path:
    values = sorted({item.resolve() for item in items})
    if len(values) != 1:
        listing = "\n".join(f"  - {item}" for item in values) or "  (không tìm thấy)"
        raise RuntimeError(f"Cần đúng một {label}, nhưng tìm thấy {len(values)}:\n{listing}")
    return values[0]


def find_stvqa_manifest(input_root: Path) -> Path:
    candidates = []
    for manifest in input_root.rglob("final_main_manifest.csv"):
        try:
            columns = set(pd.read_csv(manifest, nrows=1).columns)
        except Exception:
            continue
        if {"source_row_index", "source_split", "image_id", "question_id"}.issubset(columns):
            degraded = manifest.parent / "images/realistic_mix/L2"
            if degraded.is_dir(): candidates.append(manifest)
    return _unique(candidates, "ST-VQA final_main_manifest.csv kèm images/realistic_mix/L2")


def ensure_stvqa_clean(input_root: Path, manifest: Path, *, offline: bool = False) -> Path:
    """Materialize the exact clean ST-VQA rows from a pinned public HF snapshot."""
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required = {"image_id", "source_row_index", "question", "answers_json"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"ST-VQA manifest thiếu cột: {sorted(required - set(frame.columns))}")
    target = input_root / "stvqa_clean_from_huggingface" / "stvqa_source_v1" / "images"
    marker = target.parent / "materialization_report.json"
    manifest_hash = sha256_file(manifest)
    unique = frame[["image_id", "source_row_index"]].drop_duplicates()
    if marker.is_file():
        report = json.loads(marker.read_text(encoding="utf-8"))
        if report.get("manifest_sha256") == manifest_hash and report.get("images") == len(unique):
            missing = [image_id for image_id in unique.image_id if not (target / f"{image_id}.png").is_file()]
            if not missing: return target
    if offline:
        raise FileNotFoundError("Thiếu ảnh ST-VQA sạch đã materialize; bỏ --offline một lần để tải snapshot công khai")
    from datasets import load_dataset
    print(f"[ST-VQA SẠCH] Đang tải snapshot {STVQA_HF_DATASET}@{STVQA_HF_REVISION} ...", flush=True)
    dataset = load_dataset(STVQA_HF_DATASET, split="train", revision=STVQA_HF_REVISION,
                           cache_dir=str(input_root / ".huggingface_cache"))
    target.mkdir(parents=True, exist_ok=True)
    by_index = {int(key): group for key, group in frame.groupby(frame.source_row_index.astype(int), sort=True)}
    if max(by_index) >= len(dataset):
        raise RuntimeError(f"source_row_index vượt kích thước ST-VQA snapshot: {max(by_index)} >= {len(dataset)}")
    for completed, (source_index, group) in enumerate(by_index.items(), 1):
        row = dataset[source_index]
        qas = row.get("qas") or []
        source_pairs = {(str(item.get("question", "")), tuple(str(x) for x in item.get("answers", []))) for item in qas}
        for record in group.to_dict("records"):
            expected = (record["question"], tuple(str(x) for x in json.loads(record["answers_json"])))
            if expected not in source_pairs:
                raise RuntimeError(f"ST-VQA clean mapping mismatch tại source_row_index={source_index}, question_id={record['question_id']}")
        image_ids = group.image_id.unique().tolist()
        if len(image_ids) != 1:
            raise RuntimeError(f"Một source_row_index ánh xạ nhiều image_id: {source_index} -> {image_ids}")
        destination = target / f"{image_ids[0]}.png"
        if not destination.is_file():
            row["image"].convert("RGB").save(destination, format="PNG", compress_level=1)
        if completed % 250 == 0:
            print(f"[ST-VQA SẠCH] {completed}/{len(by_index)} ảnh", flush=True)
    atomic_write_json(marker, {"status": "COMPLETE", "dataset": STVQA_HF_DATASET,
        "revision": STVQA_HF_REVISION, "manifest_sha256": manifest_hash,
        "images": len(unique), "questions": len(frame), "image_root": str(target.resolve())})
    return target


def _find_clean_root(input_root: Path, phase1_root: Path, degraded_root: Path,
                     image_ids: set[str], label: str) -> Path:
    candidates: dict[Path, set[str]] = {}
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        for path in input_root.rglob(suffix):
            if path.parent == degraded_root or phase1_root in path.parents:
                continue
            if path.stem in image_ids:
                candidates.setdefault(path.parent.resolve(), set()).add(path.stem)
    complete = [root for root, found in candidates.items() if found == image_ids]
    return _unique(complete, label)


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

    textvqa_benchmarks = []
    stvqa_benchmarks = []
    for manifest in input_root.rglob("final_main_manifest.csv"):
        degraded = manifest.parent / "images/realistic_mix/L2"
        if not degraded.is_dir(): continue
        columns = set(pd.read_csv(manifest, nrows=1).columns)
        if {"source_row_index", "source_split"}.issubset(columns): stvqa_benchmarks.append(manifest)
        else: textvqa_benchmarks.append(manifest)
    benchmark_manifest = _unique(textvqa_benchmarks, "TextVQA final_main_manifest.csv kèm images/realistic_mix/L2")
    stvqa_manifest = _unique(stvqa_benchmarks, "ST-VQA final_main_manifest.csv kèm images/realistic_mix/L2")
    degraded_root = benchmark_manifest.parent / "images/realistic_mix/L2"
    stvqa_degraded_root = stvqa_manifest.parent / "images/realistic_mix/L2"

    frame = pd.read_csv(benchmark_manifest, dtype=str, keep_default_na=False)
    if not {"image_id", "question_id", "question", "answers_json"}.issubset(frame.columns):
        raise RuntimeError(f"Manifest benchmark thiếu cột bắt buộc: {benchmark_manifest}")
    image_ids = set(frame.image_id.astype(str))
    clean_root = _find_clean_root(input_root, phase1_root, degraded_root, image_ids,
                                  f"thư mục ảnh TextVQA sạch chứa đủ {len(image_ids)} ảnh benchmark")

    if len(frame) != 1000 or frame.image_id.nunique() != 1000 or frame.question_id.nunique() != 1000:
        raise RuntimeError("Benchmark phải có đúng 1.000 image_id và 1.000 question_id duy nhất")
    stvqa = pd.read_csv(stvqa_manifest, dtype=str, keep_default_na=False)
    stvqa_ids = set(stvqa.image_id.astype(str))
    if len(stvqa) < 1000 or stvqa.question_id.nunique() != len(stvqa):
        raise RuntimeError("ST-VQA benchmark phải có question_id duy nhất và ít nhất 1.000 câu hỏi")
    canonical_stvqa_clean = input_root / "stvqa_clean_from_huggingface" / "stvqa_source_v1" / "images"
    canonical_ids = {path.stem for path in canonical_stvqa_clean.glob("*.png")} if canonical_stvqa_clean.is_dir() else set()
    stvqa_clean_root = (canonical_stvqa_clean.resolve() if canonical_ids == stvqa_ids else
        _find_clean_root(input_root, phase1_root, stvqa_degraded_root, stvqa_ids,
                         f"thư mục ảnh ST-VQA sạch chứa đủ {len(stvqa_ids)} ảnh"))
    return {
        "input_root": input_root,
        "phase1_root": phase1_root,
        "benchmark_manifest": benchmark_manifest,
        "clean_root": clean_root,
        "degraded_root": degraded_root,
        "stvqa_manifest": stvqa_manifest,
        "stvqa_clean_root": stvqa_clean_root,
        "stvqa_degraded_root": stvqa_degraded_root,
    }


def describe_inputs(paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}
