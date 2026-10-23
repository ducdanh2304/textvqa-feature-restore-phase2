from __future__ import annotations

import sys
from types import SimpleNamespace

from textvqa_feature_restore import pipeline


def _materialize(destination):
    name = destination.name
    if name == "phase1":
        for rel in ("data/train/clean", "data/train/degraded", "data/dev/clean", "data/dev/degraded"):
            (destination / rel).mkdir(parents=True, exist_ok=True)
        (destination / "train_manifest.csv").write_text("x\n", encoding="utf-8")
        (destination / "dev_manifest.csv").write_text("x\n", encoding="utf-8")
    elif name in {"degraded", "stvqa_degraded"}:
        main = destination / "main"
        (main / "images/realistic_mix/L2").mkdir(parents=True, exist_ok=True)
        header = "source_row_index,source_split\n" if name == "stvqa_degraded" else "image_id,path\n"
        (main / "final_main_manifest.csv").write_text(header, encoding="utf-8")
    elif name == "clean":
        images = destination / "train_images"
        images.mkdir(parents=True, exist_ok=True)
        (images / "sample.jpg").write_bytes(b"image")


def test_public_kaggle_download_uses_fixed_directories_without_credentials(tmp_path, monkeypatch):
    calls = []

    def dataset_download(handle, *, output_dir, force_download):
        assert force_download is True
        destination = tmp_path / output_dir.split(str(tmp_path))[-1].lstrip("/\\")
        calls.append((handle, destination.name))
        _materialize(destination)
        return str(destination)

    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=dataset_download))
    pipeline._download(tmp_path)

    assert calls == [
        ("embechuangoan/textvqa-feature-restore-v1", "phase1"),
        ("anhthu128/data-suy-thoai-text-vqa", "degraded"),
        ("dmytruto/textvqa", "clean"),
        ("anhthu128/st-vqa-suy-thoi", "stvqa_degraded"),
    ]
