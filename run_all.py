#!/usr/bin/env python
"""Tệp duy nhất cần chạy trên server: python /datastore/dungmt/textvqa/run_all.py."""
from __future__ import annotations

import os
import importlib.util
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
STATUS_PATH = OUTPUT_ROOT / "bootstrap_status.json"

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

RUNTIME_PACKAGES = {
    "torch": "torch==2.6.0",
    "transformers": "transformers==4.57.6",
    "qwen_vl_utils": "qwen-vl-utils==0.0.14",
    "accelerate": "accelerate==1.10.1",
    "huggingface_hub": "huggingface-hub==0.36.0",
    "numpy": "numpy==2.1.3",
    "pandas": "pandas==2.2.3",
    "PIL": "Pillow==11.3.0",
    "yaml": "PyYAML==6.0.2",
    "matplotlib": "matplotlib==3.10.3",
    "tqdm": "tqdm==4.67.1",
    "rapidfuzz": "rapidfuzz==3.14.1",
    "kaggle": "kaggle==1.7.4.5",
    "datasets": "datasets==3.6.0",
}


def _write_status(stage: str, **extra: object) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "repository": str(ROOT),
        "output_root": str(OUTPUT_ROOT),
        **extra,
    }
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _missing_packages() -> list[str]:
    return [requirement for module, requirement in RUNTIME_PACKAGES.items()
            if importlib.util.find_spec(module) is None]


def _install_into_active_environment() -> None:
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Chưa kích hoạt môi trường. Chạy trước: "
            "source /datastore/dungmt/venv/bin/activate"
        )
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError(f"Cần Python 3.10, 3.11 hoặc 3.12; hiện tại là {sys.version.split()[0]}")
    missing = _missing_packages()
    print(f"[MÔI TRƯỜNG] Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"[MÔI TRƯỜNG] Môi trường: {sys.prefix}", flush=True)
    print(f"[MÔI TRƯỜNG] Output sẽ nằm tại: {OUTPUT_ROOT}", flush=True)
    _write_status("INSTALLING", missing_packages=missing)
    if missing:
        print(f"[MÔI TRƯỜNG] Chỉ cài các thư viện còn thiếu: {missing}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "--no-input", *missing], check=True)
    # Cài chính repository nhưng không yêu cầu pip giải lại toàn bộ dependency.
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--no-input", "--no-deps", "-e", str(ROOT)], check=True)
    _write_status("ENVIRONMENT_READY", installed_missing_packages=missing)
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())])


print(f"[KHỞI ĐỘNG] Repository: {ROOT}", flush=True)
print(f"[KHỞI ĐỘNG] Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
print(f"[KHỞI ĐỘNG] Kết quả: {OUTPUT_ROOT}", flush=True)
_write_status("BOOTSTRAP_STARTED")

if _missing_packages() or importlib.util.find_spec("textvqa_feature_restore") is None:
    _install_into_active_environment()

if __name__ == "__main__":
    try:
        from textvqa_feature_restore.pipeline import main
        _write_status("PIPELINE_RUNNING")
        exit_code = main([])
        _write_status("COMPLETE", exit_code=exit_code)
        raise SystemExit(exit_code)
    except SystemExit:
        raise
    except Exception as error:
        _write_status("FAILED", error_type=type(error).__name__, error=str(error))
        print("\n[THẤT BẠI] Pipeline gặp lỗi:", flush=True)
        traceback.print_exc(file=sys.stdout)
        print(f"[THẤT BẠI] Trạng thái: {STATUS_PATH}", flush=True)
        raise SystemExit(1)
