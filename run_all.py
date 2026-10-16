#!/usr/bin/env python
"""Tệp duy nhất cần chạy trên server: python /datastore/dungmt/textvqa/run_all.py."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _dependencies_ready() -> bool:
    try:
        import datasets  # noqa: F401
        import kaggle  # noqa: F401
        import qwen_vl_utils  # noqa: F401
        import textvqa_feature_restore.pipeline  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _install_into_active_environment() -> None:
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Chưa kích hoạt môi trường. Chạy trước: "
            "source /datastore/dungmt/venv/bin/activate"
        )
    print(f"[MÔI TRƯỜNG] Cài thư viện vào: {sys.prefix}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(ROOT)], check=True)
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())])


if not _dependencies_ready():
    _install_into_active_environment()

from textvqa_feature_restore.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main([]))
