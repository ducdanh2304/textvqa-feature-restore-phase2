#!/usr/bin/env python3
"""Điểm chạy duy nhất cho toàn bộ thí nghiệm Giai đoạn 2–7."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _inside_project_environment() -> bool:
    try:
        import textvqa_feature_restore.pipeline  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _prepare_environment() -> None:
    venv = ROOT / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-e", str(ROOT)], check=True)
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


if not _inside_project_environment():
    _prepare_environment()

from textvqa_feature_restore.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
