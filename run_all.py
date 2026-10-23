#!/usr/bin/env python3
"""Tệp duy nhất cần chạy trên server: ``python run_all.py``.

Môi trường phải được cài trước bằng ``python -m pip install -r requirements.txt``.
Tệp này chỉ xác định đường dẫn, kiểm tra môi trường và gọi nguyên pipeline
Giai đoạn 2–7; nó không cài hoặc thay đổi thư viện trong lúc thí nghiệm chạy.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
STATUS_PATH = OUTPUT_ROOT / "bootstrap_status.json"

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Giữ thông báo tiếng Việt đọc được khi chạy từ PowerShell/cmd trên Windows.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _write_status(stage: str, **extra: object) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "repository": str(REPOSITORY_ROOT),
        "git_commit": _git_commit(),
        "output_root": str(OUTPUT_ROOT),
        **extra,
    }
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _contains_dataset_anchors(root: Path) -> bool:
    """Nhận diện thư mục cha chứa đủ đầu vào mà không phụ thuộc tên thư mục con."""
    if not root.is_dir():
        return False
    phase1 = False
    textvqa = False
    stvqa = False
    try:
        for manifest in root.rglob("train_manifest.csv"):
            parent = manifest.parent
            if (parent / "dev_manifest.csv").is_file() and (parent / "data/train/clean").is_dir():
                phase1 = True
                break
        for manifest in root.rglob("final_main_manifest.csv"):
            if not (manifest.parent / "images/realistic_mix/L2").is_dir():
                continue
            header = manifest.open("r", encoding="utf-8-sig", errors="replace").readline()
            if "source_row_index" in header and "source_split" in header:
                stvqa = True
            else:
                textvqa = True
            if textvqa and stvqa:
                break
    except OSError:
        return False
    return phase1 and textvqa and stvqa


def _resolve_input_root() -> Path:
    """Tìm thư mục cha dữ liệu; nếu chưa có thì dùng ``<repository>/data`` để tải."""
    configured = os.environ.get("TEXTVQA_INPUT_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"TEXTVQA_INPUT_ROOT không tồn tại: {root}")
        return root

    candidates = [
        REPOSITORY_ROOT / "data",
        REPOSITORY_ROOT.parent / "data",
        REPOSITORY_ROOT.parent / "input",
        REPOSITORY_ROOT.parent / "datasets",
        REPOSITORY_ROOT.parent,
    ]
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique_candidates:
            unique_candidates.append(resolved)
    for candidate in unique_candidates:
        if _contains_dataset_anchors(candidate):
            return candidate
    return (REPOSITORY_ROOT / "data").resolve()


def _check_runtime() -> dict[str, object]:
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError(
            f"Python không tương thích: {sys.version.split()[0]}. "
            "Hãy dùng Python 3.10, 3.11 hoặc 3.12 theo requirements.txt."
        )
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Thiếu mã nguồn package: {SOURCE_ROOT}")

    requirements_path = REPOSITORY_ROOT / "requirements.txt"
    mismatches: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        package, required = (part.strip() for part in line.split("==", 1))
        try:
            installed = version(package)
        except PackageNotFoundError:
            mismatches.append(f"{package}: chưa cài (cần {required})")
            continue
        # Chấp nhận hậu tố CUDA/local build, ví dụ torch 2.6.0+cu124.
        if installed.split("+", 1)[0] != required.split("+", 1)[0]:
            mismatches.append(f"{package}: đang có {installed}, cần {required}")
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Môi trường không khớp requirements.txt:\n  - " + details + "\n"
            "Hãy kích hoạt đúng venv rồi chạy: "
            f"{sys.executable} -m pip install -r {requirements_path}"
        )

    sys.path.insert(0, str(SOURCE_ROOT))
    try:
        import torch
        import transformers
        import qwen_vl_utils  # noqa: F401
        import accelerate
    except ImportError as error:
        raise RuntimeError(
            "Thiếu thư viện chạy. Trong môi trường Python hiện tại, chạy: "
            f"{sys.executable} -m pip install -r {REPOSITORY_ROOT / 'requirements.txt'}"
        ) from error

    devices: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": properties.name,
            "memory_gib": round(properties.total_memory / (1024 ** 3), 2),
            "compute_capability": f"{properties.major}.{properties.minor}",
        })
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
        "devices": devices,
    }


def main() -> int:
    os.chdir(REPOSITORY_ROOT)
    input_root = _resolve_input_root()
    print(f"[KHỞI ĐỘNG] Repository: {REPOSITORY_ROOT}", flush=True)
    print(f"[KHỞI ĐỘNG] Git commit: {_git_commit()}", flush=True)
    print(f"[KHỞI ĐỘNG] Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"[KHỞI ĐỘNG] Dữ liệu: {input_root}", flush=True)
    print(f"[KHỞI ĐỘNG] Kết quả: {OUTPUT_ROOT}", flush=True)
    _write_status("BOOTSTRAP_STARTED", input_root=str(input_root))

    profile = _check_runtime()
    print(f"[PHẦN CỨNG] {json.dumps(profile, ensure_ascii=False)}", flush=True)
    _write_status("PIPELINE_RUNNING", input_root=str(input_root), server_profile=profile)

    from textvqa_feature_restore.pipeline import main as run_pipeline

    # Chỉ hoàn chỉnh đường dẫn từ thư mục cha đã dò; không đổi phase hoặc tham số thí nghiệm.
    return run_pipeline([
        "--input-root", str(input_root),
        "--output-root", str(OUTPUT_ROOT),
        "--no-download-data",
    ])


if __name__ == "__main__":
    try:
        exit_code = main()
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
