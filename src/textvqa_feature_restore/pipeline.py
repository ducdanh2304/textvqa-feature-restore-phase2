from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import pandas as pd

from .config import load_config, write_resolved_config
from .input_discovery import describe_inputs, discover_inputs, ensure_stvqa_clean, find_stvqa_manifest
from .io_utils import atomic_write_json


DATASETS = {
    "phase1": "embechuangoan/textvqa-feature-restore-v1",
    "degraded": "anhthu128/data-suy-thoai-text-vqa",
    "clean": "dmytruto/textvqa",
    "stvqa_degraded": "anhthu128/st-vqa-suy-thoi",
}

# Các giá trị mặc định để người chạy không phải truyền tham số dòng lệnh.
# Tất cả dữ liệu và kết quả nằm ngay trong thư mục repository.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPOSITORY_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
DEFAULT_PHASES = "2,3,4,5,6,7"
DEFAULT_PHASE7_EPOCHS = 3


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Chạy một lệnh cho thí nghiệm phục hồi feature TextVQA, Giai đoạn 2–7")
    value.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help=argparse.SUPPRESS)
    value.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help=argparse.SUPPRESS)
    value.add_argument("--model-path", help="Qwen2.5-VL-3B-Instruct cục bộ; bỏ trống để tải từ Hugging Face")
    value.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--no-download-data", dest="download_data", action="store_false", help=argparse.SUPPRESS)
    value.set_defaults(download_data=True)
    value.add_argument("--phases", default=DEFAULT_PHASES, help=argparse.SUPPRESS)
    value.add_argument("--allow-non-h200", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--phase7-epochs", type=int, default=DEFAULT_PHASE7_EPOCHS, help=argparse.SUPPRESS)
    return value


def _download(input_root: Path) -> None:
    """Tải ẩn danh các Kaggle Dataset công khai còn thiếu vào thư mục chuẩn."""
    import kagglehub

    input_root.mkdir(parents=True, exist_ok=True)
    for name, slug in DATASETS.items():
        destination = input_root / name
        existing = _find_existing_dataset(input_root, name)
        if existing is not None:
            print(f"[DỮ LIỆU] Giữ dataset đã có: {existing}", flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        print(f"[DỮ LIỆU] Tải công khai {slug} -> {destination}", flush=True)
        try:
            kagglehub.dataset_download(slug, output_dir=str(destination))
        except Exception as error:
            raise RuntimeError(
                f"Không tải được Kaggle Dataset công khai {slug}. "
                "Kiểm tra Internet và bảo đảm Dataset đang ở chế độ Public."
            ) from error
        if _find_existing_dataset(input_root, name) is None:
            raise RuntimeError(
                f"Đã tải {slug} nhưng cấu trúc dữ liệu không hợp lệ tại {destination}."
            )


def _find_existing_dataset(input_root: Path, name: str) -> Path | None:
    """Tìm dataset theo cấu trúc bên trong, không phụ thuộc tên hay số tầng thư mục."""
    if name == "phase1":
        for manifest in input_root.rglob("train_manifest.csv"):
            root = manifest.parent
            required = [root / "dev_manifest.csv", root / "data/train/clean",
                        root / "data/train/degraded", root / "data/dev/clean",
                        root / "data/dev/degraded"]
            if all(path.exists() for path in required):
                return root.resolve()
        return None
    if name in {"degraded", "stvqa_degraded"}:
        wants_stvqa = name == "stvqa_degraded"
        for manifest in input_root.rglob("final_main_manifest.csv"):
            if not (manifest.parent / "images/realistic_mix/L2").is_dir():
                continue
            try:
                columns = set(pd.read_csv(manifest, nrows=1).columns)
            except Exception:
                continue
            is_stvqa = {"source_row_index", "source_split"}.issubset(columns)
            if is_stvqa == wants_stvqa:
                return manifest.parent.resolve()
        return None
    if name == "clean":
        for directory in input_root.rglob("train_images"):
            has_image = any(next(directory.glob(pattern), None) is not None
                            for pattern in ("*.jpg", "*.jpeg", "*.png"))
            if directory.is_dir() and has_image:
                return directory.resolve()
        return None
    raise ValueError(f"Tên dataset không được hỗ trợ: {name}")


def _phase2(cfg: dict[str, Any], destination: Path) -> dict[str, Any]:
    from .cli import run_bundle, run_verify
    from .feature_cache import run_cache
    from .trainer import run_training
    from .validation import run_preflight
    destination.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["output_root"] = str(destination.resolve())
    write_resolved_config(cfg, destination / "phase2_config_resolved.yaml")
    preflight = destination / "preflight_report.json"
    if not preflight.is_file() or json.loads(preflight.read_text(encoding="utf-8")).get("status") != "READY":
        run_preflight(cfg)
    cache_summary = destination / "cache/cache_summary.json"
    cache_ready = False
    if cache_summary.is_file():
        cached = json.loads(cache_summary.read_text(encoding="utf-8"))
        cache_ready = cached.get("total_cached") == 4400 and cached.get("failures") == 0
    if not cache_ready: run_cache(cfg)
    best = destination / "training/best_adapter.pt"
    summary = destination / "training/training_summary.json"
    if not (best.is_file() and summary.is_file()):
        resume = destination / "training/last_checkpoint.pt"
        run_training(cfg, str(resume) if resume.is_file() else None)
    verify = destination / "verification/checkpoint_verification.json"
    if not verify.is_file(): run_verify(cfg, str(best))
    archive = Path(str(destination) + ".zip")
    if not archive.is_file(): run_bundle(cfg)
    return json.loads(summary.read_text(encoding="utf-8"))


def _need(path: Path, explanation: str) -> Path:
    if not path.is_file(): raise FileNotFoundError(f"{explanation}: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    input_root, output_root = Path(args.input_root).expanduser().resolve(), Path(args.output_root).expanduser().resolve()
    print(f"[MẶC ĐỊNH] Dữ liệu: {input_root}", flush=True)
    print(f"[MẶC ĐỊNH] Kết quả: {output_root}", flush=True)
    print(f"[MẶC ĐỊNH] Giai đoạn: {args.phases}", flush=True)
    if args.download_data: _download(input_root)
    stvqa_manifest = find_stvqa_manifest(input_root)
    ensure_stvqa_clean(input_root, stvqa_manifest, offline=bool(args.offline))
    inputs = discover_inputs(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "resolved_inputs.json", describe_inputs(inputs))
    phases = [int(value.strip()) for value in args.phases.split(",") if value.strip()]
    if any(value not in {2, 3, 4, 5, 6, 7} for value in phases): raise ValueError("--phases chỉ nhận các số từ 2 đến 7")
    if phases != sorted(set(phases)): raise ValueError("--phases phải tăng dần và không lặp")
    if not torch.cuda.is_available(): raise RuntimeError("Không tìm thấy GPU CUDA")
    print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)
    cfg = load_config(Path(__file__).parents[2] / "configs/phase2_h200.yaml",
                      data_root=str(input_root), output_root=str(output_root / "phase2_training"),
                      model_path=args.model_path, offline=args.offline or None)
    cfg["paths"]["phase1_root"] = str(inputs["phase1_root"])
    cfg["hardware"]["allow_non_h200"] = bool(args.allow_non_h200)
    state_path = output_root / "pipeline_state.json"
    state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"status": "RUNNING", "phases": {}}

    from .research_training import create_feature_diagnostics, train_answer_phase, train_feature_phase
    from .vqa_evaluation import evaluate_vqa
    p2 = output_root / "phase2_training"; p3 = output_root / "phase3_baseline_evaluation"
    p4 = output_root / "phase4_identity"; p5 = output_root / "phase5_token_gate"
    p6 = output_root / "phase6_question_gate"; p7 = output_root / "phase7_answer_loss"

    def evaluate_both(phase: int, checkpoint: Path, destination: Path,
                      text_base: Path | None = None, stvqa_base: Path | None = None) -> dict[str, Any]:
        textvqa = evaluate_vqa(phase=phase, cfg=cfg, checkpoint=checkpoint,
            benchmark_manifest=inputs["benchmark_manifest"], clean_root=inputs["clean_root"],
            degraded_root=inputs["degraded_root"], destination=destination / "textvqa",
            base_predictions=text_base, benchmark_name="textvqa")
        stvqa = evaluate_vqa(phase=phase, cfg=cfg, checkpoint=checkpoint,
            benchmark_manifest=inputs["stvqa_manifest"], clean_root=inputs["stvqa_clean_root"],
            degraded_root=inputs["stvqa_degraded_root"], destination=destination / "stvqa",
            base_predictions=stvqa_base, benchmark_name="stvqa")
        return {"textvqa": textvqa, "stvqa": stvqa}
    try:
        if 2 in phases:
            print("\n=== GIAI ĐOẠN 2: HUẤN LUYỆN ADAPTER FEATURE ===", flush=True)
            state["phases"]["2"] = _phase2(cfg, p2); atomic_write_json(state_path, state)
        phase2_checkpoint = _need(p2 / "training/best_adapter.pt", "Cần chạy Giai đoạn 2 trước")
        phase2_features = create_feature_diagnostics(
            phase=2, cfg=cfg, phase1_root=inputs["phase1_root"], checkpoint=phase2_checkpoint,
            destination=p2 / "visualizations",
        )
        state["phases"].setdefault("2", {})["feature_diagnostics"] = phase2_features
        atomic_write_json(state_path, state)
        if 3 in phases:
            print("\n=== GIAI ĐOẠN 3: ĐÁNH GIÁ TEXTVQA + ST-VQA ===", flush=True)
            state["phases"]["3"] = evaluate_both(3, phase2_checkpoint, p3)
            atomic_write_json(state_path, state)
        text_base = _need(p3 / "textvqa/predictions.csv", "Cần chạy đánh giá TextVQA Giai đoạn 3 trước")
        stvqa_base = _need(p3 / "stvqa/predictions.csv", "Cần chạy đánh giá ST-VQA Giai đoạn 3 trước")
        previous = phase2_checkpoint
        for phase, destination in ((4, p4), (5, p5), (6, p6)):
            if phase not in phases:
                previous = _need(destination / "best_model.pt", f"Cần có checkpoint Giai đoạn {phase}")
                continue
            title = {4: "IDENTITY LOSS", 5: "TOKEN GATE", 6: "QUESTION-GUIDED GATE"}[phase]
            print(f"\n=== GIAI ĐOẠN {phase}: {title} ===", flush=True)
            training = train_feature_phase(phase=phase, cfg=cfg, phase1_root=inputs["phase1_root"],
                                           source_checkpoint=previous, destination=destination)
            previous = destination / "best_model.pt"
            features = create_feature_diagnostics(
                phase=phase, cfg=cfg, phase1_root=inputs["phase1_root"], checkpoint=previous,
                destination=destination / "visualizations",
            )
            evaluation = evaluate_both(phase, previous, destination / "evaluation", text_base, stvqa_base)
            state["phases"][str(phase)] = {"training": training, "feature_diagnostics": features,
                                           "evaluation": evaluation}; atomic_write_json(state_path, state)
        if 7 in phases:
            previous = _need(p6 / "best_model.pt", "Cần chạy Giai đoạn 6 trước")
            print("\n=== GIAI ĐOẠN 7: TỐI ƯU TRỰC TIẾP CÂU TRẢ LỜI VQA ===", flush=True)
            training = train_answer_phase(cfg=cfg, phase1_root=inputs["phase1_root"], source_checkpoint=previous,
                                          destination=p7, max_epochs=args.phase7_epochs)
            features = create_feature_diagnostics(
                phase=7, cfg=cfg, phase1_root=inputs["phase1_root"], checkpoint=p7 / "best_model.pt",
                destination=p7 / "visualizations",
            )
            evaluation = evaluate_both(7, p7 / "best_model.pt", p7 / "evaluation", text_base, stvqa_base)
            state["phases"]["7"] = {"training": training, "feature_diagnostics": features,
                                      "evaluation": evaluation}; atomic_write_json(state_path, state)
        state["status"] = "COMPLETE"; atomic_write_json(state_path, state)
        atomic_write_json(output_root / "final_summary.json", state)
        print(f"\nHOÀN TẤT. Xem kết quả tại: {output_root}", flush=True)
        return 0
    except Exception as error:
        state["status"] = "FAILED"; state["error"] = f"{type(error).__name__}: {error}"; atomic_write_json(state_path, state)
        raise
