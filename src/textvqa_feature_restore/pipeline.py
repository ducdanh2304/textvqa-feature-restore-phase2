from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .config import load_config, write_resolved_config
from .input_discovery import describe_inputs, discover_inputs
from .io_utils import atomic_write_json


DATASETS = {
    "phase1": "embechuangoan/textvqa-feature-restore-v1",
    "degraded": "anhthu128/data-suy-thoai-text-vqa",
    "clean": "dmytruto/textvqa",
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Chạy một lệnh cho thí nghiệm phục hồi feature TextVQA, Giai đoạn 2–7")
    value.add_argument("--input-root", required=True, help="Thư mục chứa ba dataset đã giải nén")
    value.add_argument("--output-root", required=True, help="Thư mục nhận toàn bộ kết quả")
    value.add_argument("--model-path", help="Qwen2.5-VL-3B-Instruct cục bộ; bỏ trống để tải từ Hugging Face")
    value.add_argument("--offline", action="store_true", help="Cấm tải mạng; yêu cầu --model-path")
    value.add_argument("--download-data", action="store_true", help="Tự tải và giải nén ba dataset bằng tài khoản Kaggle")
    value.add_argument("--phases", default="2,3,4,5,6,7", help="Các giai đoạn cần chạy, mặc định 2,3,4,5,6,7")
    value.add_argument("--allow-non-h200", action="store_true", help="Cho phép GPU khác H200 để chạy thử")
    value.add_argument("--phase7-epochs", type=int, default=3)
    return value


def _download(input_root: Path) -> None:
    input_root.mkdir(parents=True, exist_ok=True)
    for name, slug in DATASETS.items():
        destination = input_root / name
        if destination.is_dir() and any(destination.iterdir()):
            print(f"[DỮ LIỆU] Giữ thư mục đã có: {destination}", flush=True); continue
        destination.mkdir(parents=True, exist_ok=True)
        print(f"[DỮ LIỆU] Đang tải {slug} ...", flush=True)
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", slug,
                        "--path", str(destination), "--unzip"], check=True)


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
    if args.download_data: _download(input_root)
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

    from .research_training import train_answer_phase, train_feature_phase
    from .vqa_evaluation import evaluate_vqa
    p2 = output_root / "phase2_training"; p3 = output_root / "phase3_baseline_evaluation"
    p4 = output_root / "phase4_identity"; p5 = output_root / "phase5_token_gate"
    p6 = output_root / "phase6_question_gate"; p7 = output_root / "phase7_answer_loss"
    try:
        if 2 in phases:
            print("\n=== GIAI ĐOẠN 2: HUẤN LUYỆN ADAPTER FEATURE ===", flush=True)
            state["phases"]["2"] = _phase2(cfg, p2); atomic_write_json(state_path, state)
        phase2_checkpoint = _need(p2 / "training/best_adapter.pt", "Cần chạy Giai đoạn 2 trước")
        if 3 in phases:
            print("\n=== GIAI ĐOẠN 3: ĐÁNH GIÁ VQA BASE/ADAPTER ===", flush=True)
            state["phases"]["3"] = evaluate_vqa(phase=3, cfg=cfg, checkpoint=phase2_checkpoint,
                benchmark_manifest=inputs["benchmark_manifest"], clean_root=inputs["clean_root"],
                degraded_root=inputs["degraded_root"], destination=p3)
            atomic_write_json(state_path, state)
        base_predictions = _need(p3 / "predictions.csv", "Cần chạy Giai đoạn 3 trước")
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
            evaluation = evaluate_vqa(phase=phase, cfg=cfg, checkpoint=previous,
                benchmark_manifest=inputs["benchmark_manifest"], clean_root=inputs["clean_root"],
                degraded_root=inputs["degraded_root"], destination=destination / "evaluation",
                base_predictions=base_predictions)
            state["phases"][str(phase)] = {"training": training, "evaluation": evaluation}; atomic_write_json(state_path, state)
        if 7 in phases:
            previous = _need(p6 / "best_model.pt", "Cần chạy Giai đoạn 6 trước")
            print("\n=== GIAI ĐOẠN 7: TỐI ƯU TRỰC TIẾP CÂU TRẢ LỜI VQA ===", flush=True)
            training = train_answer_phase(cfg=cfg, phase1_root=inputs["phase1_root"], source_checkpoint=previous,
                                          destination=p7, max_epochs=args.phase7_epochs)
            evaluation = evaluate_vqa(phase=7, cfg=cfg, checkpoint=p7 / "best_model.pt",
                benchmark_manifest=inputs["benchmark_manifest"], clean_root=inputs["clean_root"],
                degraded_root=inputs["degraded_root"], destination=p7 / "evaluation",
                base_predictions=base_predictions)
            state["phases"]["7"] = {"training": training, "evaluation": evaluation}; atomic_write_json(state_path, state)
        state["status"] = "COMPLETE"; atomic_write_json(state_path, state)
        atomic_write_json(output_root / "final_summary.json", state)
        print(f"\nHOÀN TẤT. Xem kết quả tại: {output_root}", flush=True)
        return 0
    except Exception as error:
        state["status"] = "FAILED"; state["error"] = f"{type(error).__name__}: {error}"; atomic_write_json(state_path, state)
        raise
