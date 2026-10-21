from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torch import nn


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
STVQA_ID = "vikhyatk/st-vqa"
STVQA_REVISION = "3453b8a0359fd2cea7f0c2c5602ba8618af59794"
EVAL_REVISION = "72f898ec35af86567423c7a019dd1dc5175d1a5a"
EVAL_URL = f"https://raw.githubusercontent.com/facebookresearch/mmf/{EVAL_REVISION}/mmf/utils/m4c_evaluators.py"
PROMPT = "Answer the question using the image. Reply with only the short answer, with no explanation. Question: {question}"


class ResidualFeatureRestorer(nn.Module):
    def __init__(self, hidden_size: int = 2048, bottleneck: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, bottleneck), nn.GELU(),
                                 nn.Dropout(0.0), nn.Linear(bottleneck, hidden_size))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.net(feature)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint Phase 2 trên ST-VQA bằng hai GPU Kaggle T4")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stvqa_phase3_evaluation")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--worker-gpu", type=int)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--clean-root", default="")
    parser.add_argument("--degraded-root", default="")
    return parser.parse_args()


def find_stvqa_manifest(root: Path) -> Path:
    found = []
    for path in root.rglob("final_main_manifest.csv"):
        try:
            columns = set(pd.read_csv(path, nrows=1).columns)
        except Exception:
            continue
        image_root = path.parent / "images" / "realistic_mix" / "L2"
        if {"image_id", "question_id", "question", "answers_json", "source_row_index"}.issubset(columns) and image_root.is_dir():
            found.append(path.resolve())
    if len(set(found)) != 1:
        raise RuntimeError(f"Cần đúng một ST-VQA final_main_manifest.csv, tìm thấy: {sorted(set(found))}")
    return found[0]


def find_checkpoint(root: Path, override: str) -> Path:
    if override:
        path = Path(override).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted({path.resolve() for path in root.rglob("best_adapter.pt")})
    if len(candidates) != 1:
        listing = "\n".join(f"  - {path}" for path in candidates) or "  (không tìm thấy)"
        raise RuntimeError("Cần attach đúng một dataset chứa best_adapter.pt. Các file tìm thấy:\n" + listing)
    return candidates[0]


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("adapter_state_dict", "model_state_dict", "adapter"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            return payload[key]
    if isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise RuntimeError(f"Không nhận ra cấu trúc checkpoint: {path}")


def select_manifest(source: Path, destination: Path, sample_size: int, seed: int) -> pd.DataFrame:
    required = {"image_id", "question_id", "question", "answers_json", "source_row_index"}
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Manifest thiếu cột: {sorted(missing)}")
    if frame.question_id.duplicated().any():
        raise RuntimeError("question_id trong ST-VQA manifest không duy nhất")
    n = min(sample_size, len(frame))
    selected = frame.sample(n=n, random_state=seed).sort_values("question_id").reset_index(drop=True)
    selected.insert(0, "sample_index", np.arange(len(selected)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination, index=False)
    return selected


def materialize_clean(selected: pd.DataFrame, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    missing = selected[~selected.image_id.map(lambda value: (destination / f"{value}.png").is_file())]
    if missing.empty:
        return destination
    from datasets import load_dataset
    print(f"[ST-VQA sạch] Tải snapshot khóa {STVQA_ID}@{STVQA_REVISION}", flush=True)
    dataset = load_dataset(STVQA_ID, split="train", revision=STVQA_REVISION,
                           cache_dir="/kaggle/working/huggingface_cache")
    groups = missing.groupby(missing.source_row_index.astype(int), sort=True)
    for completed, (source_index, group) in enumerate(groups, 1):
        if source_index >= len(dataset):
            raise RuntimeError(f"source_row_index={source_index} vượt kích thước ST-VQA sạch")
        source = dataset[source_index]
        available = {(str(item.get("question", "")), tuple(str(value) for value in item.get("answers", [])))
                     for item in (source.get("qas") or [])}
        for record in group.to_dict("records"):
            expected = (record["question"], tuple(str(value) for value in json.loads(record["answers_json"])))
            if expected not in available:
                raise RuntimeError(f"Ghép sai ST-VQA sạch tại question_id={record['question_id']}")
        image_ids = group.image_id.unique().tolist()
        if len(image_ids) != 1:
            raise RuntimeError(f"Một source_row_index ánh xạ nhiều image_id: {image_ids}")
        source["image"].convert("RGB").save(destination / f"{image_ids[0]}.png", compress_level=1)
        if completed % 100 == 0:
            print(f"[ST-VQA sạch] {completed}/{len(groups)} ảnh", flush=True)
    return destination


def find_image(root: Path, image_id: str) -> Path:
    found = [root / f"{image_id}{suffix}" for suffix in (".png", ".jpg", ".jpeg")]
    found = [path for path in found if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Cần đúng một ảnh {image_id} trong {root}, tìm thấy {found}")
    return found[0]


def worker(args: argparse.Namespace) -> int:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    gpu = int(args.worker_gpu)
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    selected = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    selected = selected[selected.sample_index.astype(int) % args.world_size == gpu]
    destination = Path(args.output_root); shard = destination / "shards" / f"gpu_{gpu}.csv"
    shard.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(shard, dtype=str, keep_default_na=False).to_dict("records") if shard.is_file() else []
    done = {(row["question_id"], row["condition"], row["mode"]) for row in existing}
    rows = list(existing)

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, min_pixels=65536,
                                               max_pixels=786432, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device).eval()
    model.model.visual.to(device=device, dtype=torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = ResidualFeatureRestorer().to(device=device, dtype=torch.float32).eval()
    adapter.load_state_dict(checkpoint_state(Path(args.checkpoint)), strict=True)
    state = {"enabled": False, "calls": 0}

    def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        state["calls"] += 1
        if not state["enabled"]:
            return output
        restored = adapter(output.float())
        if restored.shape != output.shape or not torch.isfinite(restored).all():
            raise RuntimeError("Adapter sinh feature không hợp lệ")
        return restored.to(output.dtype)

    handle = model.model.visual.register_forward_hook(hook)
    conditions = [("degraded", Path(args.degraded_root))]
    if args.clean_root:
        conditions.insert(0, ("clean", Path(args.clean_root)))
    try:
        total = len(selected) * len(conditions) * 2
        completed = 0
        for record in selected.to_dict("records"):
            for condition, image_root in conditions:
                image_path = find_image(image_root, str(record["image_id"]))
                for mode in ("base", "adapter"):
                    key = (record["question_id"], condition, mode)
                    if key in done:
                        completed += 1; continue
                    state["enabled"] = mode == "adapter"
                    messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)},
                                {"type": "text", "text": PROMPT.format(question=record["question"])}]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images, videos = process_vision_info(messages)
                    inputs = processor(text=[text], images=images, videos=videos, padding=True,
                                       return_tensors="pt").to(device)
                    inputs["pixel_values"] = inputs["pixel_values"].float()
                    before = state["calls"]
                    if hasattr(model.model, "rope_deltas"):
                        model.model.rope_deltas = None
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=20)
                    prediction = processor.batch_decode(
                        generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                    if state["calls"] - before != 1 or not prediction:
                        raise RuntimeError(f"Inference lỗi tại question_id={record['question_id']}")
                    rows.append({"sample_index": record["sample_index"], "image_id": record["image_id"],
                                 "question_id": record["question_id"], "question": record["question"],
                                 "answers_json": record["answers_json"], "condition": condition,
                                 "mode": mode, "prediction": prediction})
                    done.add(key); completed += 1
                    pd.DataFrame(rows).to_csv(shard, index=False)
                    if completed % 20 == 0:
                        print(f"[GPU {gpu}] {completed}/{total}", flush=True)
    finally:
        handle.remove()
    return 0


def answer_processor(cache: Path):
    if not cache.is_file():
        with urllib.request.urlopen(EVAL_URL, timeout=90) as response:
            cache.write_bytes(response.read())
    source = cache.read_text(encoding="utf-8"); tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "EvalAIAnswerProcessor")
    namespace: dict[str, Any] = {"re": re}
    exec("import re\n" + ast.get_source_segment(source, node), namespace)
    return namespace["EvalAIAnswerProcessor"]()


def anls(prediction: str, answers: list[str]) -> float:
    pred = str(prediction).lower().strip()
    scores = []
    for answer in answers:
        ref = str(answer).lower().strip()
        distance = Levenshtein.distance(pred, ref) / max(len(pred), len(ref), 1)
        scores.append(1.0 - distance if distance < 0.5 else 0.0)
    return float(max(scores, default=0.0))


def vqa_accuracy(prediction: str, answers: list[str], normalize: Any) -> float:
    pred = normalize(prediction); refs = [normalize(str(value)) for value in answers]
    if not refs:
        return 0.0
    matches = sum(ref == pred for ref in refs)
    return float(np.mean([min((matches - int(ref == pred)) / 3.0, 1.0) for ref in refs]))


def bleu_scores(prediction: str, answers: list[str]) -> list[float]:
    tokenize = lambda value: re.findall(r"\w+|[^\w\s]", str(value).lower().strip(), flags=re.UNICODE)
    hypothesis = tokenize(prediction); references = [tokenize(value) for value in answers if tokenize(value)]
    if not hypothesis or not references:
        return [0.0] * 4
    smooth = SmoothingFunction().method1
    result = []
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        result.append(float(sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smooth)))
    return result


def bootstrap(base: np.ndarray, adapter: np.ndarray, seed: int) -> dict[str, Any]:
    delta = np.asarray(adapter, float) - np.asarray(base, float)
    rng = np.random.default_rng(seed); values = np.empty(10000, dtype=float)
    for index in range(len(values)):
        values[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {"mean_delta": float(delta.mean()), "ci95_low": float(np.quantile(values, .025)),
            "ci95_high": float(np.quantile(values, .975)), "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())}


def score_and_report(args: argparse.Namespace, output: Path, clean_root: Path | None, degraded_root: Path) -> None:
    shards = sorted((output / "shards").glob("gpu_*.csv"))
    if len(shards) != args.world_size:
        raise RuntimeError(f"Thiếu shard dự đoán: {shards}")
    table = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False) for path in shards], ignore_index=True)
    table = table.drop_duplicates(["question_id", "condition", "mode"], keep="last")
    normalize = answer_processor(output / "m4c_evaluators_pinned.py")
    scored = []
    for row in table.to_dict("records"):
        answers = [str(value) for value in json.loads(row["answers_json"])]
        bleu = bleu_scores(row["prediction"], answers)
        exact = float(str(row["prediction"]).lower().strip() in {value.lower().strip() for value in answers})
        scored.append({**row, "exact_match": exact, "vqa_accuracy": vqa_accuracy(row["prediction"], answers, normalize),
                       "anls": anls(row["prediction"], answers), "bleu1": bleu[0], "bleu2": bleu[1],
                       "bleu3": bleu[2], "bleu4": bleu[3]})
    scored_frame = pd.DataFrame(scored).sort_values(["sample_index", "condition", "mode"])
    scored_frame.to_csv(output / "predictions_scored.csv", index=False)
    metrics = ["anls", "vqa_accuracy", "exact_match", "bleu1", "bleu2", "bleu3", "bleu4"]
    arms = {f"{condition}_{mode}": {metric: float(part[metric].mean()) for metric in metrics}
            for (condition, mode), part in scored_frame.groupby(["condition", "mode"])}
    comparisons: dict[str, Any] = {}
    visual = output / "visualizations"; visual.mkdir(exist_ok=True)
    for condition in scored_frame.condition.unique():
        subset = scored_frame[scored_frame.condition == condition]
        metadata = subset[subset["mode"] == "base"][["question_id", "image_id", "question", "answers_json"]]
        wide = subset.pivot(index="question_id", columns="mode", values=["prediction", *metrics])
        wide.columns = [f"{metric}_{mode}" for metric, mode in wide.columns]
        detail = metadata.merge(wide.reset_index(), on="question_id", validate="one_to_one")
        detail["anls_delta"] = detail["anls_adapter"] - detail["anls_base"]
        detail["case"] = np.select([detail.anls_delta > 0, detail.anls_delta < 0],
                                    ["ADAPTER_SUA_DUNG", "ADAPTER_LAM_SAI"], default="KHONG_DOI")
        detail.sort_values("anls_delta", ascending=False).to_csv(output / f"comparison_{condition}.csv", index=False)
        comparisons[condition] = {metric: bootstrap(wide[(metric, "base")].to_numpy(float),
                                                    wide[(metric, "adapter")].to_numpy(float), args.seed)
                                  for metric in metrics}
        if condition == "degraded":
            counts = detail.case.value_counts().reindex(["ADAPTER_SUA_DUNG", "KHONG_DOI", "ADAPTER_LAM_SAI"], fill_value=0)
            plt.figure(figsize=(7, 4.5)); bars = plt.bar(["Sửa đúng", "Không đổi", "Làm sai"], counts.values,
                                                        color=["#66bb6a", "#9e9e9e", "#ef5350"])
            for bar, value in zip(bars, counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, value, str(int(value)), ha="center", va="bottom")
            plt.ylabel("Số câu hỏi"); plt.title("ST-VQA suy thoái — so sánh ANLS")
            plt.tight_layout(); plt.savefig(visual / "win_tie_loss_anls.png", dpi=180); plt.close()
            selected = pd.concat([detail[detail.anls_delta > 0].head(4),
                                  detail[detail.anls_delta < 0].sort_values("anls_delta").head(4)])
            if not selected.empty:
                figure, axes = plt.subplots(len(selected), 3 if clean_root else 2,
                                            figsize=(16 if clean_root else 12, 4.0 * len(selected)), squeeze=False)
                for index, record in enumerate(selected.to_dict("records")):
                    column = 0
                    if clean_root:
                        axes[index, column].imshow(Image.open(find_image(clean_root, str(record["image_id"]))).convert("RGB"))
                        axes[index, column].set_title("Ảnh sạch đối chiếu"); axes[index, column].axis("off"); column += 1
                    axes[index, column].imshow(Image.open(find_image(degraded_root, str(record["image_id"]))).convert("RGB"))
                    axes[index, column].set_title("Ảnh suy thoái — đầu vào thật"); axes[index, column].axis("off"); column += 1
                    axis = axes[index, column]; axis.axis("off")
                    answers = ", ".join(json.loads(record["answers_json"])[:5])
                    message = (f"{record['case']}\n\nCâu hỏi: {record['question']}\nĐáp án: {answers}\n\n"
                               f"Base: {record['prediction_base']}\nANLS: {record['anls_base']:.3f}\n\n"
                               f"Adapter: {record['prediction_adapter']}\nANLS: {record['anls_adapter']:.3f}\n"
                               f"Delta: {record['anls_delta']:+.3f}")
                    color = "#e8f5e9" if record["anls_delta"] > 0 else "#ffebee"
                    axis.text(0, 1, textwrap.fill(message, 60, replace_whitespace=False), va="top",
                              bbox={"boxstyle": "round", "facecolor": color})
                figure.tight_layout(); figure.savefig(visual / "qualitative_examples.png", dpi=160, bbox_inches="tight")
                plt.close(figure)

    summary = {"status": "COMPLETE", "sample_size": int(scored_frame.question_id.nunique()),
               "primary_metric": "anls", "secondary_metrics": metrics[1:], "arms": arms,
               "paired_bootstrap": comparisons, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "checkpoint": str(Path(args.checkpoint).resolve()),
               "metric_notes": {"anls": "max normalized Levenshtein similarity; distance >= 0.5 nhận 0",
                                  "vqa_accuracy": "Official VQA leave-one-out consensus sau EvalAI normalization",
                                  "bleu": "sentence BLEU-n đa tham chiếu, NLTK smoothing method1; lấy trung bình theo câu"}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"arm": arm, **values} for arm, values in arms.items()]).to_csv(output / "metrics_by_arm.csv", index=False)
    metric_frame = pd.DataFrame([{"arm": arm, **{key: 100*value for key, value in values.items()}}
                                 for arm, values in arms.items()]).set_index("arm")
    metric_frame.plot(kind="bar", figsize=(13, 6)); plt.ylabel("Điểm trung bình (%)")
    plt.title("ST-VQA: toàn bộ metric theo nhánh"); plt.xticks(rotation=15, ha="right"); plt.legend(ncol=4)
    plt.tight_layout(); plt.savefig(visual / "all_metrics_by_arm.png", dpi=180); plt.close()
    shutil.make_archive(str(output), "zip", root_dir=output)


def main() -> int:
    args = arguments()
    if args.worker_gpu is not None:
        return worker(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(f"Notebook cần GPU T4 x2; hiện thấy {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    input_root = Path(args.input_root).resolve(); output = Path(args.output_root).resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest = find_stvqa_manifest(input_root)
    checkpoint = find_checkpoint(input_root, args.checkpoint)
    args.checkpoint = str(checkpoint)
    degraded_root = manifest.parent / "images" / "realistic_mix" / "L2"
    selected_path = output / "selected_manifest.csv"
    selected = select_manifest(manifest, selected_path, args.sample_size, args.seed)
    clean_root = materialize_clean(selected, Path("/kaggle/working/stvqa_clean_selected")) if args.include_clean else None
    print(f"Manifest: {manifest}\nCheckpoint: {checkpoint}\nSố câu: {len(selected)}", flush=True)
    processes = []
    for gpu in range(2):
        log_path = output / f"gpu_{gpu}.log"; handle = log_path.open("w", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-gpu", str(gpu), "--world-size", "2",
                   "--output-root", str(output), "--manifest", str(selected_path), "--checkpoint", str(checkpoint),
                   "--degraded-root", str(degraded_root)]
        if clean_root:
            command.extend(["--clean-root", str(clean_root)])
        processes.append((subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT), handle, log_path, gpu))
    failed = []
    for process, handle, log_path, gpu in processes:
        code = process.wait(); handle.close()
        if code:
            failed.append((gpu, log_path, code))
    if failed:
        for gpu, path, code in failed:
            print(f"\n--- GPU {gpu}, exit={code} ---\n{path.read_text(errors='replace')[-12000:]}")
        raise RuntimeError("Worker inference thất bại; shard đã hoàn thành vẫn được giữ để chạy tiếp")
    score_and_report(args, output, clean_root, degraded_root)
    print(f"HOÀN TẤT: {output}\nZIP: {output}.zip", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
