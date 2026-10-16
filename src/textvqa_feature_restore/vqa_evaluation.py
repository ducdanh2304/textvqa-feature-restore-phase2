from __future__ import annotations

import ast
import json
import re
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from rapidfuzz.distance import Levenshtein

from .adapter import ResidualFeatureRestorer
from .io_utils import atomic_write_json
from .research_training import load_research_model

EVAL_REV = "72f898ec35af86567423c7a019dd1dc5175d1a5a"
EVAL_URL = f"https://raw.githubusercontent.com/facebookresearch/mmf/{EVAL_REV}/mmf/utils/m4c_evaluators.py"
STVQA_METRIC_URL = "https://rrc.cvc.uab.es/?ch=11&com=tasks"
PROMPT = "Answer the question using the image. Reply with only the short answer, with no explanation. Question: {question}"


def _answer_processor(cache: Path):
    if not cache.is_file():
        with urllib.request.urlopen(EVAL_URL, timeout=90) as response:
            cache.write_bytes(response.read())
    source = cache.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "EvalAIAnswerProcessor")
    namespace: dict[str, Any] = {"re": re}
    exec("import re\n" + ast.get_source_segment(source, node), namespace)
    return namespace["EvalAIAnswerProcessor"]()


def _score(prediction: str, answers: list[str], normalize: Any, benchmark_name: str) -> tuple[float, float, float]:
    if benchmark_name == "stvqa":
        pred = str(prediction).lower().strip(); refs = [str(value).lower().strip() for value in answers]
    else:
        pred = normalize(prediction); refs = [normalize(str(value)) for value in answers]
    matches = sum(value == pred for value in refs)
    official = (sum(min((matches - int(value == pred)) / 3.0, 1.0) for value in refs) / len(refs)
                if len(refs) == 10 else float(matches > 0))
    exact = float(matches > 0)
    similarities = []
    for ref in refs:
        distance = Levenshtein.distance(pred, ref) / max(len(pred), len(ref), 1)
        similarities.append(1.0 - distance if distance < 0.5 else 0.0)
    return float(official), exact, float(max(similarities, default=0.0))


def _bootstrap(base: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(candidate, float) - np.asarray(base, float)
    rng = np.random.default_rng(42); n = len(delta); samples = np.empty(10000)
    for index in range(10000):
        samples[index] = delta[rng.integers(0, n, n)].mean()
    return {"mean_delta": float(delta.mean()), "ci95_low": float(np.quantile(samples, .025)),
            "ci95_high": float(np.quantile(samples, .975)), "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())}


def _comparison_table(table: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
    columns = ["question_id", "image_id", "question", "answers_json", "raw_prediction", metric]
    base = table[(table.condition == condition) & (table["mode"] == "base")][columns].copy()
    adapter = table[(table.condition == condition) & (table["mode"] == "adapter")][columns].copy()
    base = base.rename(columns={"raw_prediction": "base_prediction", metric: "base_score"})
    adapter = adapter.rename(columns={"raw_prediction": "adapter_prediction", metric: "adapter_score"})
    adapter = adapter[["question_id", "adapter_prediction", "adapter_score"]]
    result = base.merge(adapter, on="question_id", how="inner", validate="one_to_one")
    result["delta"] = result.adapter_score - result.base_score
    result["case"] = np.select(
        [result.delta > 0, result.delta < 0],
        ["ADAPTER_SUA_DUNG", "ADAPTER_LAM_SAI"],
        default="KHONG_DOI",
    )
    return result.sort_values(["delta", "question_id"], ascending=[False, True])


def _find_image(root: Path, image_id: str) -> Path | None:
    matches = [root / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png")]
    found = [path for path in matches if path.is_file()]
    return found[0] if len(found) == 1 else None


def _create_evaluation_visualizations(
    *, table: pd.DataFrame, destination: Path, clean_root: Path, degraded_root: Path,
    benchmark_name: str, primary_metric: str,
) -> dict[str, Any]:
    visual = destination / "visualizations"
    visual.mkdir(parents=True, exist_ok=True)
    comparisons: dict[str, pd.DataFrame] = {}
    for condition in ("clean", "degraded"):
        comparison = _comparison_table(table, condition, primary_metric)
        comparisons[condition] = comparison
        comparison.to_csv(destination / f"per_question_comparison_{condition}.csv", index=False)

    degraded = comparisons["degraded"]
    counts = degraded["case"].value_counts().reindex(
        ["ADAPTER_SUA_DUNG", "KHONG_DOI", "ADAPTER_LAM_SAI"], fill_value=0
    )
    labels = ["Sửa đúng", "Không đổi", "Làm sai"]
    colors = ["#66bb6a", "#9e9e9e", "#ef5350"]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, counts.values, color=colors)
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(int(value)), ha="center", va="bottom")
    plt.ylabel("Số câu hỏi"); plt.title("Ảnh suy thoái: thay đổi sau phục hồi feature")
    plt.tight_layout(); plt.savefig(visual / "win_tie_loss.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(degraded.delta, bins=25, color="#4c78a8", edgecolor="white")
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel(f"Điểm adapter - điểm base ({primary_metric})"); plt.ylabel("Số câu hỏi")
    plt.title("Phân bố thay đổi điểm trên ảnh suy thoái")
    plt.tight_layout(); plt.savefig(visual / "score_delta_histogram.png", dpi=180); plt.close()

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(degraded.base_score, degraded.adapter_score, alpha=.65, s=28)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlim(-.03, 1.03); plt.ylim(-.03, 1.03)
    plt.xlabel("Điểm base"); plt.ylabel("Điểm adapter")
    plt.title("So sánh từng câu hỏi trên ảnh suy thoái")
    plt.tight_layout(); plt.savefig(visual / "base_vs_adapter_scatter.png", dpi=180); plt.close()

    chosen = pd.concat([
        degraded[degraded.delta > 0].head(4),
        degraded[degraded.delta < 0].sort_values("delta").head(4),
    ], ignore_index=True)
    if chosen.empty:
        chosen = degraded.head(6)
    if not chosen.empty:
        figure, axes = plt.subplots(len(chosen), 3, figsize=(16, max(4, 4.1 * len(chosen))))
        axes = np.atleast_2d(axes)
        for row_index, record in enumerate(chosen.to_dict("records")):
            for column, (root, title) in enumerate(((clean_root, "Ảnh sạch đối chiếu"),
                                                    (degraded_root, "Ảnh suy thoái - đầu vào thật"))):
                axis = axes[row_index, column]; path = _find_image(root, str(record["image_id"]))
                if path:
                    with Image.open(path) as image:
                        axis.imshow(image.convert("RGB"))
                else:
                    axis.text(.5, .5, "Không tìm thấy ảnh", ha="center", va="center")
                axis.set_title(title); axis.axis("off")
            axis = axes[row_index, 2]; axis.axis("off")
            heading = {"ADAPTER_SUA_DUNG": "ADAPTER SỬA ĐÚNG", "ADAPTER_LAM_SAI": "ADAPTER LÀM SAI"}.get(
                record["case"], "KHÔNG ĐỔI")
            color = {"ADAPTER_SUA_DUNG": "#e8f5e9", "ADAPTER_LAM_SAI": "#ffebee"}.get(record["case"], "#eeeeee")
            answers = ", ".join(json.loads(record["answers_json"])[:4])
            message = (f"{heading}\n\nCâu hỏi: {record['question']}\nĐáp án tham chiếu: {answers}\n\n"
                       f"Base: {record['base_prediction']}\nĐiểm: {record['base_score']:.3f}\n\n"
                       f"Adapter: {record['adapter_prediction']}\nĐiểm: {record['adapter_score']:.3f}\n\n"
                       f"Chênh lệch: {record['delta']:+.3f}")
            axis.text(0, 1, textwrap.fill(message, width=58, replace_whitespace=False), va="top", fontsize=10,
                      bbox={"boxstyle": "round", "facecolor": color, "alpha": .9})
        figure.suptitle("Ví dụ định tính — adapter phục hồi feature, không sinh ảnh pixel mới", fontsize=14)
        figure.tight_layout(rect=[0, 0, 1, .99])
        figure.savefig(visual / "qualitative_examples.png", dpi=160, bbox_inches="tight"); plt.close(figure)

    summary = {
        "benchmark": benchmark_name, "primary_metric": primary_metric,
        "degraded": {"wins": int(counts.iloc[0]), "ties": int(counts.iloc[1]), "losses": int(counts.iloc[2]),
                     "mean_delta": float(degraded.delta.mean())},
        "note": "Adapter phục hồi feature ẩn. Vì vậy không có ảnh pixel phục hồi; ảnh suy thoái là đầu vào thật.",
    }
    atomic_write_json(destination / "visualization_report.json", summary)
    (visual / "README.md").write_text(
        "# Cách đọc biểu đồ\n\n"
        "- `win_tie_loss.png`: số câu adapter sửa đúng, giữ nguyên hoặc làm sai so với base.\n"
        "- `score_delta_histogram.png`: phân bố mức tăng/giảm điểm của từng câu.\n"
        "- `base_vs_adapter_scatter.png`: mỗi chấm là một câu hỏi; chấm phía trên đường chéo là cải thiện.\n"
        "- `qualitative_examples.png`: ảnh sạch đối chiếu, ảnh suy thoái đầu vào và câu trả lời trước/sau.\n\n"
        "Phương pháp này phục hồi **feature ẩn trong mô hình**, không phục hồi pixel. Do đó không có cột ảnh "
        "`restored image`; tác dụng phục hồi được đo bằng feature và câu trả lời VQA.\n",
        encoding="utf-8",
    )
    return summary


def evaluate_vqa(*, phase: int, cfg: dict[str, Any], checkpoint: Path, benchmark_manifest: Path,
                 clean_root: Path, degraded_root: Path, destination: Path,
                 base_predictions: Path | None = None, benchmark_name: str = "textvqa") -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "evaluation_report.json"
    predictions_path = destination / "predictions.csv"
    scored_path = destination / "scored_predictions.csv"
    if report_path.is_file() and predictions_path.is_file() and scored_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _create_evaluation_visualizations(
            table=pd.read_csv(scored_path, dtype={"question_id": str, "image_id": str}),
            destination=destination, clean_root=clean_root, degraded_root=degraded_root,
            benchmark_name=benchmark_name, primary_metric=report["primary_metric"],
        )
        return report
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    frame = pd.read_csv(benchmark_manifest, dtype=str, keep_default_na=False)
    device = torch.device("cuda:0")
    source = cfg["model"].get("path") or cfg["model"]["id"]
    kwargs: dict[str, Any] = {"local_files_only": bool(cfg["model"].get("offline"))}
    if not cfg["model"].get("path"):
        kwargs["revision"] = cfg["model"]["revision"]
    processor = AutoProcessor.from_pretrained(source, min_pixels=cfg["processor"]["min_pixels"],
                                              max_pixels=cfg["processor"]["max_pixels"], use_fast=False, **kwargs)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source, dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True, **kwargs
    ).to(device).eval()
    qwen.model.visual.to(device=device, dtype=torch.float32)
    for parameter in qwen.parameters(): parameter.requires_grad_(False)
    if phase == 3:
        adapter = ResidualFeatureRestorer().to(device).eval()
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        adapter.load_state_dict(payload.get("adapter_state_dict") or payload.get("adapter"), strict=True)
    else:
        adapter = load_research_model(phase, checkpoint, device)
    state: dict[str, Any] = {"enabled": False, "question": None, "calls": 0}

    def hook(_module: Any, _args: Any, output: torch.Tensor):
        state["calls"] += 1
        if not state["enabled"]: return output
        restored = adapter(output.float(), state["question"]) if phase >= 6 else adapter(output.float())
        if restored.shape != output.shape or not torch.isfinite(restored).all():
            raise RuntimeError("Adapter tạo feature sai kích thước hoặc không hữu hạn")
        return restored.to(output.dtype)

    handle = qwen.model.visual.register_forward_hook(hook)
    existing = pd.read_csv(predictions_path, dtype=str, keep_default_na=False).to_dict("records") if predictions_path.is_file() else []
    done = {(row["question_id"], row["condition"], row["mode"]) for row in existing}
    rows: list[dict[str, Any]] = list(existing)
    modes = ("base", "adapter") if phase == 3 else ("adapter",)
    try:
        for condition, root in (("clean", clean_root), ("degraded", degraded_root)):
            for mode in modes:
                state["enabled"] = mode == "adapter"
                for record in frame.to_dict("records"):
                    key = (record["question_id"], condition, mode)
                    if key in done: continue
                    matches = [path for suffix in (".jpg", ".jpeg", ".png") if (path := root / f'{record["image_id"]}{suffix}').is_file()]
                    if len(matches) != 1: raise FileNotFoundError(f"Không tìm thấy đúng một ảnh cho {record['image_id']} trong {root}")
                    if phase >= 6:
                        tokens = processor.tokenizer(record["question"], return_tensors="pt", add_special_tokens=True).to(device)
                        with torch.inference_mode():
                            values = qwen.get_input_embeddings()(tokens.input_ids).float()
                            state["question"] = (values * tokens.attention_mask.unsqueeze(-1)).sum(1) / tokens.attention_mask.sum(1, keepdim=True)
                    messages = [{"role": "user", "content": [{"type": "image", "image": str(matches[0])},
                                {"type": "text", "text": PROMPT.format(question=record["question"])}]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images, videos = process_vision_info(messages)
                    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(device)
                    inputs["pixel_values"] = inputs["pixel_values"].float()
                    before = state["calls"]
                    if hasattr(qwen.model, "rope_deltas"): qwen.model.rope_deltas = None
                    with torch.inference_mode():
                        generated = qwen.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=20)
                    answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)[0].strip()
                    if state["calls"] - before != 1 or not answer:
                        raise RuntimeError(f"Inference không hợp lệ cho question_id={record['question_id']}")
                    rows.append({"image_id": record["image_id"], "question_id": record["question_id"],
                                 "question": record["question"], "answers_json": record["answers_json"],
                                 "condition": condition, "mode": mode, "raw_prediction": answer})
                    done.add(key); pd.DataFrame(rows).to_csv(predictions_path, index=False)
    finally:
        handle.remove()

    if base_predictions is not None:
        base = pd.read_csv(base_predictions, dtype=str, keep_default_na=False)
        base = base[base["mode"] == "base"]
        rows.extend(base.to_dict("records"))
    normalize = (_answer_processor(destination / "m4c_evaluators_pinned.py")
                 if benchmark_name == "textvqa" else lambda value: str(value).lower().strip())
    scored = []
    for row in rows:
        official, exact, anls = _score(row["raw_prediction"], json.loads(row["answers_json"]), normalize, benchmark_name)
        scored.append({**row, "official_textvqa": official, "exact_match": exact, "anls": anls})
    table = pd.DataFrame(scored).drop_duplicates(["question_id", "condition", "mode"], keep="last")
    table.to_csv(scored_path, index=False)
    metrics = ("official_textvqa", "anls") if benchmark_name == "textvqa" else ("exact_match", "anls")
    primary_metric = "official_textvqa" if benchmark_name == "textvqa" else "anls"
    arms = {f"{condition}_{mode}": {metric: float(part[metric].mean()) for metric in metrics}
            for (condition, mode), part in table.groupby(["condition", "mode"])}
    comparison: dict[str, Any] = {}
    for condition in ("clean", "degraded"):
        subset = table[table.condition == condition].pivot(index="question_id", columns="mode", values=primary_metric)
        if {"base", "adapter"}.issubset(subset.columns):
            comparison[f"{condition}_adapter_minus_base"] = _bootstrap(subset.base.values, subset.adapter.values)
            changes = subset.reset_index()
            changes["delta"] = changes["adapter"] - changes["base"]
            changes["case"] = np.select([changes.delta > 0, changes.delta < 0],
                                         ["ADAPTER_SUA_DUNG", "ADAPTER_LAM_SAI"], default="KHONG_DOI")
            changes.to_csv(destination / f"case_changes_{condition}.csv", index=False)
    labels = list(arms)
    values = [100.0 * arms[label][primary_metric] for label in labels]
    plt.figure(figsize=(max(7, len(labels) * 1.4), 4.5))
    bars = plt.bar(labels, values)
    ylabel = "Official TextVQA Accuracy (%)" if benchmark_name == "textvqa" else "ST-VQA ANLS (%)"
    plt.ylabel(ylabel); plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    figure_name = "official_textvqa_accuracy.png" if benchmark_name == "textvqa" else "stvqa_anls.png"
    plt.tight_layout(); plt.savefig(destination / figure_name, dpi=180); plt.close()
    report = {"status": "COMPLETE", "phase": phase, "benchmark": benchmark_name,
              "primary_metric": primary_metric, "n_questions": int(frame.question_id.nunique()),
              "arms": arms, "paired_bootstrap": comparison,
              "checkpoint": str(checkpoint.resolve()),
              "metric_source": EVAL_URL if benchmark_name == "textvqa" else STVQA_METRIC_URL}
    atomic_write_json(report_path, report)
    _create_evaluation_visualizations(
        table=table, destination=destination, clean_root=clean_root, degraded_root=degraded_root,
        benchmark_name=benchmark_name, primary_metric=primary_metric,
    )
    return report
