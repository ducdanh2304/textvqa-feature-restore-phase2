from __future__ import annotations

import ast
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein

from .adapter import ResidualFeatureRestorer
from .io_utils import atomic_write_json
from .research_training import load_research_model

EVAL_REV = "72f898ec35af86567423c7a019dd1dc5175d1a5a"
EVAL_URL = f"https://raw.githubusercontent.com/facebookresearch/mmf/{EVAL_REV}/mmf/utils/m4c_evaluators.py"
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


def _score(prediction: str, answers: list[str], normalize: Any) -> tuple[float, float]:
    pred = normalize(prediction); refs = [normalize(str(value)) for value in answers]
    matches = sum(value == pred for value in refs)
    official = sum(min((matches - int(value == pred)) / 3.0, 1.0) for value in refs) / len(refs)
    similarities = []
    for ref in refs:
        distance = Levenshtein.distance(pred, ref) / max(len(pred), len(ref), 1)
        similarities.append(1.0 - distance if distance < 0.5 else 0.0)
    return float(official), float(max(similarities, default=0.0))


def _bootstrap(base: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(candidate, float) - np.asarray(base, float)
    rng = np.random.default_rng(42); n = len(delta); samples = np.empty(10000)
    for index in range(10000):
        samples[index] = delta[rng.integers(0, n, n)].mean()
    return {"mean_delta": float(delta.mean()), "ci95_low": float(np.quantile(samples, .025)),
            "ci95_high": float(np.quantile(samples, .975)), "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())}


def evaluate_vqa(*, phase: int, cfg: dict[str, Any], checkpoint: Path, benchmark_manifest: Path,
                 clean_root: Path, degraded_root: Path, destination: Path,
                 base_predictions: Path | None = None) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "evaluation_report.json"
    predictions_path = destination / "predictions.csv"
    if report_path.is_file() and predictions_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
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
    normalize = _answer_processor(destination / "m4c_evaluators_pinned.py")
    scored = []
    for row in rows:
        official, anls = _score(row["raw_prediction"], json.loads(row["answers_json"]), normalize)
        scored.append({**row, "official_textvqa": official, "anls": anls})
    table = pd.DataFrame(scored).drop_duplicates(["question_id", "condition", "mode"], keep="last")
    table.to_csv(destination / "scored_predictions.csv", index=False)
    arms = {f"{condition}_{mode}": {metric: float(part[metric].mean()) for metric in ("official_textvqa", "anls")}
            for (condition, mode), part in table.groupby(["condition", "mode"])}
    comparison: dict[str, Any] = {}
    for condition in ("clean", "degraded"):
        subset = table[table.condition == condition].pivot(index="question_id", columns="mode", values="official_textvqa")
        if {"base", "adapter"}.issubset(subset.columns):
            comparison[f"{condition}_adapter_minus_base"] = _bootstrap(subset.base.values, subset.adapter.values)
            changes = subset.reset_index()
            changes["delta"] = changes["adapter"] - changes["base"]
            changes["case"] = np.select([changes.delta > 0, changes.delta < 0],
                                         ["ADAPTER_SUA_DUNG", "ADAPTER_LAM_SAI"], default="KHONG_DOI")
            changes.to_csv(destination / f"case_changes_{condition}.csv", index=False)
    labels = list(arms)
    values = [100.0 * arms[label]["official_textvqa"] for label in labels]
    plt.figure(figsize=(max(7, len(labels) * 1.4), 4.5))
    bars = plt.bar(labels, values)
    plt.ylabel("Official TextVQA Accuracy (%)"); plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(destination / "official_textvqa_accuracy.png", dpi=180); plt.close()
    report = {"status": "COMPLETE", "phase": phase, "n_questions": int(frame.question_id.nunique()),
              "arms": arms, "paired_bootstrap_official_textvqa": comparison,
              "checkpoint": str(checkpoint.resolve()), "metric_source": EVAL_URL}
    atomic_write_json(report_path, report)
    return report
