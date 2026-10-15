from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .adapter import GatedFeatureRestorer, QuestionGatedFeatureRestorer, ResidualFeatureRestorer
from .io_utils import atomic_torch_save, atomic_write_json
from .trainer import CachePairDataset, batches


def _feature_terms(restored: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = ((restored - target).float().pow(2).mean(-1))[mask].mean()
    cosine = (1.0 - F.cosine_similarity(restored.float(), target.float(), dim=-1))[mask].mean()
    return mse + 0.1 * cosine, mse, cosine


def _question_lookup(phase1_root: Path) -> dict[str, str]:
    frames = [pd.read_csv(phase1_root / name, dtype=str, keep_default_na=False)
              for name in ("train_manifest.csv", "dev_manifest.csv")]
    frame = pd.concat(frames, ignore_index=True)
    return dict(zip(frame.question_id.astype(str), frame.question.astype(str)))


def _question_vectors(cfg: dict[str, Any], phase1_root: Path, destination: Path) -> dict[str, torch.Tensor]:
    if destination.is_file():
        return torch.load(destination, map_location="cpu", weights_only=False)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    source = cfg["model"].get("path") or cfg["model"]["id"]
    kwargs: dict[str, Any] = {"local_files_only": bool(cfg["model"].get("offline"))}
    if not cfg["model"].get("path"):
        kwargs["revision"] = cfg["model"]["revision"]
    processor = AutoProcessor.from_pretrained(source, use_fast=False, **kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source, dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True, **kwargs
    ).to("cuda:0").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    lookup = _question_lookup(phase1_root)
    result: dict[str, torch.Tensor] = {}
    embedding = model.get_input_embeddings()
    with torch.inference_mode():
        for question_id, question in lookup.items():
            tokens = processor.tokenizer(question, return_tensors="pt", add_special_tokens=True).to("cuda:0")
            value = embedding(tokens.input_ids).float()
            mask = tokens.attention_mask.unsqueeze(-1)
            result[question_id] = ((value * mask).sum(1) / mask.sum(1)).squeeze(0).cpu()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(result, destination)
    del model
    torch.cuda.empty_cache()
    return result


def _model_for_phase(phase: int) -> torch.nn.Module:
    if phase == 4:
        return ResidualFeatureRestorer()
    if phase == 5:
        return GatedFeatureRestorer()
    if phase in {6, 7}:
        return QuestionGatedFeatureRestorer()
    raise ValueError(phase)


def _state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload.get("model_state_dict") or payload.get("adapter_state_dict") or payload.get("adapter")


def _initialize(model: torch.nn.Module, phase: int, checkpoint: Path) -> None:
    old = _state(checkpoint)
    if phase == 4:
        model.load_state_dict(old, strict=True)
    elif phase == 5:
        model.restorer.load_state_dict(old, strict=True)
    elif phase in {6, 7}:
        # Giai đoạn 5/6 đều chứa nhánh restorer với cùng kiến trúc.
        prefix = "restorer."
        nested = {key[len(prefix):]: value for key, value in old.items() if key.startswith(prefix)}
        model.restorer.load_state_dict(nested or old, strict=True)
        if phase == 7:
            model.load_state_dict(old, strict=True)


def train_feature_phase(
    *, phase: int, cfg: dict[str, Any], phase1_root: Path, source_checkpoint: Path,
    destination: Path, max_epochs: int = 10, patience: int = 4,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    complete_path = destination / "training_summary.json"
    if complete_path.is_file() and (destination / "best_model.pt").is_file():
        return json.loads(complete_path.read_text(encoding="utf-8"))
    device = torch.device("cuda:0")
    model = _model_for_phase(phase).to(device=device, dtype=torch.float32)
    _initialize(model, phase, source_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    train_set, dev_set = CachePairDataset(cfg, "train"), CachePairDataset(cfg, "dev")
    questions = None
    if phase >= 6:
        questions = _question_vectors(cfg, phase1_root, destination.parent / "shared_question_embeddings.pt")

    def forward(batch: dict[str, Any], training: bool) -> tuple[torch.Tensor, dict[str, float]]:
        clean, degraded, mask = (batch[key].to(device) for key in ("clean", "degraded", "valid_tokens"))
        question = None
        if questions is not None:
            question = torch.stack([questions[str(qid)] for qid in batch["question_ids"]]).to(device)
        restored = model(degraded, question) if question is not None else model(degraded)
        restore, mse, cosine = _feature_terms(restored, clean, mask)
        unchanged = model(clean, question) if question is not None else model(clean)
        identity, _, _ = _feature_terms(unchanged, clean, mask)
        total = restore + 0.2 * identity
        result = {"total": float(total.detach()), "restore": float(restore.detach()),
                  "mse": float(mse.detach()), "cosine": float(cosine.detach()),
                  "identity": float(identity.detach())}
        if phase >= 5:
            _, gate = model(degraded, question, return_gate=True) if question is not None else model(degraded, return_gate=True)
            result["gate_mean"] = float(gate[mask].mean().detach())
        return total, result

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        model.eval(); values: list[dict[str, float]] = []
        for batch in batches(dev_set, list(range(len(dev_set))), 16):
            _, terms = forward(batch, False); values.append(terms)
        return {f"dev_{key}": float(np.nanmean([row[key] for row in values])) for key in values[0]}

    history: list[dict[str, Any]] = []
    best, best_epoch, stale, start_epoch = float("inf"), 0, 0, 1
    last_path = destination / "last_model.pt"
    if last_path.is_file():
        resume = torch.load(last_path, map_location=device, weights_only=False)
        if int(resume.get("phase", -1)) != phase or resume.get("source_checkpoint") != str(source_checkpoint.resolve()):
            raise RuntimeError(f"Checkpoint tiếp tục không thuộc đúng Giai đoạn {phase} hoặc sai checkpoint nguồn")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        history = list(resume.get("history", [])); start_epoch = int(resume["epoch"]) + 1
        selected = [row for row in history if bool(row.get("selected"))]
        if selected:
            best_epoch = int(selected[-1]["epoch"]); best = float(selected[-1]["dev_total"])
            stale = len(history) - history.index(selected[-1]) - 1
    for epoch in range(start_epoch, max_epochs + 1):
        model.train(); rows: list[dict[str, float]] = []
        order = np.random.default_rng(42 + epoch).permutation(len(train_set)).tolist()
        for batch in batches(train_set, order, 16):
            optimizer.zero_grad(set_to_none=True)
            loss, terms = forward(batch, True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Giai đoạn {phase}: loss không hữu hạn")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            rows.append(terms)
        dev = evaluate()
        row: dict[str, Any] = {"epoch": epoch, **{f"train_{k}": float(np.nanmean([x[k] for x in rows])) for k in rows[0]}, **dev}
        improved = dev["dev_total"] < best - 1e-10
        row["selected"] = improved; history.append(row)
        pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
        payload = {"phase": phase, "architecture": type(model).__name__, "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "history": history,
                   "source_checkpoint": str(source_checkpoint.resolve())}
        atomic_torch_save(payload, last_path)
        if improved:
            best, best_epoch, stale = dev["dev_total"], epoch, 0
            atomic_torch_save(payload, destination / "best_model.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    summary = {"status": "COMPLETE", "phase": phase, "architecture": type(model).__name__,
               "best_epoch": best_epoch, "best_dev_total": best, "epochs_completed": history[-1]["epoch"],
               "objective": "feature_restore + 0.2 * clean_identity",
               "question_conditioned": phase >= 6, "checkpoint": str((destination / "best_model.pt").resolve())}
    atomic_write_json(complete_path, summary)
    return summary


def load_research_model(phase: int, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = _model_for_phase(phase).to(device=device, dtype=torch.float32)
    model.load_state_dict(_state(checkpoint), strict=True)
    return model.eval()


def train_answer_phase(*, cfg: dict[str, Any], phase1_root: Path, source_checkpoint: Path,
                       destination: Path, max_epochs: int = 3, patience: int = 2) -> dict[str, Any]:
    """Fine-tune only the question gate/restorer through frozen-Qwen answer loss."""
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "training_summary.json"
    if summary_path.is_file() and (destination / "best_model.pt").is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = torch.device("cuda:0")
    adapter = QuestionGatedFeatureRestorer().to(device).train()
    adapter.load_state_dict(_state(source_checkpoint), strict=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-5, weight_decay=0.01)
    source = cfg["model"].get("path") or cfg["model"]["id"]
    kwargs: dict[str, Any] = {"local_files_only": bool(cfg["model"].get("offline"))}
    if not cfg["model"].get("path"): kwargs["revision"] = cfg["model"]["revision"]
    processor = AutoProcessor.from_pretrained(source, min_pixels=cfg["processor"]["min_pixels"],
                                              max_pixels=cfg["processor"]["max_pixels"], use_fast=False, **kwargs)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source, dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True, **kwargs
    ).to(device).eval()
    qwen.model.visual.to(device=device, dtype=torch.float32)
    for parameter in qwen.parameters(): parameter.requires_grad_(False)
    states: dict[str, Any] = {"question": None, "clean": None, "feature_loss": None, "identity_loss": None}

    def hook(_module: Any, _args: Any, output: torch.Tensor):
        restored = adapter(output.float(), states["question"])
        clean = states["clean"]
        if clean.shape != restored.shape:
            raise RuntimeError(f"Feature cache/hook mismatch: {clean.shape} != {restored.shape}")
        mask = torch.ones(1, clean.shape[0], dtype=torch.bool, device=device)
        states["feature_loss"] = _feature_terms(restored.unsqueeze(0), clean.unsqueeze(0), mask)[0]
        unchanged = adapter(clean, states["question"])
        states["identity_loss"] = _feature_terms(unchanged.unsqueeze(0), clean.unsqueeze(0), mask)[0]
        return restored.to(output.dtype)

    handle = qwen.model.visual.register_forward_hook(hook)
    prompt_template = "Answer the question using the image. Reply with only the short answer, with no explanation. Question: {question}"

    def rows(split: str) -> tuple[list[dict[str, str]], CachePairDataset, dict[str, int]]:
        frame = pd.read_csv(phase1_root / f"{split}_manifest.csv", dtype=str, keep_default_na=False)
        records = frame.to_dict("records"); cache = CachePairDataset(cfg, split)
        index = {pair.image_id: position for position, pair in enumerate(cache.pairs)}
        return records, cache, index

    train_rows, train_cache, train_index = rows("train")
    dev_rows, dev_cache, dev_index = rows("dev")

    def one(record: dict[str, str], cache: CachePairDataset, index: dict[str, int], training: bool) -> tuple[torch.Tensor, dict[str, float]]:
        item = cache[index[str(record["image_id"])]]
        states["clean"] = item["clean"].to(device)
        question = str(record["question"])
        with torch.no_grad():
            tokens = processor.tokenizer(question, return_tensors="pt", add_special_tokens=True).to(device)
            embedded = qwen.get_input_embeddings()(tokens.input_ids).float()
            states["question"] = (embedded * tokens.attention_mask.unsqueeze(-1)).sum(1) / tokens.attention_mask.sum(1, keepdim=True)
        answers = json.loads(record["answers_json"])
        answer = Counter(str(value).strip() for value in answers if str(value).strip()).most_common(1)[0][0]
        image = phase1_root / "data" / ("train" if cache is train_cache else "dev") / "degraded" / f'{record["image_id"]}.png'
        if not image.is_file():
            matches = list(image.parent.glob(f'{record["image_id"]}.*'))
            if len(matches) != 1: raise FileNotFoundError(f"Không tìm thấy ảnh suy thoái: {record['image_id']}")
            image = matches[0]
        messages = [{"role": "user", "content": [{"type": "image", "image": str(image)},
                    {"type": "text", "text": prompt_template.format(question=question)}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].float()
        answer_ids = processor.tokenizer(answer + processor.tokenizer.eos_token, add_special_tokens=False,
                                         return_tensors="pt").input_ids.to(device)
        prompt_length = inputs.input_ids.shape[1]
        inputs["input_ids"] = torch.cat([inputs.input_ids, answer_ids], dim=1)
        inputs["attention_mask"] = torch.cat([inputs.attention_mask, torch.ones_like(answer_ids)], dim=1)
        labels = inputs.input_ids.clone(); labels[:, :prompt_length] = -100
        if hasattr(qwen.model, "rope_deltas"): qwen.model.rope_deltas = None
        output = qwen(**inputs, labels=labels, use_cache=False)
        answer_loss = output.loss.float()
        total = answer_loss + 0.1 * states["feature_loss"] + 0.02 * states["identity_loss"]
        return total, {"total": float(total.detach()), "answer": float(answer_loss.detach()),
                       "feature": float(states["feature_loss"].detach()), "identity": float(states["identity_loss"].detach())}

    def evaluate() -> dict[str, float]:
        adapter.eval(); values = []
        with torch.no_grad():
            for record in dev_rows:
                _, terms = one(record, dev_cache, dev_index, False); values.append(terms)
        adapter.train()
        return {f"dev_{key}": float(np.mean([row[key] for row in values])) for key in values[0]}

    history: list[dict[str, Any]] = []; best = float("inf"); best_epoch = stale = 0; start_epoch = 1
    last_path = destination / "last_model.pt"
    if last_path.is_file():
        resume = torch.load(last_path, map_location=device, weights_only=False)
        if int(resume.get("phase", -1)) != 7 or resume.get("source_checkpoint") != str(source_checkpoint.resolve()):
            raise RuntimeError("Checkpoint tiếp tục Giai đoạn 7 không khớp checkpoint nguồn")
        adapter.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        history = list(resume.get("history", [])); start_epoch = int(resume["epoch"]) + 1
        selected = [row for row in history if bool(row.get("selected"))]
        if selected:
            best_epoch = int(selected[-1]["epoch"]); best = float(selected[-1]["dev_total"])
            stale = len(history) - history.index(selected[-1]) - 1
    try:
        for epoch in range(start_epoch, max_epochs + 1):
            order = np.random.default_rng(42 + epoch).permutation(len(train_rows)); terms_epoch = []
            for idx in order:
                optimizer.zero_grad(set_to_none=True)
                total, terms = one(train_rows[int(idx)], train_cache, train_index, True)
                if not torch.isfinite(total): raise RuntimeError("Giai đoạn 7: loss không hữu hạn")
                total.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); optimizer.step()
                terms_epoch.append(terms)
            dev = evaluate(); improved = dev["dev_total"] < best - 1e-10
            row: dict[str, Any] = {"epoch": epoch, **{f"train_{key}": float(np.mean([x[key] for x in terms_epoch])) for key in terms_epoch[0]}, **dev, "selected": improved}
            history.append(row); pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
            payload = {"phase": 7, "architecture": type(adapter).__name__, "model_state_dict": adapter.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "history": history,
                       "source_checkpoint": str(source_checkpoint.resolve())}
            atomic_torch_save(payload, last_path)
            if improved:
                best, best_epoch, stale = dev["dev_total"], epoch, 0
                atomic_torch_save(payload, destination / "best_model.pt")
            else: stale += 1
            if stale >= patience: break
    finally:
        handle.remove()
    summary = {"status": "COMPLETE", "phase": 7, "architecture": type(adapter).__name__,
               "best_epoch": best_epoch, "best_dev_total": best, "epochs_completed": history[-1]["epoch"],
               "objective": "answer_CE + 0.1 * feature_restore + 0.02 * clean_identity",
               "checkpoint": str((destination / "best_model.pt").resolve())}
    atomic_write_json(summary_path, summary)
    return summary
