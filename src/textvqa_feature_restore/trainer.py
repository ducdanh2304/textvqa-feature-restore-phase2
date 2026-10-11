from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import CACHE_IMPLEMENTATION_VERSION, MODEL_ID, MODEL_REVISION
from .adapter import ResidualFeatureRestorer, adapter_config, masked_feature_loss
from .checkpointing import expected_provenance, load_checkpoint, validate_checkpoint_provenance
from .config import output_root, phase1_root
from .diagnostics import append_error, configure_logging, tensor_stats
from .feature_cache import expected_cache_provenance, tensor_bytes, validate_cache_file
from .io_utils import atomic_torch_save, atomic_write_json, directory_manifest, read_jsonl, sha256_bytes, sha256_file
from .reproducibility import capture_rng_state, restore_rng_state, set_reproducible


@dataclass(frozen=True)
class CachedPair:
    split: str
    image_id: str
    question_id: str
    degradation_type: str
    clean_file: Path
    degraded_file: Path
    clean_metadata: dict[str, Any]
    degraded_metadata: dict[str, Any]


class CachePairDataset:
    def __init__(self, cfg: dict[str, Any], split: str) -> None:
        self.cfg = cfg
        out = output_root(cfg)
        manifest_path = out / "cache" / "manifest.jsonl"
        records = read_jsonl(manifest_path)
        if len(records) != 4400 or len({(item.get("split"), item.get("image_id"), item.get("condition")) for item in records}) != 4400:
            raise ValueError("Cache manifest must contain exactly 4,400 unique split/image/condition records")
        expected_global = expected_cache_provenance(cfg)
        for item in records:
            mismatch = {key: (value, item.get(key)) for key, value in expected_global.items() if item.get(key) != value}
            if mismatch:
                raise ValueError(f"Cache manifest provenance mismatch for {item.get('image_id')}: {mismatch}")
        by_key = {(item["split"], item["image_id"], item["condition"]): item for item in records}
        image_ids = sorted({item["image_id"] for item in records if item["split"] == split})
        expected_count = 2000 if split == "train" else 200
        if len(image_ids) != expected_count:
            raise ValueError(f"Cache has {len(image_ids)} {split} pairs; expected {expected_count}")
        self.pairs: list[CachedPair] = []
        for image_id in image_ids:
            clean = by_key.get((split, image_id, "clean"))
            degraded = by_key.get((split, image_id, "degraded"))
            if clean is None or degraded is None:
                raise ValueError(f"Incomplete cache pair: {split}/{image_id}")
            if clean["question_id"] != degraded["question_id"]:
                raise ValueError(f"Question mismatch in cache pair: {split}/{image_id}")
            self.pairs.append(CachedPair(
                split=split, image_id=image_id, question_id=clean["question_id"],
                degradation_type=degraded["degradation_type"], clean_file=Path(clean["cache_file"]),
                degraded_file=Path(degraded["cache_file"]), clean_metadata=clean, degraded_metadata=degraded,
            ))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        clean_payload = torch.load(pair.clean_file, map_location="cpu", weights_only=False)
        degraded_payload = torch.load(pair.degraded_file, map_location="cpu", weights_only=False)
        if sha256_file(pair.clean_file) != pair.clean_metadata["cache_file_sha256"]:
            raise ValueError(f"Clean cache file checksum mismatch: {pair.image_id}")
        if sha256_file(pair.degraded_file) != pair.degraded_metadata["cache_file_sha256"]:
            raise ValueError(f"Degraded cache file checksum mismatch: {pair.image_id}")
        clean_manifest_metadata = {key: value for key, value in pair.clean_metadata.items() if key not in {"cache_file", "cache_file_sha256"}}
        degraded_manifest_metadata = {key: value for key, value in pair.degraded_metadata.items() if key not in {"cache_file", "cache_file_sha256"}}
        if clean_payload.get("metadata") != clean_manifest_metadata:
            raise ValueError(f"Clean cache payload metadata differs from manifest: {pair.image_id}")
        if degraded_payload.get("metadata") != degraded_manifest_metadata:
            raise ValueError(f"Degraded cache payload metadata differs from manifest: {pair.image_id}")
        clean, degraded = clean_payload["feature"], degraded_payload["feature"]
        if sha256_bytes(tensor_bytes(clean)) != pair.clean_metadata["cached_feature_sha256"]:
            raise ValueError(f"Clean cache checksum mismatch: {pair.image_id}")
        if sha256_bytes(tensor_bytes(degraded)) != pair.degraded_metadata["cached_feature_sha256"]:
            raise ValueError(f"Degraded cache checksum mismatch: {pair.image_id}")
        if clean.shape != degraded.shape or clean.ndim != 2 or clean.shape[-1] != 2048:
            raise ValueError(f"Feature shape mismatch: {pair.image_id}: {clean.shape}/{degraded.shape}")
        if not torch.isfinite(clean).all() or not torch.isfinite(degraded).all():
            raise ValueError(f"Non-finite cached pair: {pair.image_id}")
        return {
            "image_id": pair.image_id, "question_id": pair.question_id, "degradation_type": pair.degradation_type,
            "clean": clean.float(), "degraded": degraded.float(),
            "clean_metadata": pair.clean_metadata, "degraded_metadata": pair.degraded_metadata,
        }


def padded_collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    lengths = [item["clean"].shape[0] for item in items]
    if any(item["clean"].shape != item["degraded"].shape for item in items):
        raise ValueError("A clean/degraded pair has mismatched shape")
    max_tokens, hidden = max(lengths), items[0]["clean"].shape[-1]
    clean = torch.zeros(len(items), max_tokens, hidden, dtype=torch.float32)
    degraded = torch.zeros_like(clean)
    mask = torch.zeros(len(items), max_tokens, dtype=torch.bool)
    for index, (item, length) in enumerate(zip(items, lengths)):
        clean[index, :length] = item["clean"]
        degraded[index, :length] = item["degraded"]
        mask[index, :length] = True
    if int(mask.sum()) != sum(lengths):
        raise ValueError("Padded valid-token mask count is incorrect")
    return {
        "clean": clean, "degraded": degraded, "valid_tokens": mask,
        "image_ids": [item["image_id"] for item in items],
        "question_ids": [item["question_id"] for item in items],
        "degradation_types": [item["degradation_type"] for item in items],
        "metadata": [{"clean": item["clean_metadata"], "degraded": item["degraded_metadata"]} for item in items],
    }


def batches(dataset: CachePairDataset, indices: Sequence[int], batch_size: int) -> Iterable[dict[str, Any]]:
    for start in range(0, len(indices), batch_size):
        yield padded_collate([dataset[index] for index in indices[start:start + batch_size]])


def _append_csv(path: Path, fields: Sequence[str], row: dict[str, Any]) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _finite_parameters(module: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in module.parameters())


def _probe_batch_size(cfg: dict[str, Any], dataset: CachePairDataset, device: torch.device, logger: Any) -> int:
    configured = cfg["training"]["batch_size"]
    candidates = [32, 16, 8] if configured == "auto" else [int(configured)]
    failures: dict[int, str] = {}
    for candidate in candidates:
        probe = ResidualFeatureRestorer().to(device=device, dtype=torch.float32)
        try:
            batch = next(batches(dataset, list(range(min(candidate, len(dataset)))), candidate))
            clean, degraded, mask = (batch[key].to(device) for key in ("clean", "degraded", "valid_tokens"))
            restored = probe(degraded)
            loss, _, _ = masked_feature_loss(restored, clean, mask)
            loss.backward()
            if not torch.isfinite(loss) or any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in probe.parameters()):
                raise RuntimeError("Non-finite result in batch-size probe")
            logger.info("Batch-size probe selected %d (configured=%s)", candidate, configured)
            del probe, batch, clean, degraded, mask, restored, loss
            torch.cuda.empty_cache()
            return candidate
        except torch.cuda.OutOfMemoryError as error:
            failures[candidate] = str(error)
            del probe
            torch.cuda.empty_cache()
            if configured != "auto":
                raise RuntimeError(f"Configured batch_size={candidate} is not memory-safe; change YAML explicitly") from error
    raise RuntimeError(f"No candidate batch size is memory-safe: {failures}")


def _evaluate(adapter: ResidualFeatureRestorer, dataset: CachePairDataset, batch_size: int, device: torch.device) -> dict[str, float]:
    adapter.eval(); total_sum = mse_sum = cosine_sum = 0.0; count = 0
    with torch.no_grad():
        for batch in batches(dataset, list(range(len(dataset))), batch_size):
            clean, degraded, mask = (batch[key].to(device) for key in ("clean", "degraded", "valid_tokens"))
            if not torch.isfinite(clean).all() or not torch.isfinite(degraded).all():
                raise RuntimeError(f"Non-finite dev cache: {batch['image_ids']}")
            restored = adapter(degraded)
            total, mse, cosine = masked_feature_loss(restored, clean, mask)
            if not all(torch.isfinite(value).item() for value in (restored, total, mse, cosine)):
                raise RuntimeError(f"Non-finite dev output/loss: {batch['image_ids']}")
            n = len(batch["image_ids"]); count += n
            total_sum += float(total.item()) * n; mse_sum += float(mse.item()) * n; cosine_sum += float(cosine.item()) * n
    if count != 200:
        raise RuntimeError(f"Dev evaluation used {count} pairs; expected 200")
    result = {"mean_dev_total_loss": total_sum / count, "mean_dev_mse_loss": mse_sum / count,
              "mean_dev_cosine_loss": cosine_sum / count, "number_of_samples": count}
    if not all(math.isfinite(value) for key, value in result.items() if key != "number_of_samples"):
        raise RuntimeError("Non-finite dev aggregate")
    return result


def _make_checkpoint(
    adapter: ResidualFeatureRestorer, optimizer: torch.optim.Optimizer, epoch: int, global_step: int,
    best_dev_loss: float, current_dev_loss: float, counter: int, provenance: dict[str, Any], history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "adapter_state_dict": adapter.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "global_step": global_step, "best_dev_loss": best_dev_loss,
        "current_dev_loss": current_dev_loss, "early_stopping_counter": counter,
        "adapter_config": adapter_config(), "training_history": history, **provenance, **capture_rng_state(),
    }


def _save_figures(train_csv: Path, dev_csv: Path, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    train, dev = pd.read_csv(train_csv), pd.read_csv(dev_csv)
    for column, name in [("train_total_loss", "train_total_loss.png"), ("train_mse_loss", "train_mse_loss.png"), ("train_cosine_loss", "train_cosine_loss.png")]:
        plt.figure(figsize=(9, 4)); plt.plot(train.global_step, train[column], linewidth=.7); plt.xlabel("Global step"); plt.ylabel(column); plt.tight_layout(); plt.savefig(figure_dir / name, dpi=180); plt.close()
    for column, name in [("mean_dev_total_loss", "dev_total_loss.png"), ("mean_dev_mse_loss", "dev_mse_loss.png"), ("mean_dev_cosine_loss", "dev_cosine_loss.png")]:
        plt.figure(figsize=(7, 4)); plt.plot(dev.epoch, dev[column], marker="o"); plt.xlabel("Epoch"); plt.ylabel(column); plt.tight_layout(); plt.savefig(figure_dir / name, dpi=180); plt.close()


def run_training(cfg: dict[str, Any], resume_path: str | None = None) -> dict[str, Any]:
    out, root = output_root(cfg), phase1_root(cfg)
    training_dir, log_dir = out / "training", out / "training" / "logs"
    training_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_dir / "train.log", "phase2.train")
    errors = log_dir / "errors.jsonl"
    if not resume_path:
        if (training_dir / "last_checkpoint.pt").exists() or (training_dir / "best_adapter.pt").exists():
            raise RuntimeError("Training outputs already exist. Use --resume explicitly or choose an empty output root.")
        errors.write_text("", encoding="utf-8")
    set_reproducible(int(cfg["reproducibility"]["seed"]), bool(cfg["hardware"]["allow_tf32"]))
    if not (out / "cache" / "manifest.jsonl").is_file():
        raise RuntimeError("Feature cache manifest is missing; run cache-features first")
    cache_summary = json.loads((out / "cache" / "cache_summary.json").read_text())
    if cache_summary.get("failures") != 0 or cache_summary.get("total_cached") != 4400:
        raise RuntimeError("Feature cache summary is incomplete")
    train_set, dev_set = CachePairDataset(cfg, "train"), CachePairDataset(cfg, "dev")
    device = torch.device("cuda:0")
    batch_size = _probe_batch_size(cfg, train_set, device, logger)
    adapter = ResidualFeatureRestorer().to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.01)
    resolved = out / "phase2_config_resolved.yaml"
    provenance = expected_provenance(
        model_id=MODEL_ID, model_revision=MODEL_REVISION, train_manifest=root / "train_manifest.csv",
        dev_manifest=root / "dev_manifest.csv", benchmark_ids=root / "benchmark_excluded_image_ids.csv",
        cache_manifest=out / "cache" / "manifest.jsonl", resolved_config=resolved, batch_size=batch_size,
        cache_storage_dtype=cfg["cache"]["storage_dtype"], cache_implementation_version=CACHE_IMPLEMENTATION_VERSION,
    )
    train_csv, dev_csv = training_dir / "train_history.csv", training_dir / "dev_history.csv"
    train_fields = ["epoch", "global_step", "image_ids", "train_total_loss", "train_mse_loss", "train_cosine_loss", "gradient_norm", "valid_tokens", "feature_shape", "degradation_types", "qwen_gradient_count", "finite"]
    dev_fields = ["epoch", "mean_dev_total_loss", "mean_dev_mse_loss", "mean_dev_cosine_loss", "number_of_samples", "finite", "checkpoint_selected", "early_stopping_counter", "early_stopping_triggered"]
    start_epoch, global_step, best_dev, patience, best_epoch, history = 1, 0, float("inf"), 0, None, []
    initial_dev = None
    try:
        if resume_path:
            checkpoint = load_checkpoint(resume_path, device)
            validate_checkpoint_provenance(checkpoint, provenance)
            if "phase1" in Path(resume_path).name.lower() or "smoke" in Path(resume_path).name.lower():
                raise ValueError("Refusing to resume from a Phase-1 smoke checkpoint")
            adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            restore_rng_state(checkpoint)
            start_epoch = int(checkpoint["epoch"]) + 1; global_step = int(checkpoint["global_step"])
            best_dev = float(checkpoint["best_dev_loss"]); patience = int(checkpoint["early_stopping_counter"])
            history = list(checkpoint.get("training_history", []))
            old_dev = pd.read_csv(dev_csv)
            old_train = pd.read_csv(train_csv)
            if int(old_dev.iloc[-1].epoch) != int(checkpoint["epoch"]) or int(old_train.iloc[-1].global_step) != int(checkpoint["global_step"]):
                raise ValueError("History tail does not match the explicit resume checkpoint; refusing to append duplicate/divergent history")
            initial_dev = float(old_dev.iloc[0].mean_dev_total_loss)
            selected = old_dev[old_dev.checkpoint_selected.astype(str).str.lower() == "true"]
            best_epoch = int(selected.iloc[-1].epoch) if len(selected) else None
            logger.info("Resuming at epoch %d from %s", start_epoch, resume_path)
        else:
            initial = _evaluate(adapter, dev_set, batch_size, device)
            initial_dev = initial["mean_dev_total_loss"]; best_dev = initial_dev; best_epoch = 0
            initial.update(epoch=0, finite=True, checkpoint_selected=True, early_stopping_counter=0, early_stopping_triggered=False)
            _append_csv(dev_csv, dev_fields, initial)
            atomic_torch_save(_make_checkpoint(adapter, optimizer, 0, 0, best_dev, best_dev, 0, provenance, history), training_dir / "best_adapter.pt")
        stopped = False; completed_epoch = start_epoch - 1
        for epoch in range(start_epoch, int(cfg["training"]["max_epochs"]) + 1):
            adapter.train()
            order = np.random.default_rng(int(cfg["reproducibility"]["seed"]) + epoch).permutation(len(train_set)).tolist()
            epoch_losses: list[float] = []
            for batch in batches(train_set, order, batch_size):
                global_step += 1; optimizer.zero_grad(set_to_none=True)
                clean, degraded, mask = (batch[key].to(device) for key in ("clean", "degraded", "valid_tokens"))
                if not torch.isfinite(clean).all() or not torch.isfinite(degraded).all():
                    raise RuntimeError(f"Non-finite cached train feature: {batch['image_ids']}")
                restored = adapter(degraded)
                total, mse, cosine = masked_feature_loss(restored, clean, mask)
                if not all(torch.isfinite(value).all().item() for value in (restored, total, mse, cosine)):
                    raise RuntimeError(f"Non-finite train output/loss: {batch['image_ids']}")
                (total / int(cfg["training"]["gradient_accumulation"])).backward()
                gradients = [parameter.grad for parameter in adapter.parameters() if parameter.grad is not None]
                if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                    raise RuntimeError(f"Missing/non-finite adapter gradient: {batch['image_ids']}")
                gradient_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(cfg["training"]["gradient_clip_norm"]))
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(f"Non-finite gradient norm: {batch['image_ids']}")
                optimizer.step()
                if not _finite_parameters(adapter):
                    raise RuntimeError(f"Non-finite adapter parameter: {batch['image_ids']}")
                record = {"epoch": epoch, "global_step": global_step, "image_ids": json.dumps(batch["image_ids"]),
                    "train_total_loss": float(total.item()), "train_mse_loss": float(mse.item()),
                    "train_cosine_loss": float(cosine.item()), "gradient_norm": float(gradient_norm.item()),
                    "valid_tokens": int(mask.sum().item()), "feature_shape": json.dumps(list(restored.shape)),
                    "degradation_types": json.dumps(batch["degradation_types"]), "qwen_gradient_count": 0, "finite": True}
                _append_csv(train_csv, train_fields, record); history.append(record); epoch_losses.append(record["train_total_loss"])
                if global_step % 25 == 0:
                    logger.info("epoch=%d step=%d loss=%.8f grad=%.6f", epoch, global_step, record["train_total_loss"], record["gradient_norm"])
            dev = _evaluate(adapter, dev_set, batch_size, device)
            improved = dev["mean_dev_total_loss"] < best_dev
            if improved:
                best_dev = dev["mean_dev_total_loss"]; best_epoch = epoch; patience = 0
            else:
                patience += 1
            stopped = patience >= int(cfg["training"]["early_stopping_patience"])
            dev.update(epoch=epoch, finite=True, checkpoint_selected=improved, early_stopping_counter=patience, early_stopping_triggered=stopped)
            _append_csv(dev_csv, dev_fields, dev)
            payload = _make_checkpoint(adapter, optimizer, epoch, global_step, best_dev, dev["mean_dev_total_loss"], patience, provenance, history)
            atomic_torch_save(payload, training_dir / "last_checkpoint.pt")
            if improved:
                atomic_torch_save(payload, training_dir / "best_adapter.pt")
            completed_epoch = epoch
            atomic_write_json(training_dir / "early_stopping.json", {"patience": int(cfg["training"]["early_stopping_patience"]),
                "counter": patience, "best_epoch": best_epoch, "best_dev_loss": best_dev,
                "last_completed_epoch": epoch, "triggered": stopped})
            logger.info("epoch=%d train_mean=%.8f dev=%.8f improved=%s patience=%d", epoch, float(np.mean(epoch_losses)), dev["mean_dev_total_loss"], improved, patience)
            if stopped:
                break
        _save_figures(train_csv, dev_csv, out / "figures")
        summary = {"status": "COMPLETE", "train_pairs": len(train_set), "dev_pairs": len(dev_set), "benchmark_overlap": 0,
            "qwen_trainable_parameters": 0, "qwen_parameters_with_grad": 0,
            "adapter_trainable_parameters": sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad),
            "batch_size": batch_size, "max_epochs": int(cfg["training"]["max_epochs"]), "actual_epochs_completed": completed_epoch,
            "best_epoch": best_epoch, "initial_dev_total_loss": initial_dev, "best_dev_total_loss": best_dev,
            "early_stopping_triggered": stopped, "best_checkpoint": str(training_dir / "best_adapter.pt"),
            "locked_benchmark_evaluated": False, "textvqa_accuracy_computed": False, "language_generation_run": False}
        atomic_write_json(training_dir / "training_summary.json", summary)
        atomic_write_json(training_dir / "checkpoint_manifest.json", directory_manifest(training_dir, {"checkpoint_manifest.json"}))
        return summary
    except Exception as error:
        append_error(errors, "training", error, epoch=locals().get("epoch"), global_step=global_step,
                     image_ids=locals().get("batch", {}).get("image_ids") if isinstance(locals().get("batch"), dict) else None,
                     metadata=locals().get("batch", {}).get("metadata") if isinstance(locals().get("batch"), dict) else None)
        try:
            emergency = _make_checkpoint(adapter, optimizer, locals().get("epoch", start_epoch - 1), global_step, best_dev,
                                         locals().get("dev", {}).get("mean_dev_total_loss", float("inf")) if isinstance(locals().get("dev"), dict) else float("inf"),
                                         patience, provenance, history)
            atomic_torch_save(emergency, training_dir / "emergency_checkpoint.pt")
        except Exception as checkpoint_error:
            append_error(errors, "emergency_checkpoint", checkpoint_error)
        raise

