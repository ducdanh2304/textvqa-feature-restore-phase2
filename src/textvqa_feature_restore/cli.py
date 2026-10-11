from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML protocol configuration")
    parser.add_argument("--data-root", help="Override paths.data_root")
    parser.add_argument("--output-root", help="Override paths.output_root")
    parser.add_argument("--model-path", help="Override model.path")
    parser.add_argument("--offline", action="store_true", help="Require local files and disable network model access")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="textvqa-feature-restore-phase2")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "cache-features", "train", "verify", "bundle"):
        command = sub.add_parser(name)
        _common(command)
        if name == "train":
            command.add_argument("--resume", help="Explicit Phase-2 last_checkpoint.pt")
        if name == "verify":
            command.add_argument("--checkpoint", required=True, help="Phase-2 best_adapter.pt")
    return parser


def _config(args: argparse.Namespace) -> dict[str, Any]:
    from .config import load_config

    offline = True if args.offline else None
    return load_config(args.config, data_root=args.data_root, output_root=args.output_root,
                       model_path=args.model_path, offline=offline)


def run_verify(cfg: dict[str, Any], checkpoint_path: str) -> dict[str, Any]:
    import torch

    from . import CACHE_IMPLEMENTATION_VERSION, MODEL_ID, MODEL_REVISION
    from .adapter import ResidualFeatureRestorer, masked_feature_loss
    from .checkpointing import expected_provenance, load_checkpoint, validate_checkpoint_provenance
    from .config import output_root, phase1_root
    from .feature_cache import load_phase1_records
    from .io_utils import atomic_write_json, atomic_write_text, read_jsonl
    from .trainer import CachePairDataset, padded_collate

    out, root = output_root(cfg), phase1_root(cfg)
    verification = out / "verification"; verification.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "cache" / "cache_summary.json").read_text(encoding="utf-8"))
    batch_size = int(load_checkpoint(checkpoint_path)["batch_size"])
    provenance = expected_provenance(
        model_id=MODEL_ID, model_revision=MODEL_REVISION, train_manifest=root / "train_manifest.csv",
        dev_manifest=root / "dev_manifest.csv", benchmark_ids=root / "benchmark_excluded_image_ids.csv",
        cache_manifest=out / "cache" / "manifest.jsonl", resolved_config=out / "phase2_config_resolved.yaml",
        batch_size=batch_size, cache_storage_dtype=cfg["cache"]["storage_dtype"],
        cache_implementation_version=CACHE_IMPLEMENTATION_VERSION,
    )
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    validate_checkpoint_provenance(checkpoint, provenance)
    adapter = ResidualFeatureRestorer().float()
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True); adapter.eval()
    dev = CachePairDataset(cfg, "dev")
    batch = padded_collate([dev[0]])
    with torch.no_grad():
        restored = adapter(batch["degraded"])
        total, mse, cosine = masked_feature_loss(restored, batch["clean"], batch["valid_tokens"])
    report = {
        "status": "VERIFIED", "checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_epoch": int(checkpoint["epoch"]),
        "image_id": batch["image_ids"][0], "clean_shape": list(batch["clean"].shape),
        "degraded_shape": list(batch["degraded"].shape), "restored_shape": list(restored.shape),
        "finite": bool(torch.isfinite(restored).all() and torch.isfinite(total) and torch.isfinite(mse) and torch.isfinite(cosine)),
        "total_feature_loss": float(total), "mse_loss": float(mse), "cosine_loss": float(cosine),
        "language_generation_run": False, "textvqa_accuracy_computed": False,
    }
    if report["clean_shape"] != report["degraded_shape"] or report["clean_shape"] != report["restored_shape"] or not report["finite"]:
        raise RuntimeError(f"Checkpoint feature-only verification failed: {report}")
    atomic_write_json(verification / "checkpoint_verification.json", report)
    atomic_write_text(verification / "checkpoint_verification.md", "# Checkpoint verification — VERIFIED\n\n" + "\n".join(f"- **{key}**: `{value}`" for key, value in report.items()) + "\n")
    return report


def run_bundle(cfg: dict[str, Any]) -> dict[str, Any]:
    from .config import output_root
    from .io_utils import atomic_write_json, create_zip, directory_manifest, sha256_file

    out = output_root(cfg)
    required = [
        "preflight_report.json", "preflight_report.md", "environment.json", "cache/manifest.jsonl",
        "cache/cache_summary.json", "cache_report.json", "cache_report.md", "phase2_config_resolved.yaml",
        "training/train_history.csv", "training/dev_history.csv", "training/best_adapter.pt",
        "training/last_checkpoint.pt", "training/checkpoint_manifest.json", "training/early_stopping.json",
        "training/training_summary.json", "verification/checkpoint_verification.json",
        "verification/checkpoint_verification.md",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot bundle; missing artifacts: {missing}")
    manifest = {"root": str(out), "files": directory_manifest(out, {"artifact_manifest.json"})}
    atomic_write_json(out / "artifact_manifest.json", manifest)
    destination = Path(str(out) + ".zip")
    create_zip(out, destination)
    return {"status": "COMPLETE", "archive": str(destination), "archive_sha256": sha256_file(destination)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config(args)
    from .config import output_root
    from .diagnostics import configure_logging

    command_logger = configure_logging(output_root(cfg) / "logs" / f"{args.command}.log", f"phase2.cli.{args.command}")
    command_logger.info("command=%s config=%s data_root=%s output_root=%s offline=%s",
                        args.command, args.config, cfg["paths"]["data_root"], cfg["paths"]["output_root"], cfg["model"]["offline"])
    try:
        if args.command == "preflight":
            from .validation import run_preflight
            result = run_preflight(cfg)
        elif args.command == "cache-features":
            from .feature_cache import run_cache
            result = run_cache(cfg)
        elif args.command == "train":
            from .trainer import run_training
            result = run_training(cfg, args.resume)
        elif args.command == "verify":
            result = run_verify(cfg, args.checkpoint)
        elif args.command == "bundle":
            result = run_bundle(cfg)
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        command_logger.info("completed: %s", json.dumps(result, ensure_ascii=False, default=str))
        return 0
    except Exception as error:
        command_logger.exception("failed: %s", error)
        print(f"ERROR [{type(error).__name__}]: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

