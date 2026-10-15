from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from . import MODEL_ID, MODEL_REVISION
from .io_utils import atomic_write_text


class ConfigError(RuntimeError):
    pass


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(
    path: str | Path,
    *,
    data_root: str | None = None,
    output_root: str | None = None,
    model_path: str | None = None,
    offline: bool | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")
    cfg = _expand(copy.deepcopy(raw))
    if data_root is not None:
        cfg["paths"]["data_root"] = data_root
    if output_root is not None:
        cfg["paths"]["output_root"] = output_root
    if model_path is not None:
        cfg["model"]["path"] = model_path
    if offline is not None:
        cfg["model"]["offline"] = bool(offline)
    cfg["paths"]["data_root"] = str(Path(cfg["paths"]["data_root"]).expanduser().resolve())
    cfg["paths"]["output_root"] = str(Path(cfg["paths"]["output_root"]).expanduser().resolve())
    if cfg["model"].get("path"):
        cfg["model"]["path"] = str(Path(cfg["model"]["path"]).expanduser().resolve())
    elif cfg["model"].get("offline"):
        cfg["model"]["path"] = str(Path(cfg["paths"]["data_root"]) / "models" / "Qwen2.5-VL-3B-Instruct")
    cfg["_config_path"] = str(config_path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required_sections = {"paths", "model", "processor", "cache", "training", "hardware", "reproducibility"}
    missing = required_sections - set(cfg)
    if missing:
        raise ConfigError(f"Missing config sections: {sorted(missing)}")
    if cfg["model"]["id"] != MODEL_ID or cfg["model"]["revision"] != MODEL_REVISION:
        raise ConfigError("Model ID or revision differs from the frozen protocol")
    if cfg["model"]["attention_implementation"] != "sdpa":
        raise ConfigError("Default Phase-2 protocol requires SDPA")
    if cfg["model"]["visual_dtype"] != "float32" or cfg["model"]["language_dtype"] != "float16":
        raise ConfigError("Required dtype policy is visual=float32, language=float16")
    if bool(cfg["processor"]["use_fast"]):
        raise ConfigError("processor.use_fast must be false")
    if int(cfg["processor"]["min_pixels"]) != 256 * 256 or int(cfg["processor"]["max_pixels"]) != 786432:
        raise ConfigError("Frozen min/max pixel settings changed")
    if cfg["cache"]["storage_dtype"] not in {"float16", "bfloat16"}:
        raise ConfigError("cache.storage_dtype must be float16 or bfloat16")
    if int(cfg["cache"]["batch_size"]) not in {1, 2, 4, 8}:
        raise ConfigError("cache.batch_size must be one of 1, 2, 4, or 8; default protocol is 1")
    batch = cfg["training"]["batch_size"]
    if batch != "auto" and int(batch) not in {8, 16, 32}:
        raise ConfigError("training.batch_size must be auto, 8, 16, or 32")
    fixed = {"learning_rate": 1e-4, "weight_decay": 0.01, "max_epochs": 20, "early_stopping_patience": 4,
             "gradient_clip_norm": 1.0, "gradient_accumulation": 1}
    for key, value in fixed.items():
        if cfg["training"][key] != value:
            raise ConfigError(f"Frozen training value changed: {key}={cfg['training'][key]!r}, expected {value!r}")


def write_resolved_config(cfg: dict[str, Any], destination: str | Path) -> None:
    serializable = {key: value for key, value in cfg.items() if not key.startswith("_")}
    atomic_write_text(destination, yaml.safe_dump(serializable, sort_keys=True, allow_unicode=True))


def phase1_root(cfg: dict[str, Any]) -> Path:
    if cfg["paths"].get("phase1_root"):
        return Path(cfg["paths"]["phase1_root"])
    return Path(cfg["paths"]["data_root"]) / "phase1" / "textvqa_feature_restore_v1"


def output_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["output_root"])

