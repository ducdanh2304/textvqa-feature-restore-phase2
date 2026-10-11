from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from . import MODEL_ID, MODEL_REVISION


class FrozenQwenVisualExtractor:
    def __init__(self, cfg: dict[str, Any]) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg = cfg
        self.device = torch.device("cuda:0")
        source = cfg["model"].get("path") or MODEL_ID
        offline = bool(cfg["model"].get("offline"))
        kwargs = {"local_files_only": offline}
        if not cfg["model"].get("path"):
            kwargs["revision"] = MODEL_REVISION
        self.processor = AutoProcessor.from_pretrained(
            source,
            min_pixels=int(cfg["processor"]["min_pixels"]),
            max_pixels=int(cfg["processor"]["max_pixels"]),
            use_fast=False,
            **kwargs,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source,
            dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            **kwargs,
        ).to(self.device).eval()
        self.model.model.visual.to(device=self.device, dtype=torch.float32)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        if not hasattr(self.model.model, "visual"):
            raise RuntimeError("Runtime model has no model.model.visual injection point")
        if next(self.model.model.visual.parameters()).dtype != torch.float32:
            raise RuntimeError("Visual tower is not FP32")
        if next(self.model.model.language_model.parameters()).dtype != torch.float16:
            raise RuntimeError("Language model is not FP16")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Qwen contains trainable parameters")
        source = inspect.getsource(self.model.model.get_image_features)
        if "self.visual" not in source:
            raise RuntimeError("Installed get_image_features no longer calls self.visual")
        if int(self.model.get_input_embeddings().embedding_dim) != 2048:
            raise RuntimeError("Runtime language hidden dimension is not 2048")

    @property
    def processor_metadata(self) -> dict[str, Any]:
        return {
            "processor_use_fast": False,
            "min_pixels": int(self.cfg["processor"]["min_pixels"]),
            "max_pixels": int(self.cfg["processor"]["max_pixels"]),
            "attention_implementation": "sdpa",
            "visual_dtype": "torch.float32",
        }

    def extract_batch(self, paths: Sequence[Path]) -> list[torch.Tensor]:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        batch = self.processor.image_processor(images=images, return_tensors="pt").to(self.device)
        batch["pixel_values"] = batch["pixel_values"].to(torch.float32)
        with torch.inference_mode():
            pieces = self.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
        result = [piece.detach().float().cpu().contiguous() for piece in pieces]
        if len(result) != len(paths):
            raise RuntimeError(f"Feature extractor returned {len(result)} tensors for {len(paths)} images")
        for path, feature in zip(paths, result):
            if feature.ndim != 2 or feature.shape[-1] != 2048:
                raise RuntimeError(f"Unexpected visual feature shape for {path}: {tuple(feature.shape)}")
            if not torch.isfinite(feature).all():
                raise RuntimeError(f"Non-finite FP32 visual feature for {path}")
        return result

    def qwen_gradient_count(self) -> int:
        return sum(parameter.grad is not None for parameter in self.model.parameters())

