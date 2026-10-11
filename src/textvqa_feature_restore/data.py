from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

from .io_utils import sha256_file


REQUIRED_MANIFEST_COLUMNS = {
    "image_id", "question_id", "question", "answers_json", "clean_image_path", "degraded_image_path",
    "degradation_type", "degradation_level", "degradation_parameters_json", "seed", "clean_sha256", "degraded_sha256",
}
ALLOWED_DEGRADATIONS = {"defocus_blur", "motion_camera_shake"}


@dataclass(frozen=True)
class PairRecord:
    split: str
    image_id: str
    question_id: str
    question: str
    answers_json: str
    clean_path: Path
    degraded_path: Path
    clean_sha256: str
    degraded_sha256: str
    degradation_type: str
    degradation_level: str
    source_width: int
    source_height: int


def read_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _validate_answers(value: str, question_id: str) -> None:
    try:
        answers = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid answers_json for question_id={question_id}") from exc
    if not isinstance(answers, list) or len(answers) != 10 or any(not isinstance(item, str) for item in answers):
        raise ValueError(f"answers_json must be exactly 10 strings for question_id={question_id}")


def canonical_image_path(phase1_root: Path, split: str, condition: str, image_id: str) -> Path:
    extension = ".jpg" if condition == "clean" else ".png"
    return phase1_root / "data" / split / condition / f"{image_id}{extension}"


def validate_manifest(
    path: str | Path,
    *,
    phase1_root: str | Path,
    split: str,
    expected_rows: int,
    benchmark_ids: set[str],
    verify_files: bool = True,
) -> tuple[pd.DataFrame, list[PairRecord]]:
    frame = read_manifest(path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{split} manifest missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"{split} manifest has {len(frame)} rows; expected {expected_rows}")
    if frame["image_id"].nunique() != expected_rows:
        raise ValueError(f"{split} manifest contains duplicate image_id")
    if frame["question_id"].nunique() != expected_rows:
        raise ValueError(f"{split} manifest contains duplicate question_id")
    if set(frame["image_id"]) & benchmark_ids:
        raise ValueError(f"{split} manifest overlaps the locked benchmark")
    if not set(frame["degradation_type"]).issubset(ALLOWED_DEGRADATIONS):
        raise ValueError(f"{split} manifest has an invalid degradation_type")
    if set(frame["degradation_level"]) != {"L2"}:
        raise ValueError(f"{split} manifest must contain only L2")

    phase1 = Path(phase1_root)
    records: list[PairRecord] = []
    for row in frame.itertuples(index=False):
        image_id, question_id = str(row.image_id), str(row.question_id)
        _validate_answers(row.answers_json, question_id)
        clean = canonical_image_path(phase1, split, "clean", image_id)
        degraded = canonical_image_path(phase1, split, "degraded", image_id)
        if clean.stem != image_id or degraded.stem != image_id:
            raise ValueError(f"Filename stem differs from image_id={image_id}")
        if verify_files:
            if not clean.is_file() or not degraded.is_file():
                raise FileNotFoundError(f"Missing pair for {split}/{image_id}: {clean}, {degraded}")
            if sha256_file(clean) != row.clean_sha256:
                raise ValueError(f"Clean SHA256 mismatch for {split}/{image_id}")
            if sha256_file(degraded) != row.degraded_sha256:
                raise ValueError(f"Degraded SHA256 mismatch for {split}/{image_id}")
            with Image.open(clean) as image:
                clean_size = image.size
                image.verify()
            with Image.open(degraded) as image:
                degraded_size = image.size
                image.verify()
            if clean_size != degraded_size:
                raise ValueError(f"Clean/degraded dimensions differ for {split}/{image_id}")
            width, height = clean_size
        else:
            width = height = -1
        records.append(PairRecord(
            split=split, image_id=image_id, question_id=question_id, question=str(row.question),
            answers_json=str(row.answers_json), clean_path=clean, degraded_path=degraded,
            clean_sha256=str(row.clean_sha256), degraded_sha256=str(row.degraded_sha256),
            degradation_type=str(row.degradation_type), degradation_level=str(row.degradation_level),
            source_width=int(width), source_height=int(height),
        ))
    return frame, records


def assert_disjoint(train: pd.DataFrame, dev: pd.DataFrame) -> None:
    image_overlap = set(train["image_id"]) & set(dev["image_id"])
    question_overlap = set(train["question_id"]) & set(dev["question_id"])
    if image_overlap or question_overlap:
        raise ValueError(f"Train/dev overlap: image_id={len(image_overlap)}, question_id={len(question_overlap)}")


def iter_conditions(records: Iterable[PairRecord]) -> Iterable[tuple[PairRecord, str, Path, str]]:
    for record in records:
        yield record, "clean", record.clean_path, record.clean_sha256
        yield record, "degraded", record.degraded_path, record.degraded_sha256

