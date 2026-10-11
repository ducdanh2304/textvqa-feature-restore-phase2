from pathlib import Path

import pandas as pd
import pytest

from textvqa_feature_restore.data import assert_disjoint, validate_manifest


FIXTURES = Path(__file__).parent / "fixtures"


def test_manifest_validation_detects_duplicate_ids(tmp_path):
    frame = pd.read_csv(FIXTURES / "tiny_train_manifest.csv", dtype=str)
    frame.loc[1, "image_id"] = frame.loc[0, "image_id"]
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate image_id"):
        validate_manifest(path, phase1_root=tmp_path, split="train", expected_rows=2, benchmark_ids=set(), verify_files=False)


def test_manifest_validation_rejects_benchmark_overlap(tmp_path):
    path = FIXTURES / "tiny_train_manifest.csv"
    with pytest.raises(ValueError, match="locked benchmark"):
        validate_manifest(path, phase1_root=tmp_path, split="train", expected_rows=2, benchmark_ids={"img001"}, verify_files=False)


def test_train_dev_question_overlap_is_rejected():
    train = pd.DataFrame({"image_id": ["a"], "question_id": ["q"]})
    dev = pd.DataFrame({"image_id": ["b"], "question_id": ["q"]})
    with pytest.raises(ValueError, match="question_id=1"):
        assert_disjoint(train, dev)

