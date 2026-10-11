import pytest

from textvqa_feature_restore.checkpointing import REQUIRED_CHECKPOINT_FIELDS, assert_checkpoint_fields, validate_checkpoint_provenance


def complete_payload():
    payload = {field: None for field in REQUIRED_CHECKPOINT_FIELDS}
    payload.update(model_id="model", model_revision="revision", batch_size=16)
    return payload


def test_checkpoint_payload_required_fields_exist():
    assert_checkpoint_fields(complete_payload())
    payload = complete_payload()
    payload.pop("cache_manifest_sha256")
    with pytest.raises(ValueError, match="missing required fields"):
        assert_checkpoint_fields(payload)


def test_checkpoint_provenance_mismatch_is_rejected():
    payload = complete_payload()
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_checkpoint_provenance(payload, {"model_revision": "different", "batch_size": 16})

