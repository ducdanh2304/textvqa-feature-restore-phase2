import torch
import pytest

from textvqa_feature_restore.feature_cache import tensor_bytes, validate_cache_file
from textvqa_feature_restore.io_utils import sha256_bytes


def test_stale_cache_metadata_is_rejected(tmp_path):
    feature = torch.randn(3, 2048, dtype=torch.float16)
    metadata = {
        "model_revision": "old-revision",
        "feature_shape": [3, 2048],
        "feature_dtype": "torch.float16",
        "cached_feature_sha256": sha256_bytes(tensor_bytes(feature)),
    }
    path = tmp_path / "cache.pt"
    torch.save({"metadata": metadata, "feature": feature}, path)
    with pytest.raises(ValueError, match="Stale/mismatched"):
        validate_cache_file(path, {"model_revision": "required-revision"})


def test_corrupted_cached_tensor_is_rejected(tmp_path):
    feature = torch.ones(3, 2048, dtype=torch.float16)
    metadata = {
        "model_revision": "right",
        "feature_shape": [3, 2048],
        "feature_dtype": "torch.float16",
        "cached_feature_sha256": "0" * 64,
    }
    path = tmp_path / "cache.pt"
    torch.save({"metadata": metadata, "feature": feature}, path)
    with pytest.raises(ValueError, match="checksum"):
        validate_cache_file(path, {"model_revision": "right"})

