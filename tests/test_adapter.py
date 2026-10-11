import torch

from textvqa_feature_restore.adapter import ResidualFeatureRestorer, masked_feature_loss


def test_adapter_preserves_shape_and_is_identity_at_initialization():
    adapter = ResidualFeatureRestorer(hidden_size=2048, bottleneck_dim=256)
    value = torch.randn(2, 7, 2048)
    output = adapter(value)
    assert output.shape == value.shape
    assert torch.equal(output, value)


def test_masked_loss_ignores_padded_tokens():
    clean = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [999.0, 999.0]]])
    restored = torch.tensor([[[0.5, 0.0], [0.0, 0.5], [-999.0, -999.0]]])
    mask = torch.tensor([[True, True, False]])
    total, mse, cosine = masked_feature_loss(restored, clean, mask)
    expected_mse = torch.tensor(0.125)
    assert torch.allclose(mse, expected_mse)
    assert torch.allclose(cosine, torch.tensor(0.0))
    assert torch.allclose(total, expected_mse)

