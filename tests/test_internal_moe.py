import pytest

torch = pytest.importorskip("torch")
from torch import nn

from internal_moe.layers import SparseMoE
from internal_moe.model import TinyMoEConfig, TinyMoELM


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_upcycling_preserves_dense_output_and_dispatch(top_k):
    torch.manual_seed(7)
    dense = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
    moe = SparseMoE.from_dense(dense, 8, 3, top_k)
    inputs = torch.randn(2, 5, 8)
    actual, _, _, counts = moe(inputs)
    torch.testing.assert_close(actual, dense(inputs))
    assert counts.sum().item() == 10 * top_k
    assert moe.experts[0][0].weight.data_ptr() != moe.experts[1][0].weight.data_ptr()


def test_gate_and_selected_experts_receive_gradients():
    torch.manual_seed(4)
    moe = SparseMoE(8, [nn.Linear(8, 8) for _ in range(3)], top_k=2)
    output, balance, z_loss, counts = moe(torch.randn(2, 6, 8))
    (output.square().mean() + 0.01 * balance + 0.001 * z_loss).backward()
    assert moe.gate.weight.grad.abs().sum() > 0
    assert torch.isfinite(moe.gate.weight.grad).all()
    for count, expert in zip(counts, moe.experts):
        if count:
            assert expert.weight.grad.abs().sum() > 0


def test_decoder_cannot_read_future_tokens():
    torch.manual_seed(12)
    model = TinyMoELM(TinyMoEConfig(num_layers=1)).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 9, 10]])
    with torch.no_grad():
        torch.testing.assert_close(model(first)["logits"][:, :3], model(second)["logits"][:, :3])


def test_padding_excluded_from_router_losses():
    moe = SparseMoE(8, [nn.Linear(8, 8) for _ in range(3)])
    inputs = torch.randn(1, 4, 8)
    _, balance, z_loss, _ = moe(inputs, torch.tensor([[True, True, False, False]]))
    _, expected_balance, expected_z, _ = moe(inputs[:, :2])
    torch.testing.assert_close(balance, expected_balance)
    torch.testing.assert_close(z_loss, expected_z)


def test_checkpoint_round_trip(tmp_path):
    model = TinyMoELM(TinyMoEConfig(num_layers=1)).eval()
    checkpoint = tmp_path / "tiny.pt"
    torch.save(dict(config=model.checkpoint_config(), state_dict=model.state_dict()), checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    restored = TinyMoELM(TinyMoEConfig(**saved["config"])).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    inputs = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(model(inputs)["logits"], restored(inputs)["logits"])


def test_no_training_targets_rejected():
    model = TinyMoELM(TinyMoEConfig(num_layers=1))
    inputs = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="unmasked target"):
        model(inputs, labels=inputs, token_mask=torch.zeros_like(inputs, dtype=torch.bool))
