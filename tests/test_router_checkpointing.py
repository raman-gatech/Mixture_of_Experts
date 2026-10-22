"""Real DeBERTa backward regression, run in the recovered training environment."""
import pytest
torch = pytest.importorskip("torch")

transformers = pytest.importorskip("transformers")

from moe_router.router_training import windowed_partial_loss


def test_deberta_shared_relative_embeddings_support_grouped_checkpoint_backward():
    torch.set_num_threads(2)
    config = transformers.DebertaV2Config(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                                          num_attention_heads=2, intermediate_size=64,
                                          max_position_embeddings=64, num_labels=4,
                                          relative_attention=True, pos_att_type=["p2c", "c2p"],
                                          norm_rel_ebd="layer_norm", share_att_key=True)
    model = transformers.DebertaV2ForSequenceClassification(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    inputs = torch.randint(0, 64, (3, 24))
    logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
    loss = windowed_partial_loss(logits, torch.tensor([0, 0, 1]), torch.tensor([3, -1]), torch.ones(5))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
