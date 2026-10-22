"""Loss on complete prompt groups, matching inference's maximum safety window."""
import torch


def windowed_partial_loss(logits, group_indices, targets, weights, source_weights=None):
    if logits.ndim != 2 or logits.shape[1] != 4 or len(logits) != len(group_indices):
        raise ValueError("Require four-way logits and matching window groups")
    probabilities = logits.softmax(-1)
    losses = []
    for index, target in enumerate(targets.tolist()):
        positions = (group_indices == index).nonzero(as_tuple=True)[0]
        if not len(positions):
            raise ValueError("Every prompt must have at least one window")
        safety = probabilities[positions, 3].max().clamp(1e-7, 1 - 1e-7)
        if target == 3:
            loss = -safety.log()
        elif target == -1:
            loss = -torch.log1p(-safety)
        elif target in (0, 1, 2):
            domain = logits[positions[0], :3].log_softmax(-1)[target]
            loss = -domain - torch.log1p(-safety)
        else:
            raise ValueError("Unknown partial target")
        losses.append(loss)
    row_weights = weights[torch.where(targets >= 0, targets, 4)]
    importance = torch.ones_like(row_weights) if source_weights is None else source_weights
    return (torch.stack(losses) * row_weights * importance).sum() / row_weights.sum()
