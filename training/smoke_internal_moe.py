"""CPU-sized reproducible mechanics check; synthetic tokens are not an LLM benchmark."""

import argparse
import json
import time
from pathlib import Path

import torch

from internal_moe.model import TinyMoELM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("audit/internal_moe_smoke.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = TinyMoELM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    tokens = torch.arange(16, device=device).repeat(8, 1)
    losses = []
    started = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        result = model(tokens, labels=tokens)
        result["loss"].backward()
        if not torch.isfinite(result["loss"]) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError("Non-finite loss or gradients")
        optimizer.step()
        losses.append(float(result["lm_loss"].detach()))
    model.eval()
    with torch.no_grad():
        result = model(tokens)
        accuracy = (result["logits"][:, :-1].argmax(-1) == tokens[:, 1:]).float().mean().item()
    report = dict(purpose="Synthetic overfit and routing mechanics only; no pretrained knowledge or quality claim",
                  seed=42, device=str(device), steps=args.steps, config=model.checkpoint_config(),
                  parameters=sum(p.numel() for p in model.parameters()), initial_lm_loss=losses[0],
                  final_lm_loss=losses[-1], synthetic_next_token_accuracy=accuracy,
                  expert_counts=result["expert_counts"].tolist(),
                  elapsed_seconds=time.perf_counter() - started,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save(dict(config=model.checkpoint_config(), state_dict=state), args.checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
