import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from dataset import PretrainDataset
from model.neural_archi import FrontierHybridMoE, ModelConfig


def get_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_config(preset: str, vocab_size: int, seq_len: int) -> ModelConfig:
    if preset == "tiny":
        return ModelConfig(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=4,
            n_heads=4,
            attn_interval=4,
            d_nope=16,
            d_rope=16,
            d_cq=48,
            d_ckv=32,
            n_shared_experts=1,
            n_routed_experts=4,
            top_k=2,
            d_expert=96,
            max_seq_len=seq_len,
        )
    # Default "base" matches your exact neural_archi.py specification
    return ModelConfig(vocab_size=vocab_size, max_seq_len=seq_len)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Pretrain FrontierHybridMoE")
    parser.add_argument("--preset", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--data", type=str, default="data/sample_pretrain.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_out/tokenizer.json")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", type=str, default="checkpoints/base_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    cfg = build_config(args.preset, vocab_size, args.seq_len)
    model = FrontierHybridMoE(cfg).to(device)
    total_p, active_p = model.parameter_summary()
    print(f"Device: {device.upper()} | Total: {total_p/1e6:.2f}M | Active/Token: {active_p/1e6:.2f}M")

    dataset = PretrainDataset(Path(args.data), tokenizer, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp_dtype == torch.float16))

    total_steps = args.epochs * len(loader)
    warmup_steps = max(1, int(0.1 * total_steps))
    global_step = 0

    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            lr = get_lr(global_step, total_steps, warmup_steps, args.lr, args.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, out_path)
    print(f"Pretrained checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
