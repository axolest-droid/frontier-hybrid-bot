import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from dataset import ChatSFTDataset
from model.neural_archi import FrontierHybridMoE, ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Chat SFT Fine-Tuning")
    parser.add_argument("--base_ckpt", type=str, default="checkpoints/base_model.pt")
    parser.add_argument("--data", type=str, default="data/sample_chat.jsonl")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_out/tokenizer.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="checkpoints/chat_bot.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)

    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=True)
    cfg = ModelConfig(**ckpt["config"])
    model = FrontierHybridMoE(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    dataset = ChatSFTDataset(Path(args.data), tokenizer, max_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"SFT Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": ckpt["config"]}, out_path)
    print(f"Chat-tuned bot checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
