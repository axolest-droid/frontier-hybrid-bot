import argparse
import torch
from tokenizers import Tokenizer

from model.neural_archi import FrontierHybridMoE, ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Chat with your FrontierHybridMoE Bot")
    parser.add_argument("--ckpt", type=str, default="checkpoints/chat_bot.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_out/tokenizer.json")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    im_end_id = tokenizer.token_to_id("<|im_end|>")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = ModelConfig(**ckpt["config"])
    model = FrontierHybridMoE(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("=" * 55)
    print(" FrontierBot Ready! (Type '/reset' to clear, '/exit' to quit)")
    print("=" * 55)

    history = ""
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() == "/exit":
            break
        if user_input.lower() == "/reset":
            history = ""
            print("[Conversation memory cleared]")
            continue

        prompt = history + f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)

        out_ids = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=im_end_id,
        )

        new_tokens = out_ids[0, len(input_ids):].tolist()
        if im_end_id in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(im_end_id)]

        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"Bot: {reply}")

        history = prompt + reply + "<|im_end|>\n"
        # Trim history to stay well within model context length
        if len(tokenizer.encode(history).ids) > cfg.max_seq_len - 64:
            history = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n"


if __name__ == "__main__":
    main()
