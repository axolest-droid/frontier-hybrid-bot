import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|im_start|>", "<|im_end|>"]


def create_starter_datasets(data_dir: Path):
    """Creates local starter corpora so the full pipeline runs out-of-the-box."""
    data_dir.mkdir(parents=True, exist_ok=True)
    pretrain_path = data_dir / "sample_pretrain.txt"
    chat_path = data_dir / "sample_chat.jsonl"

    if not pretrain_path.exists():
        sample_docs = [
            "FrontierHybridMoE is a hybrid language model combining Gated Linear State-Space layers with Multi-Head Latent Attention.",
            "Sparse Mixture-of-Experts activates only a subset of specialized networks per token, making inference fast and memory efficient.",
            "Multi-Token Prediction trains a language model to predict future tokens beyond the immediate next token, improving planning.",
            "Python is a versatile programming language widely used in artificial intelligence, machine learning, and cybersecurity.",
            "Neural networks learn representations from data by minimizing cross-entropy loss using gradient descent and AdamW optimization.",
            "Rotary Position Embeddings encode relative token positions directly into attention queries and keys without extra parameters.",
            "A tokenizer splits raw text into subword units so a language model can process any sentence using a fixed vocabulary.",
            "Fine-tuning a pretrained model on conversational pairs turns a text predictor into an interactive AI assistant.",
        ] * 40
        pretrain_path.write_text("\n".join(sample_docs), encoding="utf-8")

    if not chat_path.exists():
        sample_chats = [
            {"user": "Hi! Who are you?", "assistant": "Hello! I am FrontierBot, an AI assistant built on the FrontierHybridMoE architecture."},
            {"user": "What architecture do you use?", "assistant": "I combine 75% Gated Linear SSM layers, 25% Multi-Head Latent Attention, Sparse MoE, and Multi-Token Prediction."},
            {"user": "What is Sparse MoE?", "assistant": "Sparse Mixture-of-Experts routes each token to only a few specialized experts, keeping compute low while scaling capacity."},
            {"user": "Explain Python in one sentence.", "assistant": "Python is a readable, high-level programming language popular for AI, web development, and automation."},
            {"user": "How do I train a language model?", "assistant": "First train a tokenizer, then pretrain on raw text for next-token prediction, and finally fine-tune on chat conversations."},
            {"user": "Say hello to GitHub!", "assistant": "Hello GitHub! Excited to be open-sourced and ready to learn from new datasets."},
        ] * 25
        with chat_path.open("w", encoding="utf-8") as f:
            for item in sample_chats:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return pretrain_path, chat_path


def download_hf_corpus(data_dir: Path, dataset_name: str, max_samples: int) -> Path:
    """Optionally pulls a real text dataset from HuggingFace (e.g., roneneldan/TinyStories)."""
    from datasets import load_dataset

    out_path = data_dir / "hf_corpus.txt"
    print(f"Downloading {max_samples} samples from {dataset_name}...")
    ds = load_dataset(dataset_name, split=f"train[:{max_samples}]")
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip().replace("\n", " ")
            if text:
                f.write(text + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Train a custom Byte-Level BPE Tokenizer")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Use 32000 for large corpora")
    parser.add_argument("--out_dir", type=str, default="tokenizer_out")
    parser.add_argument("--use_hf", action="store_true", help="Download TinyStories from HuggingFace")
    parser.add_argument("--hf_dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--hf_samples", type=int, default=20000)
    args = parser.parse_args()

    data_dir = Path("data")
    pretrain_path, chat_path = create_starter_datasets(data_dir)

    train_files = [str(pretrain_path), str(chat_path)]
    if args.use_hf:
        hf_path = download_hf_corpus(data_dir, args.hf_dataset, args.hf_samples)
        train_files.append(str(hf_path))

    tokenizer = Tokenizer(models.BPE(unk_token="<|pad|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train(files=train_files, trainer=trainer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "tokenizer.json"
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to {save_path} | Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
