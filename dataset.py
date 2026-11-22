import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class PretrainDataset(Dataset):
    """Packs raw text documents into contiguous (seq_len) training blocks."""

    def __init__(self, text_path: Path, tokenizer: Tokenizer, seq_len: int = 256):
        self.seq_len = seq_len
        bos_id = tokenizer.token_to_id("<|bos|>")
        eos_id = tokenizer.token_to_id("<|eos|>")

        all_ids = []
        with text_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line).ids
                all_ids.extend([bos_id] + ids + [eos_id])

        # Ensure at least one full training window even on tiny smoke-test corpora
        while len(all_ids) < seq_len + 2:
            all_ids.extend(all_ids)

        self.tokens = torch.tensor(all_ids, dtype=torch.long)
        self.stride = max(1, seq_len // 2)
        self.num_samples = (len(self.tokens) - 1 - seq_len) // self.stride + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


class ChatSFTDataset(Dataset):
    """
    Formats user/assistant pairs in ChatML style and masks user prompt tokens
    with -100 so loss is computed strictly on the assistant's response.
    """

    def __init__(self, jsonl_path: Path, tokenizer: Tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("<|pad|>")
        self.samples = []

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.samples[idx]
        prompt_text = f"<|im_start|>user\n{row['user']}<|im_end|>\n<|im_start|>assistant\n"
        reply_text = f"{row['assistant']}<|im_end|>"

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        reply_ids = self.tokenizer.encode(reply_text).ids

        # Guarantee the assistant reply fits inside max_len + 1
        max_prompt = max(1, self.max_len + 1 - len(reply_ids))
        prompt_ids = prompt_ids[-max_prompt:]
        full = (prompt_ids + reply_ids)[: self.max_len + 1]

        x = full[:-1]
        y = full[1:]

        # Mask out prompt positions so model learns only to generate the answer
        mask_len = max(0, len(prompt_ids) - 1)
        y = [-100] * mask_len + y[mask_len:]

        # Right-pad to fixed max_len for batching
        pad_needed = self.max_len - len(x)
        x = x + [self.pad_id] * pad_needed
        y = y + [-100] * pad_needed

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
