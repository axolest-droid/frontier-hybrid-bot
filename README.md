# FrontierHybridMoE Bot

An end-to-end PyTorch implementation of a custom hybrid Language Model and interactive AI Chatbot combining:

- **75% Gated Linear SSM Layers** — $O(N)$ recurrent sequence mixing with fixed-size state memory
- **25% Multi-Head Latent Attention (MLA)** — Low-rank Q/KV compression with decoupled RoPE
- **Loss-Free Balanced Sparse MoE** — Shared + top-$k$ routed SwiGLU experts balanced via dynamic router bias
- **Multi-Token Prediction (MTP)** — Auxiliary $t+2$ prediction head during training

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train custom BPE tokenizer & generate starter datasets
python train_tokenizer.py --vocab_size 2048

# 2. Pretrain the base model (use --preset base on GPU)
python pretrain.py --preset tiny --epochs 5

# 3. Fine-tune for chat (SFT with ChatML prompt masking)
python sft_train.py --epochs 15

# 4. Launch interactive chatbot
python chat.py
