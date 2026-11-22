import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Configuration
# =============================================================================
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 256          # Hidden dimension (scale to 2048–7168 for LLMs)
    n_layers: int = 8           # 6 Gated Linear SSM + 2 MLA layers (75% / 25%)
    n_heads: int = 8
    attn_interval: int = 4      # Every 4th layer uses Multi-Head Latent Attention

    # Multi-Head Latent Attention (MLA) dimensions
    d_nope: int = 16            # Content head dimension (no RoPE)
    d_rope: int = 16            # Positional head dimension (RoPE applied)
    d_cq: int = 96              # Compressed Query latent dimension
    d_ckv: int = 64             # Compressed Key-Value latent dimension (4x smaller)

    # Sparse Mixture-of-Experts (MoE) dimensions
    n_shared_experts: int = 1   # Always active on every token
    n_routed_experts: int = 8   # Pool of specialized experts
    top_k: int = 2              # Number of routed experts activated per token
    d_expert: int = 192         # SwiGLU hidden size per expert
    bias_update_speed: float = 0.001  # Loss-free load-balancing step size

    # Sequence & Multi-Token Prediction (MTP)
    max_seq_len: int = 2048
    mtp_loss_weight: float = 0.3


# =============================================================================
# 2. Core Primitives: RMSNorm, Decoupled RoPE, SwiGLU
# =============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Normalization: faster and more stable than LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(x.dtype)


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes cosine and sine rotation matrices for Rotary Position Embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (max_seq_len, dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies 2D pairwise rotations to tensor of shape (B, T, H, d_rope)."""
    seq_len = x.size(1)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2).to(x.dtype)  # (1, T, 1, d_rope // 2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2).to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    """Gated Feed-Forward Network: W_down(SiLU(W_gate(x)) * W_up(x))."""
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# =============================================================================
# 3. Sequence Mixer A: Multi-Head Latent Attention (MLA) — 25% of Layers
# =============================================================================
class MultiHeadLatentAttention(nn.Module):
    """
    Compresses Q and KV projections into low-rank latent spaces (c_Q, c_KV)
    and pairs them with decoupled RoPE subspaces. Uses PyTorch SDPA (FlashAttention).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_nope = cfg.d_nope
        self.d_rope = cfg.d_rope
        self.d_head = cfg.d_nope + cfg.d_rope

        # Query low-rank compression & up-projections
        self.w_dq = nn.Linear(cfg.d_model, cfg.d_cq, bias=False)
        self.q_norm = RMSNorm(cfg.d_cq)
        self.w_uq = nn.Linear(cfg.d_cq, cfg.n_heads * cfg.d_nope, bias=False)
        self.w_qr = nn.Linear(cfg.d_cq, cfg.n_heads * cfg.d_rope, bias=False)

        # Key-Value joint low-rank compression & up-projections
        self.w_dkv = nn.Linear(cfg.d_model, cfg.d_ckv, bias=False)
        self.kv_norm = RMSNorm(cfg.d_ckv)
        self.w_uk = nn.Linear(cfg.d_ckv, cfg.n_heads * cfg.d_nope, bias=False)
        self.w_uv = nn.Linear(cfg.d_ckv, cfg.n_heads * self.d_head, bias=False)

        # Shared decoupled RoPE key across heads
        self.w_kr = nn.Linear(cfg.d_model, cfg.d_rope, bias=False)
        self.w_o = nn.Linear(cfg.n_heads * self.d_head, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Compress into low-rank latent vectors
        c_q = self.q_norm(self.w_dq(x))
        c_kv = self.kv_norm(self.w_dkv(x))

        # Reconstruct content (nope) and positional (rope) parts
        q_nope = self.w_uq(c_q).view(B, T, self.n_heads, self.d_nope)
        q_rope = apply_rope(self.w_qr(c_q).view(B, T, self.n_heads, self.d_rope), cos, sin)

        k_nope = self.w_uk(c_kv).view(B, T, self.n_heads, self.d_nope)
        k_rope = apply_rope(self.w_kr(x).view(B, T, 1, self.d_rope), cos, sin)
        k_rope = k_rope.expand(-1, -1, self.n_heads, -1)

        v = self.w_uv(c_kv).view(B, T, self.n_heads, self.d_head)

        # Concatenate content + position subspaces -> (B, H, T, d_head)
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        k = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        # Automatically uses FlashAttention / Memory-Efficient kernel on CUDA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        return self.w_o(out)


# =============================================================================
# 4. Sequence Mixer B: Gated Selective Linear State-Space — 75% of Layers
# =============================================================================
class GatedLinearSSM(nn.Module):
    """
    O(N) sequence mixer with a fixed-size recurrent memory matrix S per head:
        S_t = alpha_t * S_{t-1} + k_t^T v_t
        y_t = q_t @ S_t
    No growing KV-cache required; state size stays constant regardless of context length.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_inner = cfg.d_model

        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=False)
        # Causal depthwise Conv1d for local token interactions
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=4, padding=3, groups=self.d_inner
        )
        self.qkv_proj = nn.Linear(self.d_inner, 3 * self.d_inner, bias=False)

        # Input-dependent selective forget gate (alpha_t in (0, 1))
        self.decay_proj = nn.Linear(self.d_inner, self.n_heads, bias=True)
        nn.init.constant_(self.decay_proj.bias, 2.0)  # Bias toward remembering early on

        self.head_norm = RMSNorm(self.d_head)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def forward(
        self, x: torch.Tensor, cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)

        # Causal local convolution (slice to :T so future tokens never leak)
        u = self.conv1d(u.transpose(1, 2))[:, :, :T].transpose(1, 2)
        u = F.silu(u)

        q, k, v = self.qkv_proj(u).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # L2-normalize q and k to keep the recurrent state norm strictly bounded
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        decay = torch.sigmoid(self.decay_proj(u)).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        # Fixed-size state matrix S: (B, H, d_head, d_head)
        state = torch.zeros(
            B, self.n_heads, self.d_head, self.d_head, device=x.device, dtype=x.dtype
        )
        outputs = []
        for t in range(T):
            state = decay[:, :, t] * state + torch.einsum("bhd,bhv->bhdv", k[:, :, t], v[:, :, t])
            outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, :, t], state))

        y = torch.stack(outputs, dim=2)  # (B, H, T, d_head)
        y = self.head_norm(y).transpose(1, 2).contiguous().view(B, T, self.d_inner)
        return self.out_proj(y * F.silu(gate))


# =============================================================================
# 5. Sparse Mixture-of-Experts with Shared Experts & Loss-Free Balancing
# =============================================================================
class SparseMoE(nn.Module):
    """
    Fine-grained MoE combining always-on shared experts with top-k routed experts.
    Balances expert utilization via a dynamic non-trainable router bias.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_routed = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.bias_lr = cfg.bias_update_speed

        self.shared_experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.d_expert) for _ in range(cfg.n_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.d_expert) for _ in range(cfg.n_routed_experts)]
        )
        self.router = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        # 1. Shared expert pathway (captures universal knowledge)
        shared_out = sum(expert(x) for expert in self.shared_experts)

        # 2. Sigmoid affinity scoring + bias-shifted top-k selection
        scores = torch.sigmoid(self.router(x_flat))
        topk_idx = torch.topk(scores + self.expert_bias, k=self.top_k, dim=-1).indices

        # Use unbiased scores for actual output weighting
        topk_weights = scores.gather(dim=-1, index=topk_idx)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 3. Sparse expert dispatch
        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            token_idx, slot_idx = torch.where(topk_idx == i)
            if token_idx.numel() > 0:
                expert_y = expert(x_flat[token_idx])
                w = topk_weights[token_idx, slot_idx].unsqueeze(-1)
                routed_out.index_add_(0, token_idx, (expert_y * w).to(routed_out.dtype))

        # 4. Auxiliary-loss-free load balancing update (no gradient interference)
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.view(-1), minlength=self.n_routed).float()
                avg_load = counts.mean()
                self.expert_bias += self.bias_lr * torch.sign(avg_load - counts)

        return shared_out + routed_out.view(B, T, D)


# =============================================================================
# 6. Hybrid Transformer-SSM-MoE Block & Full Model with MTP
# =============================================================================
class HybridBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, use_attention: bool, use_dense_ffn: bool):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = MultiHeadLatentAttention(cfg) if use_attention else GatedLinearSSM(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        # First layer stays dense for training stability; deeper layers are Sparse MoE
        self.ffn = SwiGLU(cfg.d_model, cfg.d_expert * 2) if use_dense_ffn else SparseMoE(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class FrontierHybridMoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.blocks = nn.ModuleList([
            HybridBlock(
                cfg,
                use_attention=((i + 1) % cfg.attn_interval == 0),
                use_dense_ffn=(i == 0),
            )
            for i in range(cfg.n_layers)
        ])

        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

        # Multi-Token Prediction (MTP) Head (predicts t+2 during training)
        self.mtp_norm_h = RMSNorm(cfg.d_model)
        self.mtp_norm_e = RMSNorm(cfg.d_model)
        self.mtp_proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.mtp_ffn = SwiGLU(cfg.d_model, cfg.d_expert * 2)
        self.mtp_out_norm = RMSNorm(cfg.d_model)

        cos, sin = precompute_rope(cfg.d_rope, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Restore SSM forget-gate bias to 2.0 so _init_weights doesn't zero it out
        for module in self.modules():
            if isinstance(module, GatedLinearSSM):
                nn.init.constant_(module.decay_proj.bias, 2.0)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)

        h = self.final_norm(x)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            main_loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), targets.view(-1), ignore_index=-100
            )

            # Multi-Token Prediction (SFT-safe: clamp -100 before embedding lookup)
            if input_ids.size(1) > 1 and (targets[:, 1:] != -100).any():
                safe_targets = targets[:, :-1].clamp(min=0)
                mtp_in = torch.cat(
                    [self.mtp_norm_h(h[:, :-1]), self.mtp_norm_e(self.tok_emb(safe_targets))],
                    dim=-1,
                )
                mtp_h = self.mtp_proj(mtp_in)
                mtp_h = self.mtp_out_norm(mtp_h + self.mtp_ffn(mtp_h))
                mtp_logits = self.lm_head(mtp_h)
                mtp_loss = F.cross_entropy(
                    mtp_logits.reshape(-1, self.cfg.vocab_size),
                    targets[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                loss = main_loss + self.cfg.mtp_loss_weight * mtp_loss
            else:
                loss = main_loss

        return logits, loss

    def parameter_summary(self) -> Tuple[int, int]:
        """Returns (total_parameters, active_inference_parameters_per_token)."""
        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        for module in self.modules():
            if isinstance(module, SparseMoE):
                one_expert = sum(p.numel() for p in module.routed_experts[0].parameters())
                inactive += (module.n_routed - module.top_k) * one_expert

        # MTP head is disabled during inference
        mtp_modules = [self.mtp_norm_h, self.mtp_norm_e, self.mtp_proj, self.mtp_ffn, self.mtp_out_norm]
        mtp_params = sum(p.numel() for m in mtp_modules for p in m.parameters())
        return total, total - inactive - mtp_params

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                logits.scatter_(1, sorted_idx, sorted_logits.masked_fill(remove_mask, float("-inf")))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return idx


# =============================================================================
# 7. Quick Verification & Training Smoke Test
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig()
    model = FrontierHybridMoE(cfg).to(device)

    total_p, active_p = model.parameter_summary()
    print(f"Device: {device.upper()}")
    print(f"Total Parameters:  {total_p / 1e6:.2f}M")
    print(f"Active / Token:    {active_p / 1e6:.2f}M ({100 * active_p / total_p:.1f}% active)")

    # Synthetic sequence training step with AdamW + Cosine schedule + Mixed Precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    batch_size, seq_len = 4, 64
    raw_tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len + 1), device=device)
    x_in, y_target = raw_tokens[:, :-1], raw_tokens[:, 1:]

    model.train()
    for step in range(1, 6):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
            _, loss = model(x_in, y_target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        print(f"Step {step} | Combined (Next-Token + MTP) Loss: {loss.item():.4f}")

    # Test autoregressive generation
    sample_prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    generated = model.generate(sample_prompt, max_new_tokens=16)
    print(f"Generated sequence shape: {tuple(generated.shape)}")
