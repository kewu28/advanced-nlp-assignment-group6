"""
model_lora.py  —  Part 3: LoRA implementation for nanoGPT
==========================================================
Implements Low-Rank Adaptation (Hu et al., 2022) from scratch.

Key changes vs. model.py:
  1. LoRALinear class wraps any nn.Linear with trainable rank-decomposition matrices A and B.
  2. CausalSelfAttention is modified to split c_attn into separate q, k, v projections
     so LoRA can be applied selectively to Q and V only (as per the assignment spec).
  3. GPT.inject_lora() injects LoRA into all attention layers and freezes base weights.
  4. GPT.lora_state_dict() saves only the LoRA parameters (for lightweight checkpoints).

LoRA math:
  For a pretrained weight W (d_out x d_in), the adapted forward pass is:
      output = W @ x  +  (B @ A) @ x
  where A (r x d_in) is Gaussian-initialised and B (d_out x r) is zero-initialised,
  so the initial LoRA contribution is exactly zero — the model starts identical to pretrained.

Usage:
  # Load pretrained checkpoint and inject LoRA
  model = GPT(config)
  model.load_state_dict(checkpoint['model'])
  model.inject_lora(lora_rank=4)

  # Verify: only LoRA params are trainable
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total     = sum(p.numel() for p in model.parameters())
  print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ── LoRA Linear Layer ──────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a low-rank adaptation.

    forward(x) = original_linear(x)  +  (x @ A.T) @ B.T
                 ─────────────────────   ─────────────────
                   frozen pretrained        trainable LoRA
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 4):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False          # freeze base weights
        if self.original.bias is not None:
            self.original.bias.requires_grad = False        # freeze base bias too

        d_out, d_in = original_linear.weight.shape

        # A: initialised with small Gaussian  (rank x d_in)
        # B: initialised with zeros           (d_out x rank)
        # → initial LoRA output = B @ A @ x = 0 @ ... = 0
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out

    def extra_repr(self) -> str:
        d_out, d_in = self.original.weight.shape
        return f"in={d_in}, out={d_out}, rank={self.rank}"


# ── LayerNorm (unchanged) ──────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ── CausalSelfAttention with optional LoRA ─────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Modified attention: c_attn is split into q_proj, k_proj, v_proj so that
    LoRA can be applied selectively to Q and V only.

    When lora_rank=0 (default) the layer behaves identically to the original.
    Call inject_lora(rank) to wrap q_proj and v_proj with LoRALinear.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Split the combined QKV projection into three separate linears.
        # This is equivalent to the original c_attn but allows selective LoRA.
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )

    def inject_lora(self, rank: int):
        """Replace q_proj and v_proj with LoRALinear wrappers."""
        self.q_proj = LoRALinear(self.q_proj, rank=rank)
        self.v_proj = LoRALinear(self.v_proj, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ── MLP (unchanged) ────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ── Transformer Block (unchanged) ─────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    block_size: int   = 1024
    vocab_size: int   = 50304
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    dropout:    float = 0.0
    bias:       bool  = True


# ── GPT model with LoRA support ────────────────────────────────────────────────

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    # ── LoRA injection ─────────────────────────────────────────────────────────

    def inject_lora(self, lora_rank: int):
        """
        1. Freeze ALL parameters.
        2. Inject LoRALinear into q_proj and v_proj of every attention layer.
        3. Only the newly created lora_A and lora_B tensors will have requires_grad=True.
        """
        # Step 1: freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # Step 2: inject LoRA into every attention block
        for block in self.transformer.h:
            block.attn.inject_lora(rank=lora_rank)

        # Report
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"LoRA injected (rank={lora_rank})")
        print(f"  Trainable parameters : {trainable:,}")
        print(f"  Total parameters     : {total:,}")
        print(f"  Trainable fraction   : {trainable / total * 100:.4f}%")
        return trainable

    def lora_state_dict(self):
        """Returns only the LoRA parameters — for lightweight checkpointing."""
        return {k: v for k, v in self.state_dict().items()
                if 'lora_A' in k or 'lora_B' in k}

    def load_lora_state_dict(self, lora_sd):
        """Load LoRA parameters into an already-injected model."""
        missing, unexpected = self.load_state_dict(lora_sd, strict=False)
        lora_missing = [k for k in missing if 'lora' in k]
        if lora_missing:
            print(f"WARNING: Missing LoRA keys: {lora_missing}")
        return missing, unexpected

    # ── Standard GPT methods (unchanged) ──────────────────────────────────────

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t   = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss   = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Only optimise parameters that require gradients (i.e. LoRA params after injection)
        param_dict    = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params  = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params= [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups  = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay    = sum(p.numel() for p in decay_params)
        num_nodecay  = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused       = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas,
            **(dict(fused=True) if use_fused else {})
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token  = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter   = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved   = flops_per_iter * (1.0 / dt)
        flops_promised   = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
        return idx


# ── Sanity check helper ────────────────────────────────────────────────────────

def sanity_check(checkpoint_path: str, lora_rank: int = 4, device: str = 'cpu'):
    """
    Verifies two properties required by the assignment:
    1. Only LoRA parameters have requires_grad=True after injection.
    2. Model output is identical to pretrained before any LoRA training
       (because B is initialised to zero, so the LoRA contribution starts at 0).
    """
    print("=" * 60)
    print("LoRA Sanity Check")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = GPTConfig(**checkpoint['model_args'])
    model      = GPT(config)

    # Load pretrained weights — need to remap keys from original model.py
    # (original uses c_attn; our model uses q_proj/k_proj/v_proj)
    sd_pretrained = checkpoint['model']
    sd_new        = model.state_dict()
    sd_mapped     = {}

    for k, v in sd_pretrained.items():
        clean_k = k.replace('_orig_mod.', '')

        if 'attn.c_attn' in clean_k:
            # Split the combined QKV weight/bias into q, k, v
            layer_prefix = clean_k.replace('.c_attn.weight', '').replace('.c_attn.bias', '')
            n_embd = v.shape[1] if 'weight' in clean_k else v.shape[0] // 3

            if 'weight' in clean_k:
                q_w, k_w, v_w = v.split(n_embd, dim=0)
                sd_mapped[layer_prefix + '.q_proj.weight'] = q_w
                sd_mapped[layer_prefix + '.k_proj.weight'] = k_w
                sd_mapped[layer_prefix + '.v_proj.weight'] = v_w
            else:  # bias
                q_b, k_b, v_b = v.split(n_embd, dim=0)
                sd_mapped[layer_prefix + '.q_proj.bias'] = q_b
                sd_mapped[layer_prefix + '.k_proj.bias'] = k_b
                sd_mapped[layer_prefix + '.v_proj.bias'] = v_b
        else:
            if clean_k in sd_new:
                sd_mapped[clean_k] = v

    missing, unexpected = model.load_state_dict(sd_mapped, strict=False)
    print(f"Keys loaded: {len(sd_mapped)} | Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    # Test 1: output before LoRA injection
    model.eval()
    dummy_input = torch.randint(0, config.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        logits_before, _ = model(dummy_input)

    # Inject LoRA
    model.inject_lora(lora_rank=lora_rank)

    # Test 2: output after LoRA injection (should be identical — B=0)
    with torch.no_grad():
        logits_after, _ = model(dummy_input)

    outputs_match = torch.allclose(logits_before, logits_after, atol=1e-5)
    print(f"\nTest 1 — Only LoRA params trainable: ", end="")
    non_lora_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and 'lora' not in n
    ]
    print("PASS ✓" if len(non_lora_trainable) == 0 else f"FAIL ✗ — {non_lora_trainable[:3]}")

    print(f"Test 2 — Output identical before/after injection: ", end="")
    print("PASS ✓" if outputs_match else f"FAIL ✗ — max diff: {(logits_before - logits_after).abs().max():.2e}")

    print("=" * 60)
    return model


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        sanity_check(sys.argv[1], lora_rank=int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        print("Usage: python model_lora.py <checkpoint_path> [lora_rank]")
        print("Example: python model_lora.py out/scaling/M_100/ckpt.pt 4")
