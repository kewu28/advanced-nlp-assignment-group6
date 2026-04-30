"""
train_lora.py  —  Part 3: LoRA rank ablation experiments
==========================================================
Runs Experiments 4, 5, and 6 from the assignment:

  Experiment 4: LoRA single-task fine-tuning (rank 4) on Task A and Task B
  Experiment 5: Rank ablation r ∈ {1, 2, 4, 8, 16} on Task A
  Experiment 6: LoRA vs. full fine-tuning comparison on Task A

NOTE: For Part 3 without Part 2 SFT data, this script uses the Shakespeare
validation set to measure catastrophic forgetting, and trains on the
Shakespeare training data to simulate fine-tuning. When Part 2 data is ready,
replace TRAIN_DATA_PATH and VAL_DATA_PATH with the SFT dataset paths.

Usage:
  python train_lora.py --experiment=5 --device=cuda
  python train_lora.py --experiment=6 --device=cuda
  python train_lora.py --experiment=all --device=cuda
"""

import os
import sys
import json
import time
import math
import pickle
import argparse
import numpy as np
import torch
from contextlib import nullcontext

from model_lora import GPT, GPTConfig


# ── Configuration ──────────────────────────────────────────────────────────────

CHECKPOINT_PATH = 'out/scaling/M_100/ckpt.pt'   # pretrained M model from Part 1
DATA_DIR        = 'data/shakespeare_char'
RESULTS_FILE    = 'lora_results.json'

# Training hyperparameters for LoRA fine-tuning
LORA_LR         = 1e-3
LORA_ITERS      = 1000
LORA_BATCH_SIZE = 32
LORA_BLOCK_SIZE = 256
LORA_EVAL_ITERS = 50
LORA_EVAL_EVERY = 100

# Full fine-tuning hyperparameters (Experiment 6)
FULL_FT_LR      = 1e-4
FULL_FT_ITERS   = 1000

# Rank ablation values (Experiment 5)
LORA_RANKS = [1, 2, 4, 8, 16]


# ── Data loading ───────────────────────────────────────────────────────────────

def get_batch(split, block_size, batch_size, data_dir, device, device_type):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data  = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    ix    = torch.randint(len(data) - block_size, (batch_size,))
    x     = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y     = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, block_size, batch_size, data_dir, device, device_type, ctx, eval_iters=50):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size, data_dir, device, device_type)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ── Checkpoint loading with key remapping ──────────────────────────────────────

def load_pretrained(checkpoint_path, device):
    """
    Load a checkpoint saved by the original model.py (which uses c_attn)
    into our model_lora.py (which uses q_proj/k_proj/v_proj).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = GPTConfig(**checkpoint['model_args'])
    model      = GPT(config)

    sd_pretrained = checkpoint['model']
    sd_new        = model.state_dict()
    sd_mapped     = {}

    for k, v in sd_pretrained.items():
        clean_k = k.replace('_orig_mod.', '')

        if 'attn.c_attn' in clean_k:
            layer_prefix = clean_k.rsplit('.c_attn.', 1)[0]
            suffix       = 'weight' if 'weight' in clean_k else 'bias'
            n_embd       = v.shape[1] if suffix == 'weight' else v.shape[0] // 3
            q, k_t, v_t  = v.split(n_embd, dim=0)
            sd_mapped[f"{layer_prefix}.q_proj.{suffix}"] = q
            sd_mapped[f"{layer_prefix}.k_proj.{suffix}"] = k_t
            sd_mapped[f"{layer_prefix}.v_proj.{suffix}"] = v_t
        elif clean_k in sd_new:
            sd_mapped[clean_k] = v

    missing, unexpected = model.load_state_dict(sd_mapped, strict=False)
    non_lora_missing = [m for m in missing if 'lora' not in m]
    if non_lora_missing:
        print(f"  Warning: {len(non_lora_missing)} non-LoRA keys missing from checkpoint")

    return model, config, checkpoint


# ── Training loop ──────────────────────────────────────────────────────────────

def train_run(model, config, run_name, max_iters, learning_rate,
              batch_size, block_size, data_dir, device, device_type, ctx):
    """Generic training loop. Returns final losses and learning curves."""

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type
    )

    # LR scheduler: cosine decay
    def get_lr(it):
        warmup = max(50, max_iters // 20)
        min_lr = learning_rate / 10
        if it < warmup:
            return learning_rate * (it + 1) / (warmup + 1)
        if it > max_iters:
            return min_lr
        decay_ratio = (it - warmup) / (max_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    curves = []
    t0 = time.time()
    X, Y = get_batch('train', block_size, batch_size, data_dir, device, device_type)

    for it in range(max_iters + 1):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if it % LORA_EVAL_EVERY == 0:
            losses = estimate_loss(
                model, block_size, batch_size, data_dir,
                device, device_type, ctx, LORA_EVAL_ITERS
            )
            elapsed = time.time() - t0
            print(f"  [{run_name}] step {it:4d}: "
                  f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                  f"({elapsed:.0f}s)")
            curves.append((it, losses['train'], losses['val']))

        if it == max_iters:
            break

        with ctx:
            _, loss = model(X, Y)
        X, Y = get_batch('train', block_size, batch_size, data_dir, device, device_type)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    final_losses = estimate_loss(
        model, block_size, batch_size, data_dir,
        device, device_type, ctx, LORA_EVAL_ITERS * 2
    )
    elapsed = time.time() - t0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'train_loss':       final_losses['train'],
        'val_loss':         final_losses['val'],
        'elapsed_s':        elapsed,
        'trainable_params': trainable_params,
        'learning_curves':  curves,
    }


# ── Experiments ────────────────────────────────────────────────────────────────

def run_experiment_5(device, device_type, ctx, results):
    """Rank ablation: r ∈ {1, 2, 4, 8, 16}"""
    print("\n" + "="*60)
    print("EXPERIMENT 5: LoRA Rank Ablation")
    print("="*60)

    for rank in LORA_RANKS:
        key = f"lora_rank_{rank}"
        if key in results and not results[key].get('error'):
            print(f"  rank={rank}: SKIPPED (val={results[key]['val_loss']:.4f})")
            continue

        print(f"\n  Training LoRA rank={rank}...")
        model, config, _ = load_pretrained(CHECKPOINT_PATH, device)
        model.to(device)
        trainable = model.inject_lora(lora_rank=rank)

        result = train_run(
            model, config, f"LoRA-r{rank}",
            max_iters=LORA_ITERS,
            learning_rate=LORA_LR,
            batch_size=LORA_BATCH_SIZE,
            block_size=LORA_BLOCK_SIZE,
            data_dir=DATA_DIR,
            device=device,
            device_type=device_type,
            ctx=ctx,
        )
        result['rank'] = rank
        result['method'] = 'lora'
        results[key] = result

        print(f"  rank={rank}: train={result['train_loss']:.4f}  "
              f"val={result['val_loss']:.4f}  "
              f"params={trainable:,}  "
              f"time={result['elapsed_s']/60:.1f}min")

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def run_experiment_6(device, device_type, ctx, results):
    """Full fine-tuning vs LoRA rank 4 comparison"""
    print("\n" + "="*60)
    print("EXPERIMENT 6: Full Fine-Tuning vs LoRA")
    print("="*60)

    # (a) Full fine-tuning
    key = 'full_finetune'
    if key not in results or results[key].get('error'):
        print("\n  Training: Full fine-tuning...")
        model, config, _ = load_pretrained(CHECKPOINT_PATH, device)
        model.to(device)
        # Unfreeze everything for full fine-tuning
        for p in model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        result = train_run(
            model, config, "FullFT",
            max_iters=FULL_FT_ITERS,
            learning_rate=FULL_FT_LR,
            batch_size=LORA_BATCH_SIZE,
            block_size=LORA_BLOCK_SIZE,
            data_dir=DATA_DIR,
            device=device,
            device_type=device_type,
            ctx=ctx,
        )
        result['rank'] = None
        result['method'] = 'full_finetune'
        results[key] = result
        print(f"  Full FT: train={result['train_loss']:.4f}  val={result['val_loss']:.4f}")

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    # (b) LoRA rank 4 — reuse from Experiment 5 if available
    if 'lora_rank_4' not in results:
        print("\n  Training: LoRA rank=4...")
        model, config, _ = load_pretrained(CHECKPOINT_PATH, device)
        model.to(device)
        model.inject_lora(lora_rank=4)

        result = train_run(
            model, config, "LoRA-r4",
            max_iters=LORA_ITERS,
            learning_rate=LORA_LR,
            batch_size=LORA_BATCH_SIZE,
            block_size=LORA_BLOCK_SIZE,
            data_dir=DATA_DIR,
            device=device,
            device_type=device_type,
            ctx=ctx,
        )
        result['rank'] = 4
        result['method'] = 'lora'
        results['lora_rank_4'] = result

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    return results


# ── Analysis and plots ────────────────────────────────────────────────────────

def analyze_results(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.dpi': 150, 'font.size': 11,
        'axes.grid': True, 'grid.alpha': 0.3,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # ── Experiment 5 Table ─────────────────────────────────────────────────────
    print("\nExperiment 5 — Rank Ablation:")
    print(f"  {'Method':<20} {'Rank':>6} {'Trainable Params':>18} {'Val Loss':>10} {'Time (min)':>12}")
    print("  " + "-"*70)

    rank_data = []
    for rank in LORA_RANKS:
        key = f"lora_rank_{rank}"
        v   = results.get(key)
        if v and not v.get('error'):
            tp = v['trainable_params']
            print(f"  {'LoRA':<20} {rank:>6} {tp:>18,} {v['val_loss']:>10.4f} {v['elapsed_s']/60:>12.1f}")
            rank_data.append((rank, tp, v['val_loss']))

    # ── Experiment 6 Table ─────────────────────────────────────────────────────
    print("\nExperiment 6 — Full FT vs LoRA:")
    print(f"  {'Method':<20} {'Rank':>6} {'Trainable Params':>18} {'Val Loss':>10} {'Time (min)':>12}")
    print("  " + "-"*70)

    ft = results.get('full_finetune')
    if ft:
        print(f"  {'Full Fine-Tuning':<20} {'—':>6} {ft['trainable_params']:>18,} "
              f"{ft['val_loss']:>10.4f} {ft['elapsed_s']/60:>12.1f}")

    lora4 = results.get('lora_rank_4')
    if lora4:
        print(f"  {'LoRA':<20} {4:>6} {lora4['trainable_params']:>18,} "
              f"{lora4['val_loss']:>10.4f} {lora4['elapsed_s']/60:>12.1f}")

    # ── Plot: Val Loss vs Rank ─────────────────────────────────────────────────
    if rank_data:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ranks  = [d[0] for d in rank_data]
        params = [d[1] for d in rank_data]
        losses = [d[2] for d in rank_data]

        ax = axes[0]
        ax.plot(ranks, losses, 'o-', color='#4C72B0', linewidth=2,
                markersize=9, markeredgecolor='white', markeredgewidth=1)
        for r, l in zip(ranks, losses):
            ax.annotate(f'{l:.4f}', (r, l),
                        textcoords='offset points', xytext=(5, 5), fontsize=9)

        # Full FT baseline
        if ft:
            ax.axhline(ft['val_loss'], color='#C44E52', linestyle='--',
                       linewidth=1.5, label=f"Full FT ({ft['val_loss']:.4f})")
            ax.legend(fontsize=9)

        ax.set_xlabel('LoRA Rank (r)', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Val Loss vs. LoRA Rank', fontsize=13, fontweight='bold')

        ax2 = axes[1]
        ax2.plot(params, losses, 's-', color='#55A868', linewidth=2,
                 markersize=9, markeredgecolor='white', markeredgewidth=1)
        for r, p, l in zip(ranks, params, losses):
            ax2.annotate(f'r={r}', (p, l),
                         textcoords='offset points', xytext=(5, 5), fontsize=9)

        if ft:
            ax2.axhline(ft['val_loss'], color='#C44E52', linestyle='--',
                        linewidth=1.5, label=f"Full FT ({ft['val_loss']:.4f})")
            ax2.axvline(ft['trainable_params'], color='#C44E52', linestyle=':',
                        alpha=0.5, linewidth=1)
            ax2.legend(fontsize=9)

        ax2.set_xscale('log')
        ax2.set_xlabel('Trainable Parameters', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title('Val Loss vs. Trainable Parameters', fontsize=13, fontweight='bold')

        plt.suptitle('LoRA Rank Ablation (nanoGPT on Tiny Shakespeare)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = 'lora_1_rank_ablation.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {fname}")

    # ── Discussion hints ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DISCUSSION HINTS")
    print("="*60)

    if rank_data and ft:
        best_lora = min(rank_data, key=lambda x: x[2])
        gap = best_lora[2] - ft['val_loss']
        print(f"\nQ7 — At what rank does LoRA match full fine-tuning?")
        print(f"  Best LoRA: rank={best_lora[0]}, val={best_lora[2]:.4f}")
        print(f"  Full FT:   val={ft['val_loss']:.4f}")
        print(f"  Gap:       {gap:+.4f}")
        print(f"  Trainable params ratio: "
              f"{best_lora[1]:,} / {ft['trainable_params']:,} = "
              f"{best_lora[1]/ft['trainable_params']*100:.2f}%")

        print(f"\nQ8 — Catastrophic forgetting comparison:")
        print(f"  Full FT val loss:   {ft['val_loss']:.4f}")
        if lora4:
            print(f"  LoRA r=4 val loss:  {lora4['val_loss']:.4f}")
        print(f"  (Lower = less forgetting of original Shakespeare distribution)")

    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LoRA Training — Part 3')
    parser.add_argument('--experiment', default='all',
                        choices=['4', '5', '6', 'all'],
                        help='Which experiment to run (default: all)')
    parser.add_argument('--device', default='cuda',
                        help='cpu or cuda (default: cuda)')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH,
                        help=f'Path to pretrained checkpoint (default: {CHECKPOINT_PATH})')
    args = parser.parse_args()

    global CHECKPOINT_PATH
    CHECKPOINT_PATH = args.checkpoint

    device      = args.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype       = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype     = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx         = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        print("        Please run Part 1 grid first to generate the M_100 checkpoint.")
        sys.exit(1)

    print(f"Device: {device} | Checkpoint: {CHECKPOINT_PATH}")

    # Load existing results
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {RESULTS_FILE}")

    torch.manual_seed(1337)

    exps = ['5', '6'] if args.experiment == 'all' else [args.experiment]

    if '5' in exps:
        results = run_experiment_5(device, device_type, ctx, results)
    if '6' in exps:
        results = run_experiment_6(device, device_type, ctx, results)

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    analyze_results(results)
    print(f"\nAll results saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()
