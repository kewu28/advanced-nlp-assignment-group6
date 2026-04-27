"""
run_scaling_grid.py  —  Scaling Law Grid mit Auto-Push
=======================================================
Trainiert alle 16 Modell-Kombinationen und pusht nach JEDEM
einzelnen Lauf automatisch auf GitHub.
"""

import subprocess
import json
import os
import re
import sys
import time
import argparse

MODEL_CONFIGS = {
    'XS': dict(n_layer=2, n_head=2, n_embd=64),
    'S':  dict(n_layer=4, n_head=4, n_embd=128),
    'M':  dict(n_layer=6, n_head=6, n_embd=384),
    'L':  dict(n_layer=8, n_head=8, n_embd=512),
}

DATASET_FRACS = [0.10, 0.25, 0.50, 1.00]
BATCH_SIZE   = 32
BLOCK_SIZE   = 256
RESULTS_FILE = 'scaling_results.json'


def estimate_params(n_layer, n_head, n_embd, block_size=256, vocab_size=65):
    pos_emb = n_embd * block_size
    tok_emb = n_embd * vocab_size
    per_blk = (n_embd + n_embd * 3 * n_embd + n_embd * n_embd +
                n_embd + n_embd * 4 * n_embd + 4 * n_embd * n_embd)
    return pos_emb + tok_emb + n_layer * per_blk + n_embd


def parse_final_losses(stdout):
    pattern = r'step\s+\d+:\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)'
    matches = re.findall(pattern, stdout)
    if matches:
        return float(matches[-1][0]), float(matches[-1][1])
    return None, None


def parse_all_val_losses(stdout):
    pattern = r'step\s+(\d+):\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)'
    return [(int(s), float(t), float(v)) for s, t, v in re.findall(pattern, stdout)]


def format_time(seconds):
    if seconds < 60: return f"{seconds:.0f}s"
    elif seconds < 3600: return f"{seconds/60:.1f}min"
    else: return f"{seconds/3600:.1f}h"


def git_push(run_key):
    print(f"  → Pushe auf GitHub...")
    result = subprocess.run(
        f'git add scaling_results.json && git commit -m "Grid: {run_key} done" && git push',
        shell=True, capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✓ GitHub aktualisiert")
    else:
        print(f"  ⚠ Push fehlgeschlagen: {result.stderr[-150:]}")


def run_grid(device, total_tokens):
    max_iters = total_tokens // (BATCH_SIZE * BLOCK_SIZE)

    print("=" * 70)
    print("SCALING LAW GRID — nanoGPT on Tiny Shakespeare (Auto-Push aktiv)")
    print("=" * 70)
    print(f"  Iso-Compute: {total_tokens:,} Tokens | max_iters: {max_iters} | Device: {device}")
    print("=" * 70)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        done = sum(1 for v in results.values() if not v.get('error'))
        print(f"\n→ {done} abgeschlossene Läufe geladen — werden übersprungen.\n")

    total_runs = len(MODEL_CONFIGS) * len(DATASET_FRACS)
    run_idx = 0
    t_total = time.time()

    for model_name, mcfg in MODEL_CONFIGS.items():
        for frac in DATASET_FRACS:
            run_idx += 1
            run_key    = f"{model_name}_{int(frac * 100)}"
            out_dir    = os.path.join('out', 'scaling', run_key)
            train_file = f"train_{int(frac * 100)}.bin"
            params     = estimate_params(**mcfg)
            flops      = 6 * params * total_tokens

            if run_key in results and not results[run_key].get('error'):
                r = results[run_key]
                print(f"[{run_idx:2d}/{total_runs}] {run_key:<12} ÜBERSPRUNGEN (val={r['val_loss']:.4f})")
                continue

            print(f"\n{'─'*70}")
            print(f"[{run_idx:2d}/{total_runs}]  {run_key}")
            print(f"  Model : n_layer={mcfg['n_layer']} n_head={mcfg['n_head']} n_embd={mcfg['n_embd']} → {params:,} params")
            print(f"  Data  : {train_file} ({int(frac*100)}%) | FLOPs ≈ {flops:.2e}")

            eval_interval = max(50, max_iters // 20)

            cmd = [
                sys.executable, 'train.py',
                f'--n_layer={mcfg["n_layer"]}', f'--n_head={mcfg["n_head"]}',
                f'--n_embd={mcfg["n_embd"]}', '--bias=False',
                '--dataset=shakespeare_char', f'--train_file={train_file}',
                f'--block_size={BLOCK_SIZE}', f'--batch_size={BATCH_SIZE}',
                f'--max_iters={max_iters}', f'--lr_decay_iters={max_iters}',
                '--learning_rate=1e-3', '--min_lr=1e-4', '--beta2=0.99',
                '--warmup_iters=100', '--weight_decay=0.1', '--grad_clip=1.0',
                '--dropout=0.0', f'--out_dir={out_dir}',
                f'--eval_interval={eval_interval}', '--eval_iters=50',
                '--log_interval=50', '--always_save_checkpoint=False',
                '--wandb_log=False', f'--device={device}', '--compile=False',
            ]

            t0   = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt   = time.time() - t0

            train_loss, val_loss = parse_final_losses(proc.stdout)
            curves = parse_all_val_losses(proc.stdout)

            if proc.returncode != 0 or val_loss is None:
                print(f"  ✗ FEHLER (returncode={proc.returncode})")
                err = (proc.stderr or proc.stdout)[-400:].strip()
                for line in err.split('\n')[-6:]:
                    print(f"    {line}")
                entry = {'model': model_name, 'frac': frac, 'error': True, 'stderr': err[-200:]}
            else:
                print(f"  ✓ Fertig in {format_time(dt)} — train={train_loss:.4f}  val={val_loss:.4f}")
                entry = {
                    'model': model_name, 'frac': frac,
                    'n_layer': mcfg['n_layer'], 'n_head': mcfg['n_head'], 'n_embd': mcfg['n_embd'],
                    'param_count': params, 'total_tokens': total_tokens,
                    'max_iters': max_iters, 'batch_size': BATCH_SIZE, 'block_size': BLOCK_SIZE,
                    'flops': flops, 'train_loss': train_loss, 'val_loss': val_loss,
                    'elapsed_s': dt, 'learning_curves': curves, 'error': False,
                }

            results[run_key] = entry

            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

            git_push(run_key)

    elapsed_total = time.time() - t_total
    successful = sum(1 for v in results.values() if not v.get('error'))
    print(f"\n{'='*70}")
    print(f"GRID ABGESCHLOSSEN — {successful}/{total_runs} erfolgreich in {format_time(elapsed_total)}")
    print("Weiter mit: python analyze_scaling.py")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--total_tokens', type=int, default=20_000_000)
    args = parser.parse_args()
    run_grid(device=args.device, total_tokens=args.total_tokens)
