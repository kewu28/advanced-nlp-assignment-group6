"""
run_scaling_grid.py  —  Schritt 2 der Scaling-Law-Pipeline
============================================================
Trainiert automatisch alle 16 Modell-Kombinationen (4 Größen × 4 Datenfraktionen)
und speichert Ergebnisse in scaling_results.json.

Iso-Compute-Prinzip:
    Jeder Lauf sieht exakt TOTAL_TOKENS Token (unabhängig von der Datenmenge).
    Kleine Datensätze werden dabei mehrfach durchlaufen → Overfitting ist gewollt.
    FLOPs ≈ 6 × N × TOTAL_TOKENS  (variiert nur nach Modellgröße N)

Ausführung:
    python run_scaling_grid.py                  # CPU, Standard-Budget
    python run_scaling_grid.py --device=cuda    # GPU (deutlich schneller)
    python run_scaling_grid.py --device=cuda --total_tokens=50000000

Laufzeit-Schätzung (CPU):
    XS (~0.1M): ~3 min/Lauf   →  12 min gesamt
    S  (~0.8M): ~8 min/Lauf   →  32 min gesamt
    M  (~10M):  ~40 min/Lauf  →  2-3 Std gesamt  → am besten GPU nutzen!
    L  (~25M):  ~90 min/Lauf  →  Nur auf GPU sinnvoll
"""

import subprocess
import json
import os
import re
import sys
import time
import argparse

# ── Modell-Grid ────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'XS': dict(n_layer=2, n_head=2, n_embd=64),
    'S':  dict(n_layer=4, n_head=4, n_embd=128),
    'M':  dict(n_layer=6, n_head=6, n_embd=384),
    'L':  dict(n_layer=8, n_head=8, n_embd=512),
}

DATASET_FRACS = [0.10, 0.25, 0.50, 1.00]

# ── Training-Hyperparameter ────────────────────────────────────────────────────

BATCH_SIZE    = 32
BLOCK_SIZE    = 256
TOTAL_TOKENS  = 10_000_000   # Iso-Compute-Budget (10M Tokens)
DEVICE        = 'cpu'        # Überschreiben mit --device=cuda
RESULTS_FILE  = 'scaling_results.json'

# Abgeleitete Werte
MAX_ITERS     = TOTAL_TOKENS // (BATCH_SIZE * BLOCK_SIZE)   # ≈ 1220


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def estimate_params(n_layer, n_head, n_embd, block_size=256, vocab_size=65):
    """Analytische Parameteranzahl (identisch mit prepare_subsets.py)."""
    pos_emb  = n_embd * block_size
    tok_emb  = n_embd * vocab_size
    per_blk  = (n_embd +                    # attn LN
                n_embd * 3 * n_embd +        # KQV
                n_embd * n_embd +            # attn proj
                n_embd +                    # mlp LN
                n_embd * 4 * n_embd +        # mlp expand
                4 * n_embd * n_embd)         # mlp contract
    return pos_emb + tok_emb + n_layer * per_blk + n_embd   # +n_embd: ln_f


def parse_final_losses(stdout: str):
    """
    Extrahiert den letzten geloggten Train/Val-Loss aus train.py-Ausgabe.
    Format: 'step N: train loss X.XXXX, val loss Y.YYYY'
    """
    pattern = r'step\s+\d+:\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)'
    matches  = re.findall(pattern, stdout)
    if matches:
        train_l, val_l = matches[-1]
        return float(train_l), float(val_l)
    return None, None


def parse_all_val_losses(stdout: str):
    """
    Extrahiert alle Val-Loss-Werte (für Lernkurven).
    Gibt Liste von (step, train_loss, val_loss) zurück.
    """
    pattern = r'step\s+(\d+):\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)'
    return [(int(s), float(t), float(v)) for s, t, v in re.findall(pattern, stdout)]


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


# ── Hauptprogramm ──────────────────────────────────────────────────────────────

def run_grid(device: str, total_tokens: int):
    global MAX_ITERS
    MAX_ITERS = total_tokens // (BATCH_SIZE * BLOCK_SIZE)

    print("=" * 70)
    print("SCALING LAW GRID — nanoGPT on Tiny Shakespeare")
    print("=" * 70)
    print(f"  Modelle:        {list(MODEL_CONFIGS.keys())}")
    print(f"  Datenfraktionen:{[f'{int(f*100)}%' for f in DATASET_FRACS]}")
    print(f"  Iso-Compute:    {total_tokens:,} Tokens pro Lauf")
    print(f"  max_iters:      {MAX_ITERS}  (batch={BATCH_SIZE}, block={BLOCK_SIZE})")
    print(f"  Device:         {device}")
    print(f"  Ergebnisdatei:  {RESULTS_FILE}")
    print("=" * 70)

    # Vorhandene Ergebnisse laden (ermöglicht Fortsetzen unterbrochener Läufe)
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"\n→ {len(results)} Ergebnisse aus vorherigem Lauf geladen.\n")

    total_runs = len(MODEL_CONFIGS) * len(DATASET_FRACS)
    run_idx    = 0
    t_total    = time.time()

    for model_name, mcfg in MODEL_CONFIGS.items():
        for frac in DATASET_FRACS:
            run_idx  += 1
            run_key   = f"{model_name}_{int(frac * 100)}"
            out_dir   = os.path.join('out', 'scaling', run_key)
            train_file = f"train_{int(frac * 100)}.bin"
            params    = estimate_params(**mcfg)
            flops     = 6 * params * total_tokens

            # ── Bereits vorhanden? ──────────────────────────────────────────
            if run_key in results and not results[run_key].get('error'):
                r = results[run_key]
                print(f"[{run_idx:2d}/{total_runs}] {run_key:<12}  "
                      f"ÜBERSPRUNGEN  (val={r['val_loss']:.4f})")
                continue

            # ── Lauf-Header ─────────────────────────────────────────────────
            print(f"\n{'─'*70}")
            print(f"[{run_idx:2d}/{total_runs}]  {run_key}")
            print(f"  Model   : n_layer={mcfg['n_layer']}  n_head={mcfg['n_head']}  "
                  f"n_embd={mcfg['n_embd']}  → {params:,} params")
            print(f"  Data    : {train_file}  ({int(frac*100)}% von vollem Trainingsset)")
            print(f"  Iters   : {MAX_ITERS}  (Tokens gesehen ≈ {total_tokens:,})")
            print(f"  FLOPs   : ≈ {flops:.2e}  (6 × N × D)")

            # ── eval_interval: maximal 20 Checkpoints über den gesamten Lauf ──
            eval_interval = max(50, MAX_ITERS // 20)

            cmd = [
                sys.executable, 'train.py',
                # Architektur
                f'--n_layer={mcfg["n_layer"]}',
                f'--n_head={mcfg["n_head"]}',
                f'--n_embd={mcfg["n_embd"]}',
                '--bias=False',
                # Daten & Tokenisierung
                '--dataset=shakespeare_char',
                f'--train_file={train_file}',        # ← neue Variable in train.py
                f'--block_size={BLOCK_SIZE}',
                # Training
                f'--batch_size={BATCH_SIZE}',
                f'--max_iters={MAX_ITERS}',
                f'--lr_decay_iters={MAX_ITERS}',
                '--learning_rate=1e-3',
                '--min_lr=1e-4',
                '--beta2=0.99',
                '--warmup_iters=100',
                '--weight_decay=0.1',
                '--grad_clip=1.0',
                '--dropout=0.0',
                # Logging & Checkpoints
                f'--out_dir={out_dir}',
                f'--eval_interval={eval_interval}',
                '--eval_iters=50',
                '--log_interval=50',
                '--always_save_checkpoint=False',
                '--wandb_log=False',
                # System
                f'--device={device}',
                '--compile=False',   # compile=False für Kompatibilität auf CPU
            ]

            t0  = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt  = time.time() - t0

            # ── Ergebnis parsen ─────────────────────────────────────────────
            train_loss, val_loss = parse_final_losses(proc.stdout)
            curves = parse_all_val_losses(proc.stdout)

            if proc.returncode != 0 or val_loss is None:
                print(f"  ✗ FEHLER  (returncode={proc.returncode})")
                # Letzte 500 Zeichen des Fehlers anzeigen
                err = (proc.stderr or proc.stdout)[-600:].strip()
                for line in err.split('\n')[-10:]:
                    print(f"    {line}")
                entry = {
                    'model': model_name, 'frac': frac,
                    'error': True, 'stderr': err[-200:],
                }
            else:
                print(f"  ✓ Fertig in {format_time(dt)}  —  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}")
                entry = {
                    'model':       model_name,
                    'frac':        frac,
                    'n_layer':     mcfg['n_layer'],
                    'n_head':      mcfg['n_head'],
                    'n_embd':      mcfg['n_embd'],
                    'param_count': params,
                    'total_tokens': total_tokens,
                    'max_iters':   MAX_ITERS,
                    'batch_size':  BATCH_SIZE,
                    'block_size':  BLOCK_SIZE,
                    'flops':       flops,
                    'train_loss':  train_loss,
                    'val_loss':    val_loss,
                    'elapsed_s':   dt,
                    'learning_curves': curves,   # (step, train_loss, val_loss)
                    'error':       False,
                }

            results[run_key] = entry

            # ── Nach jedem Lauf speichern (ermöglicht Fortsetzen) ───────────
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # ── Zusammenfassung ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    successful = sum(1 for v in results.values() if not v.get('error'))
    print(f"\n{'='*70}")
    print(f"GRID ABGESCHLOSSEN  —  {successful}/{total_runs} erfolgreich  "
          f"in {format_time(elapsed_total)}")
    print(f"Ergebnisse gespeichert: {RESULTS_FILE}")
    print("Weiter mit: python analyze_scaling.py")
    print("=" * 70)

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Scaling Law Grid Training')
    parser.add_argument('--device', default='cpu',
                        help='cpu | cuda  (default: cpu)')
    parser.add_argument('--total_tokens', type=int, default=10_000_000,
                        help='Iso-Compute-Budget: Tokens gesehen pro Lauf (default: 10M)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_grid(device=args.device, total_tokens=args.total_tokens)
