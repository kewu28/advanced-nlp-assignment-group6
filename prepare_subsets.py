"""
prepare_subsets.py  —  Schritt 1 der Scaling-Law-Pipeline
=========================================================
Erstellt 4 Trainings-Subsets (10%, 25%, 50%, 100%) aus dem
Tiny-Shakespeare-Datensatz. Das Validation-Set bleibt unverändert.

Ausführung (einmalig, aus dem nanoGPT-Verzeichnis):
    python prepare_subsets.py

Voraussetzung: data/shakespeare_char/train.bin existiert bereits.
Falls nicht: python data/shakespeare_char/prepare.py
"""

import numpy as np
import os
import sys

# ── Konfiguration ──────────────────────────────────────────────────────────────

DATA_DIR   = 'data/shakespeare_char'
FRACTIONS  = [0.10, 0.25, 0.50, 1.00]   # 10 % | 25 % | 50 % | 100 %


# ── Hilfsfunktion: Parameteranzahl schätzen (ohne Modell instantiieren) ────────

def estimate_params(n_layer: int, n_head: int, n_embd: int,
                    block_size: int = 256, vocab_size: int = 65) -> int:
    """
    Analytische Schätzung der Parameteranzahl für nanoGPT (bias=False).
    Entspricht der Formel aus transformer_sizing.ipynb, angepasst auf
    Tiny-Shakespeare (vocab_size=65, block_size=256).

    Gewichts-Tying zwischen wte und lm_head wird berücksichtigt:
    die lm_head-Parameter zählen nicht separat.
    """
    pos_emb   = n_embd * block_size          # wpe
    tok_emb   = n_embd * vocab_size          # wte  (= lm_head wegen tying)

    # Pro Transformer-Block
    attn_ln   = n_embd                       # LayerNorm scale (keine bias)
    attn_kqv  = n_embd * 3 * n_embd         # c_attn (Q, K, V kombiniert)
    attn_proj = n_embd * n_embd              # c_proj
    mlp_ln    = n_embd                       # LayerNorm scale
    mlp_fc    = n_embd * 4 * n_embd         # c_fc  (Expansion)
    mlp_proj  = 4 * n_embd * n_embd         # c_proj (Kontraktion)
    block_params = attn_ln + attn_kqv + attn_proj + mlp_ln + mlp_fc + mlp_proj

    ln_f      = n_embd                       # finales LayerNorm
    dense     = 0                            # lm_head: durch Weight-Tying = 0

    total = pos_emb + tok_emb + n_layer * block_params + ln_f + dense
    return total


MODEL_CONFIGS = {
    'XS': dict(n_layer=2, n_head=2, n_embd=64),    # ~0.12M
    'S':  dict(n_layer=4, n_head=4, n_embd=128),   # ~0.83M
    'M':  dict(n_layer=6, n_head=6, n_embd=384),   # ~10.7M  (Baseline aus Assignment)
    'L':  dict(n_layer=8, n_head=8, n_embd=512),   # ~25.3M
}


# ── Hauptprogramm ──────────────────────────────────────────────────────────────

def main():
    train_path = os.path.join(DATA_DIR, 'train.bin')
    val_path   = os.path.join(DATA_DIR, 'val.bin')

    # Checks
    if not os.path.exists(train_path):
        print(f"[ERROR] {train_path} nicht gefunden.")
        print("        Bitte erst ausführen:  python data/shakespeare_char/prepare.py")
        sys.exit(1)
    if not os.path.exists(val_path):
        print(f"[WARN]  {val_path} nicht gefunden – Validation fehlt!")

    data = np.memmap(train_path, dtype=np.uint16, mode='r')
    total = len(data)
    print(f"Vollständiges Trainingsset:  {total:,} Tokens")
    print(f"Zielverzeichnis:             {DATA_DIR}/\n")

    # ── Subsets erstellen ────────────────────────────────────────────────────
    print("Erstelle Dataset-Subsets:")
    print(f"  {'Fraction':>10}  {'Tokens':>12}  {'Dateiname'}")
    print("  " + "-"*45)
    for frac in FRACTIONS:
        n         = int(total * frac)
        subset    = np.array(data[:n], dtype=np.uint16)
        fname     = f'train_{int(frac * 100)}.bin'
        out_path  = os.path.join(DATA_DIR, fname)
        subset.tofile(out_path)
        print(f"  {int(frac*100):>9}%  {n:>12,}  {fname}")

    # ── Parameteranzahl pro Modell ────────────────────────────────────────────
    print("\nModell-Konfigurationen (mit Parameteranzahl):")
    print(f"  {'Label':>6}  {'n_layer':>8}  {'n_head':>7}  {'n_embd':>7}  {'Params':>12}")
    print("  " + "-"*50)
    for label, cfg in MODEL_CONFIGS.items():
        p = estimate_params(**cfg)
        print(f"  {label:>6}  {cfg['n_layer']:>8}  {cfg['n_head']:>7}  "
              f"{cfg['n_embd']:>7}  {p:>12,}")

    print("\n✓ Vorbereitung abgeschlossen. Weiter mit:  python run_scaling_grid.py")


if __name__ == '__main__':
    main()
