"""
analyze_scaling.py  —  Schritt 3 der Scaling-Law-Pipeline
==========================================================
Lädt scaling_results.json und produziert alle Tabellen, Plots und
Diskussionsgrundlagen für Part 1 des Assignments.

Ausführung:
    python analyze_scaling.py

Outputs (automatisch gespeichert):
    scaling_1_val_vs_model_size.png   — Val-Loss vs. Parameteranzahl
    scaling_2_val_vs_data_size.png    — Val-Loss vs. Datenmenge
    scaling_3_val_vs_flops.png        — Val-Loss vs. FLOPs (farbig nach Modell)
    scaling_4_optimal_frontier.png    — Compute-Optimal Frontier
    scaling_loss_grid.txt             — Tabelle als Textdatei
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   # kein Display notwendig
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch

# ── Konstanten ─────────────────────────────────────────────────────────────────

RESULTS_FILE  = 'scaling_results.json'

MODEL_ORDER   = ['XS', 'S', 'M', 'L']
FRAC_ORDER    = [0.10, 0.25, 0.50, 1.00]
FRAC_LABELS   = ['10%', '25%', '50%', '100%']

# Approximative Token-Anzahl im vollen Shakespeare-Trainingsset
FULL_TRAIN_TOKENS = 1_003_854   # aus prepare.py: "train has 1003854 tokens"

COLORS = {
    'XS': '#4C72B0',   # Blau
    'S':  '#55A868',   # Grün
    'M':  '#C44E52',   # Rot
    'L':  '#8172B2',   # Lila
}
MARKERS = {'XS': 'o', 'S': 's', 'M': '^', 'L': 'D'}

# Matplotlib-Stil
plt.rcParams.update({
    'figure.dpi':          150,
    'font.size':           11,
    'font.family':         'sans-serif',
    'axes.grid':           True,
    'grid.alpha':          0.3,
    'grid.linestyle':      '--',
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'legend.framealpha':   0.9,
    'legend.fontsize':     9,
})

# ── Daten laden ────────────────────────────────────────────────────────────────

def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"[ERROR] {RESULTS_FILE} nicht gefunden.")
        print("        Bitte erst python run_scaling_grid.py ausführen.")
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        raw = json.load(f)

    # In Grid-Dict konvertieren: (model, frac) → Eintrag
    grid = {}
    for key, v in raw.items():
        if not v.get('error'):
            grid[(v['model'], v['frac'])] = v
    return raw, grid


# ── Tabelle drucken ────────────────────────────────────────────────────────────

def print_and_save_tables(grid):
    sep  = "=" * 75
    dash = "-" * 75

    def make_table(loss_key, title):
        lines = [sep, title, sep]
        header = f"  {'Modell':>6}  {'Params':>12}  |"
        for fl in FRAC_LABELS:
            header += f"  {fl:>10}"
        lines.append(header)
        lines.append(dash)

        for model in MODEL_ORDER:
            # Parameteranzahl aus erstem verfügbaren Eintrag
            params_str = '?'
            for frac in FRAC_ORDER:
                v = grid.get((model, frac))
                if v:
                    params_str = f"{v['param_count']:,}"
                    break
            row = f"  {model:>6}  {params_str:>12}  |"
            for frac in FRAC_ORDER:
                v = grid.get((model, frac))
                cell = f"{v[loss_key]:.4f}" if v else '  N/A  '
                row += f"  {cell:>10}"
            lines.append(row)

        lines.append(sep)
        return '\n'.join(lines)

    val_table   = make_table('val_loss',   'VALIDATION LOSS GRID')
    train_table = make_table('train_loss', 'TRAINING LOSS GRID')

    # Overfitting-Gap (Val - Train)
    lines_gap = [sep, 'OVERFITTING GAP  (Val Loss − Train Loss)', sep]
    header = f"  {'Modell':>6}  {'Params':>12}  |"
    for fl in FRAC_LABELS:
        header += f"  {fl:>10}"
    lines_gap.append(header)
    lines_gap.append(dash)
    for model in MODEL_ORDER:
        params_str = '?'
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v:
                params_str = f"{v['param_count']:,}"
                break
        row = f"  {model:>6}  {params_str:>12}  |"
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v:
                gap = v['val_loss'] - v['train_loss']
                row += f"  {gap:>+10.4f}"
            else:
                row += f"  {'N/A':>10}"
        lines_gap.append(row)
    lines_gap.append(sep)
    gap_table = '\n'.join(lines_gap)

    # Compute-Optimal Übersicht
    lines_opt = [sep, 'COMPUTE-OPTIMAL ANALYSE', sep]
    lines_opt.append(f"  {'Modell':>6}  {'Params':>12}  {'FLOPs':>14}  "
                     f"{'Opt.Frac':>10}  {'Opt.ValLoss':>12}  {'Tokens/Param':>13}")
    lines_opt.append(dash)
    for model in MODEL_ORDER:
        best_val, best_frac, best_v = float('inf'), None, None
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v and v['val_loss'] < best_val:
                best_val, best_frac, best_v = v['val_loss'], frac, v
        if best_v:
            tokens   = int(FULL_TRAIN_TOKENS * best_frac)
            ratio    = tokens / best_v['param_count']
            lines_opt.append(
                f"  {model:>6}  {best_v['param_count']:>12,}  "
                f"{best_v['flops']:>14.2e}  {int(best_frac*100):>9}%  "
                f"{best_val:>12.4f}  {ratio:>13.2f}×"
            )
    lines_opt.append(dash)
    lines_opt.append("  Chinchilla-Ratio (Referenz): ~20×")
    lines_opt.append(sep)
    opt_table = '\n'.join(lines_opt)

    full_text = '\n\n'.join([val_table, train_table, gap_table, opt_table])
    print(full_text)

    with open('scaling_loss_grid.txt', 'w') as f:
        f.write(full_text)
    print('\nTabellen gespeichert: scaling_loss_grid.txt')
    return full_text


# ── Plot 1: Val Loss vs. Parameteranzahl ───────────────────────────────────────

def plot_val_vs_model_size(grid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Links: Val-Loss; Rechts: Train-Loss (für Vergleich)
    for ax_idx, (loss_key, title_suffix) in enumerate(
        [('val_loss', 'Validation Loss'), ('train_loss', 'Training Loss')]
    ):
        ax = axes[ax_idx]

        for i, (frac, label) in enumerate(zip(FRAC_ORDER, FRAC_LABELS)):
            xs, ys = [], []
            for model in MODEL_ORDER:
                v = grid.get((model, frac))
                if v:
                    xs.append(v['param_count'])
                    ys.append(v[loss_key])
            if xs:
                color = cm.plasma(i / (len(FRAC_ORDER) - 1))
                ax.plot(xs, ys, 'o-', color=color, label=f"{label} Daten",
                        linewidth=2, markersize=8, markeredgecolor='white',
                        markeredgewidth=0.8)

        # Modell-Namen als vertikale Beschriftungen
        for model in MODEL_ORDER:
            for frac in FRAC_ORDER:
                v = grid.get((model, frac))
                if v:
                    ax.axvline(v['param_count'], color='gray', alpha=0.15, linewidth=0.8)
            ref_v = grid.get((model, FRAC_ORDER[-1])) or grid.get((model, FRAC_ORDER[0]))
            if ref_v:
                ax.text(ref_v['param_count'], ax.get_ylim()[0] if ax_idx else 0,
                        model, ha='center', va='bottom', fontsize=8.5,
                        color='gray', style='italic')

        ax.set_xscale('log')
        ax.set_xlabel('Parameter Count (N)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{title_suffix} vs. Modellgröße', fontsize=13, fontweight='bold')
        ax.legend(title='Datenmenge', loc='upper right' if ax_idx == 0 else 'upper left')

    plt.suptitle('Scaling mit Modellgröße (nanoGPT auf Tiny Shakespeare)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = 'scaling_1_val_vs_model_size.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')


# ── Plot 2: Val Loss vs. Datenmenge ───────────────────────────────────────────

def plot_val_vs_data_size(grid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (loss_key, title_suffix) in enumerate(
        [('val_loss', 'Validation Loss'), ('train_loss', 'Training Loss')]
    ):
        ax = axes[ax_idx]

        for model in MODEL_ORDER:
            xs_tok, ys_val, ys_trn = [], [], []
            for frac in FRAC_ORDER:
                v = grid.get((model, frac))
                if v:
                    xs_tok.append(int(FULL_TRAIN_TOKENS * frac))
                    ys_val.append(v['val_loss'])
                    ys_trn.append(v['train_loss'])

            if xs_tok:
                ys = ys_val if loss_key == 'val_loss' else ys_trn
                ax.plot(xs_tok, ys,
                        marker=MARKERS[model], color=COLORS[model],
                        label=model, linewidth=2, markersize=8,
                        markeredgecolor='white', markeredgewidth=0.8)

                # Datenpunkte beschriften (nur val-plot)
                if ax_idx == 0:
                    for x, y, fl in zip(xs_tok, ys, FRAC_LABELS):
                        ax.annotate(fl, (x, y),
                                    textcoords='offset points', xytext=(5, 3),
                                    fontsize=7, color=COLORS[model], alpha=0.7)

        ax.set_xscale('log')
        ax.set_xlabel('Trainings-Tokens (D)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{title_suffix} vs. Datenmenge', fontsize=13, fontweight='bold')
        ax.legend(title='Modellgröße', loc='upper right')

    plt.suptitle('Scaling mit Datenmenge  (iso-compute: gleiche Tokens gesehen)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = 'scaling_2_val_vs_data_size.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')


# ── Plot 3: Val Loss vs. FLOPs ─────────────────────────────────────────────────

def plot_val_vs_flops(grid):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Legende-Handles vorbereiten
    legend_handles = []

    for model in MODEL_ORDER:
        xs_flops, ys_val = [], []
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v:
                xs_flops.append(v['flops'])
                ys_val.append(v['val_loss'])

        if xs_flops:
            # Verbindungslinie (gleicher Modelltyp = gleiche FLOPs)
            ax.plot(xs_flops, ys_val, '-', color=COLORS[model], alpha=0.3, linewidth=1)

            for frac, xf, yv in zip(FRAC_ORDER, xs_flops, ys_val):
                ax.scatter(xf, yv,
                           color=COLORS[model], marker=MARKERS[model],
                           s=120, zorder=5, edgecolors='white', linewidths=0.8)
                # Datenmenge als Label
                ax.annotate(f'{int(frac*100)}%', (xf, yv),
                            textcoords='offset points', xytext=(6, 2),
                            fontsize=8, color=COLORS[model])

        legend_handles.append(
            Line2D([0], [0], color=COLORS[model], marker=MARKERS[model],
                   linewidth=1.5, markersize=8, label=model)
        )

    ax.set_xscale('log')
    ax.set_xlabel('Geschätzte FLOPs  (6 × N × D)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title(
        'Val Loss vs. Compute\n'
        '(Zahlen = Datenfraktion; gleiche Modellgröße = gleiche FLOPs)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(handles=legend_handles, title='Modell', loc='upper right')

    # Annotation: iso-compute-Linien (gleiche FLOPs = vertikale Linie pro Modell)
    ax.text(0.02, 0.02,
            'Innerhalb einer vertikalen Punktreihe:\ngleiche FLOPs, verschiedene Datenmenge',
            transform=ax.transAxes, fontsize=8, alpha=0.6,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    fname = 'scaling_3_val_vs_flops.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')


# ── Plot 4: Compute-Optimal Frontier ──────────────────────────────────────────

def plot_optimal_frontier(grid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Links: Frontier im FLOPs vs. Val-Loss Raum ─────────────────────────
    ax = axes[0]
    optimal_points = []   # (model, frac, val_loss, flops, params)

    for model in MODEL_CONFIGS_ORDERED := MODEL_ORDER:
        best_val, best_frac, best_v = float('inf'), None, None
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v and v['val_loss'] < best_val:
                best_val, best_frac, best_v = v['val_loss'], frac, v

        if best_v:
            optimal_points.append(
                (model, best_frac, best_val, best_v['flops'], best_v['param_count'])
            )

        # Alle Punkte: abgeblendet
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v:
                is_optimal = (frac == best_frac)
                ax.scatter(v['flops'], v['val_loss'],
                           color=COLORS[model], marker=MARKERS[model],
                           s=200 if is_optimal else 50,
                           alpha=1.0 if is_optimal else 0.2,
                           zorder=6 if is_optimal else 3,
                           edgecolors='black' if is_optimal else 'none',
                           linewidths=1.5)
                if is_optimal:
                    ax.annotate(f'{model}\n({int(frac*100)}%)',
                                (v['flops'], v['val_loss']),
                                textcoords='offset points', xytext=(8, 5),
                                fontsize=8.5, color=COLORS[model], fontweight='bold')

    # Frontier-Linie
    if optimal_points:
        opt_flops  = [p[3] for p in optimal_points]
        opt_losses = [p[2] for p in optimal_points]
        sorted_pts = sorted(zip(opt_flops, opt_losses))
        fx, vy = zip(*sorted_pts)
        ax.plot(fx, vy, 'k--', linewidth=1.5, alpha=0.6, label='Optimal Frontier', zorder=4)

    legend_handles = [
        Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
               linewidth=0, markersize=9, label=m)
        for m in MODEL_ORDER
    ] + [Line2D([0], [0], color='black', linestyle='--', label='Optimal Frontier')]

    ax.set_xscale('log')
    ax.set_xlabel('FLOPs (6 × N × D)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Compute-Optimal Frontier\n(Große Marker = optimale Datenmenge pro Modell)',
                 fontsize=12, fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9)

    # ── Rechts: Tokens-pro-Parameter bei optimaler Konfiguration ───────────
    ax2 = axes[1]
    if optimal_points:
        x_pos   = range(len(optimal_points))
        labels  = [p[0] for p in optimal_points]
        ratios  = [int(FULL_TRAIN_TOKENS * p[1]) / p[4] for p in optimal_points]
        colors  = [COLORS[p[0]] for p in optimal_points]

        bars = ax2.bar(x_pos, ratios, color=colors, alpha=0.85, edgecolor='black',
                       linewidth=0.8, width=0.6)
        for bar, ratio in zip(bars, ratios):
            ax2.text(bar.get_x() + bar.get_width() / 2, ratio + 0.2,
                     f'{ratio:.1f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Chinchilla-Referenzlinie
        ax2.axhline(20, color='crimson', linestyle='--', linewidth=2.0,
                    label='Chinchilla-Ratio (~20×)', zorder=5)
        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels(labels, fontsize=12)
        ax2.set_ylabel('Optimale Tokens / Parameter', fontsize=12)
        ax2.set_title('Beobachtetes Tokens-per-Parameter-Verhältnis\nbei optimaler Konfiguration',
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)

    plt.suptitle('Compute-Optimal Analyse  (nanoGPT auf Tiny Shakespeare)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = 'scaling_4_optimal_frontier.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')


# ── Lernkurven (Bonus) ─────────────────────────────────────────────────────────

def plot_learning_curves(raw_results):
    """Zeichnet Val-Loss-Lernkurven für alle Läufe (falls gespeichert)."""
    # Prüfen ob learning_curves vorhanden
    has_curves = any(
        v.get('learning_curves') and not v.get('error')
        for v in raw_results.values()
    )
    if not has_curves:
        print("(Keine Lernkurven in Ergebnissen – übersprungen)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax_idx, model in enumerate(MODEL_ORDER):
        ax = axes[ax_idx]
        for i, (frac, label) in enumerate(zip(FRAC_ORDER, FRAC_LABELS)):
            key = f"{model}_{int(frac*100)}"
            v   = raw_results.get(key)
            if not v or v.get('error') or not v.get('learning_curves'):
                continue
            curves = v['learning_curves']
            steps  = [c[0] for c in curves]
            vals   = [c[2] for c in curves]
            color  = cm.plasma(i / (len(FRAC_ORDER) - 1))
            ax.plot(steps, vals, '-', color=color, label=label, linewidth=1.8)

        ax.set_title(f'Modell {model}  ({v["param_count"]:,} Params)'
                     if (v := raw_results.get(f"{model}_100")) else f'Modell {model}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Val Loss', fontsize=10)
        ax.legend(title='Datenmenge', fontsize=8)

    plt.suptitle('Val-Loss-Lernkurven (alle Modelle & Datenfraktionen)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = 'scaling_5_learning_curves.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')


# ── Diskussionsfragen-Hilfstexte ───────────────────────────────────────────────

def print_discussion_hints(grid):
    print("\n" + "=" * 70)
    print("HINWEISE FÜR DIE DISKUSSIONSFRAGEN")
    print("=" * 70)

    # Q1: Compute-optimal trade-off
    print("\nQ1 — Compute-Optimal Trade-off & Chinchilla-Ratio:")
    for model in MODEL_ORDER:
        best_val, best_frac = float('inf'), None
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v and v['val_loss'] < best_val:
                best_val, best_frac = v['val_loss'], frac
        if best_frac:
            tokens = int(FULL_TRAIN_TOKENS * best_frac)
            v      = grid.get((model, best_frac))
            ratio  = tokens / v['param_count']
            print(f"  {model}: Optimal bei {int(best_frac*100)}% Daten  "
                  f"({tokens:,} Tokens / {v['param_count']:,} Params = {ratio:.1f}×)")

    # Q2: Overfitting-Analyse
    print("\nQ2 — Stärkste Overfitting-Fälle (Val − Train > 0.05):")
    for model in MODEL_ORDER:
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v:
                gap = v['val_loss'] - v['train_loss']
                if gap > 0.05:
                    print(f"  {model} @ {int(frac*100)}%: gap={gap:+.4f}  "
                          f"(train={v['train_loss']:.4f}, val={v['val_loss']:.4f})")

    print("\nQ3 — Skalierungslimitierungen bei Mini-Scale:")
    print("  • Zu bedenken: Shakespeare hat nur ~1M Zeichen (verglichen mit 10B+ in Chinchilla)")
    print("  • Charakter-Level-Modelle haben viel kleinere Vokabulare (65 vs. 50K BPE-Token)")
    print("  • Overfitting tritt bei uns viel früher auf")
    print("  • Die 6ND-FLOPs-Schätzung ist für sehr kleine N/D weniger akkurat")
    print("=" * 70)


# ── Hauptprogramm ──────────────────────────────────────────────────────────────

def main():
    print("SCALING LAW ANALYSE — nanoGPT auf Tiny Shakespeare")
    print("=" * 70)

    raw_results, grid = load_results()

    n_successful = sum(1 for v in raw_results.values() if not v.get('error'))
    print(f"Geladene Ergebnisse: {n_successful} / {len(raw_results)} erfolgreich\n")

    if n_successful == 0:
        print("[ERROR] Keine gültigen Ergebnisse vorhanden.")
        sys.exit(1)

    # Tabellen
    print_and_save_tables(grid)

    # Plots
    print("\nErstelle Plots...")
    plot_val_vs_model_size(grid)
    plot_val_vs_data_size(grid)
    plot_val_vs_flops(grid)
    plot_optimal_frontier(grid)
    plot_learning_curves(raw_results)

    # Diskussionshinweise
    print_discussion_hints(grid)

    print("\n✓ Analyse abgeschlossen! Alle Dateien im aktuellen Verzeichnis.")


if __name__ == '__main__':
    main()
