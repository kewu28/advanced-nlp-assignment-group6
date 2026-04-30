"""
analyze_scaling.py  —  Part 1: Scaling Law Analysis
=====================================================
Loads scaling_results.json and produces all tables, plots, and
discussion hints for Part 1 of the Advanced NLP Assignment.

Usage:
    python analyze_scaling.py

Outputs:
    scaling_1_val_vs_model_size.png
    scaling_2_val_vs_data_size.png
    scaling_3_val_vs_flops.png
    scaling_4_optimal_frontier.png
    scaling_5_learning_curves.png
    scaling_loss_grid.txt
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_FILE       = 'scaling_results.json'
MODEL_ORDER        = ['XS', 'S', 'M', 'L']
FRAC_ORDER         = [0.10, 0.25, 0.50, 1.00]
FRAC_LABELS        = ['10%', '25%', '50%', '100%']
FULL_TRAIN_TOKENS  = 1_003_854  # from prepare.py

COLORS  = {'XS': '#4C72B0', 'S': '#55A868', 'M': '#C44E52', 'L': '#8172B2'}
MARKERS = {'XS': 'o',       'S': 's',       'M': '^',       'L': 'D'}

plt.rcParams.update({
    'figure.dpi':        150,
    'font.size':         11,
    'font.family':       'sans-serif',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'legend.framealpha': 0.9,
    'legend.fontsize':   9,
})

# ── Load data ─────────────────────────────────────────────────────────────────

def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"[ERROR] {RESULTS_FILE} not found.")
        print("        Please run: python run_scaling_grid_colab.py first.")
        sys.exit(1)
    with open(RESULTS_FILE) as f:
        raw = json.load(f)
    grid = {}
    for key, v in raw.items():
        if not v.get('error'):
            grid[(v['model'], v['frac'])] = v
    return raw, grid

# ── Tables ────────────────────────────────────────────────────────────────────

def print_and_save_tables(grid):
    sep  = "=" * 75
    dash = "-" * 75

    def make_table(loss_key, title):
        lines = [sep, title, sep]
        header = f"  {'Model':>6}  {'Params':>12}  |"
        for fl in FRAC_LABELS:
            header += f"  {fl:>10}"
        lines.append(header)
        lines.append(dash)
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
                cell = f"{v[loss_key]:.4f}" if v else '  N/A  '
                row += f"  {cell:>10}"
            lines.append(row)
        lines.append(sep)
        return '\n'.join(lines)

    val_table   = make_table('val_loss',   'VALIDATION LOSS GRID')
    train_table = make_table('train_loss', 'TRAINING LOSS GRID')

    # Overfitting gap
    lines_gap = [sep, 'OVERFITTING GAP  (Val Loss - Train Loss)', sep]
    header = f"  {'Model':>6}  {'Params':>12}  |"
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

    # Compute-optimal summary
    lines_opt = [sep, 'COMPUTE-OPTIMAL ANALYSIS', sep]
    lines_opt.append(f"  {'Model':>6}  {'Params':>12}  {'FLOPs':>14}  "
                     f"{'Opt.Frac':>10}  {'Opt.ValLoss':>12}  {'Tokens/Param':>13}")
    lines_opt.append(dash)
    for model in MODEL_ORDER:
        best_val, best_frac, best_v = float('inf'), None, None
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v and v['val_loss'] < best_val:
                best_val, best_frac, best_v = v['val_loss'], frac, v
        if best_v:
            tokens = int(FULL_TRAIN_TOKENS * best_frac)
            ratio  = tokens / best_v['param_count']
            lines_opt.append(
                f"  {model:>6}  {best_v['param_count']:>12,}  "
                f"{best_v['flops']:>14.2e}  {int(best_frac*100):>9}%  "
                f"{best_val:>12.4f}  {ratio:>13.2f}x"
            )
    lines_opt.append(dash)
    lines_opt.append("  Chinchilla Reference Ratio: ~20x")
    lines_opt.append(sep)
    opt_table = '\n'.join(lines_opt)

    full_text = '\n\n'.join([val_table, train_table, gap_table, opt_table])
    print(full_text)
    with open('scaling_loss_grid.txt', 'w') as f:
        f.write(full_text)
    print('\nTables saved: scaling_loss_grid.txt')
    return full_text

# ── Plot 1: Val Loss vs. Model Size ───────────────────────────────────────────

def plot_val_vs_model_size(grid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
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
                ax.plot(xs, ys, 'o-', color=color, label=f"{label} data",
                        linewidth=2, markersize=8, markeredgecolor='white',
                        markeredgewidth=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('Parameter Count (N)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{title_suffix} vs. Model Size', fontsize=13, fontweight='bold')
        ax.legend(title='Dataset size', loc='upper right')

        # Model name annotations
        for model in MODEL_ORDER:
            ref_v = grid.get((model, FRAC_ORDER[-1])) or grid.get((model, FRAC_ORDER[0]))
            if ref_v:
                ax.axvline(ref_v['param_count'], color='gray', alpha=0.15, linewidth=0.8)
                ymin = ax.get_ylim()[0]
                ax.text(ref_v['param_count'], ymin,
                        model, ha='center', va='bottom', fontsize=8.5,
                        color='gray', style='italic')

    plt.suptitle('Scaling with Model Size (nanoGPT on Tiny Shakespeare)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = 'scaling_1_val_vs_model_size.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')

# ── Plot 2: Val Loss vs. Dataset Size ─────────────────────────────────────────

def plot_val_vs_data_size(grid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (loss_key, title_suffix) in enumerate(
        [('val_loss', 'Validation Loss'), ('train_loss', 'Training Loss')]
    ):
        ax = axes[ax_idx]
        for model in MODEL_ORDER:
            xs_tok, ys = [], []
            for frac in FRAC_ORDER:
                v = grid.get((model, frac))
                if v:
                    xs_tok.append(int(FULL_TRAIN_TOKENS * frac))
                    ys.append(v[loss_key])
            if xs_tok:
                ax.plot(xs_tok, ys,
                        marker=MARKERS[model], color=COLORS[model],
                        label=model, linewidth=2, markersize=8,
                        markeredgecolor='white', markeredgewidth=0.8)
                if ax_idx == 0:
                    for x, y, fl in zip(xs_tok, ys, FRAC_LABELS):
                        ax.annotate(fl, (x, y),
                                    textcoords='offset points', xytext=(5, 3),
                                    fontsize=7, color=COLORS[model], alpha=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('Training Tokens (D)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{title_suffix} vs. Dataset Size', fontsize=13, fontweight='bold')
        ax.legend(title='Model size', loc='upper right')

    plt.suptitle('Scaling with Dataset Size (iso-compute: same tokens seen per run)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = 'scaling_2_val_vs_data_size.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')

# ── Plot 3: Val Loss vs. FLOPs ─────────────────────────────────────────────────

def plot_val_vs_flops(grid):
    fig, ax = plt.subplots(figsize=(9, 6))
    legend_handles = []
    for model in MODEL_ORDER:
        xs_flops, ys_val = [], []
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v:
                xs_flops.append(v['flops'])
                ys_val.append(v['val_loss'])
        if xs_flops:
            ax.plot(xs_flops, ys_val, '-', color=COLORS[model], alpha=0.3, linewidth=1)
            for frac, xf, yv in zip(FRAC_ORDER, xs_flops, ys_val):
                ax.scatter(xf, yv, color=COLORS[model], marker=MARKERS[model],
                           s=120, zorder=5, edgecolors='white', linewidths=0.8)
                ax.annotate(f'{int(frac*100)}%', (xf, yv),
                            textcoords='offset points', xytext=(6, 2),
                            fontsize=8, color=COLORS[model])
        legend_handles.append(
            Line2D([0], [0], color=COLORS[model], marker=MARKERS[model],
                   linewidth=1.5, markersize=8, label=model)
        )
    ax.set_xscale('log')
    ax.set_xlabel('Estimated FLOPs  (6 × N × D)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss vs. Compute\n'
                 '(labels = data fraction; same model = same FLOPs)',
                 fontsize=13, fontweight='bold')
    ax.legend(handles=legend_handles, title='Model', loc='upper right')
    plt.tight_layout()
    fname = 'scaling_3_val_vs_flops.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')

# ── Plot 4: Compute-Optimal Frontier ──────────────────────────────────────────

def plot_optimal_frontier(grid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    optimal_points = []
    for model in MODEL_ORDER:
        best_val, best_frac, best_v = float('inf'), None, None
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v and v['val_loss'] < best_val:
                best_val, best_frac, best_v = v['val_loss'], frac, v
        if best_v:
            optimal_points.append((model, best_frac, best_val, best_v['flops'], best_v['param_count']))
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

    if optimal_points:
        opt_flops  = [p[3] for p in optimal_points]
        opt_losses = [p[2] for p in optimal_points]
        sorted_pts = sorted(zip(opt_flops, opt_losses))
        fx, vy = zip(*sorted_pts)
        ax.plot(fx, vy, 'k--', linewidth=1.5, alpha=0.6, label='Optimal Frontier')

    legend_handles = [
        Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
               linewidth=0, markersize=9, label=m)
        for m in MODEL_ORDER
    ] + [Line2D([0], [0], color='black', linestyle='--', label='Optimal Frontier')]

    ax.set_xscale('log')
    ax.set_xlabel('FLOPs (6 × N × D)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Compute-Optimal Frontier\n(large markers = optimal data fraction per model)',
                 fontsize=12, fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9)

    # Right: Tokens per parameter
    ax2 = axes[1]
    if optimal_points:
        x_pos  = range(len(optimal_points))
        labels = [p[0] for p in optimal_points]
        ratios = [int(FULL_TRAIN_TOKENS * p[1]) / p[4] for p in optimal_points]
        colors = [COLORS[p[0]] for p in optimal_points]
        bars = ax2.bar(x_pos, ratios, color=colors, alpha=0.85,
                       edgecolor='black', linewidth=0.8, width=0.6)
        for bar, ratio in zip(bars, ratios):
            ax2.text(bar.get_x() + bar.get_width() / 2, ratio + 0.05,
                     f'{ratio:.2f}x', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
        ax2.axhline(20, color='crimson', linestyle='--', linewidth=2.0,
                    label='Chinchilla Ratio (~20x)')
        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels(labels, fontsize=12)
        ax2.set_ylabel('Optimal Tokens / Parameter', fontsize=12)
        ax2.set_title('Observed Tokens-per-Parameter Ratio\nat optimal configuration',
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)

    plt.suptitle('Compute-Optimal Analysis (nanoGPT on Tiny Shakespeare)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = 'scaling_4_optimal_frontier.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')

# ── Plot 5: Learning Curves ────────────────────────────────────────────────────

def plot_learning_curves(raw_results):
    has_curves = any(
        v.get('learning_curves') and not v.get('error')
        for v in raw_results.values()
    )
    if not has_curves:
        print("(No learning curves in results — skipped)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
        ref = raw_results.get(f"{model}_100")
        title = (f'Model {model}  ({ref["param_count"]:,} params)'
                 if ref else f'Model {model}')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Validation Loss', fontsize=10)
        ax.legend(title='Dataset size', fontsize=8)

    plt.suptitle('Validation Loss Learning Curves (all models & dataset fractions)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = 'scaling_5_learning_curves.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')

# ── Discussion hints ───────────────────────────────────────────────────────────

def print_discussion_hints(grid):
    print("\n" + "=" * 70)
    print("DISCUSSION QUESTION HINTS")
    print("=" * 70)

    print("\nQ1 — Compute-Optimal Trade-off & Chinchilla Ratio:")
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
            print(f"  {model}: optimal at {int(best_frac*100)}% data  "
                  f"({tokens:,} tokens / {v['param_count']:,} params = {ratio:.2f}x)")

    print("\nQ2 — Strongest overfitting cases (Val − Train > 0.05):")
    for model in MODEL_ORDER:
        for frac in FRAC_ORDER:
            v = grid.get((model, frac))
            if v:
                gap = v['val_loss'] - v['train_loss']
                if gap > 0.05:
                    print(f"  {model} @ {int(frac*100)}%: gap={gap:+.4f}  "
                          f"(train={v['train_loss']:.4f}, val={v['val_loss']:.4f})")

    print("\nQ3 — Key limitations of mini-scale experiments:")
    print("  • Shakespeare has only ~1M characters (vs. 10B+ tokens in Chinchilla)")
    print("  • Character-level vocab is tiny (65 chars vs. 50K BPE tokens)")
    print("  • Overfitting occurs much earlier — M and L are severely data-starved")
    print("  • The 6ND FLOP estimate is less accurate at very small N and D")
    print("  • Optimal ratios far below Chinchilla's 20x — dataset is the bottleneck")
    print("=" * 70)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("SCALING LAW ANALYSIS — nanoGPT on Tiny Shakespeare")
    print("=" * 70)

    raw_results, grid = load_results()
    n_successful = sum(1 for v in raw_results.values() if not v.get('error'))
    print(f"Loaded results: {n_successful} / {len(raw_results)} successful\n")

    if n_successful == 0:
        print("[ERROR] No valid results found.")
        sys.exit(1)

    print_and_save_tables(grid)

    print("\nGenerating plots...")
    plot_val_vs_model_size(grid)
    plot_val_vs_data_size(grid)
    plot_val_vs_flops(grid)
    plot_optimal_frontier(grid)
    plot_learning_curves(raw_results)

    print_discussion_hints(grid)
    print("\n✓ Analysis complete! All files saved in current directory.")

if __name__ == '__main__':
    main()