import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log(path):
    steps, train, val = [], [], []
    with open(path) as f:
        for line in f:
            m = re.search(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                train.append(float(m.group(2)))
                val.append(float(m.group(3)))
    return steps, train, val

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {'task_a': '#4C72B0', 'task_b': '#55A868', 'multitask': '#C44E52'}
labels = {'task_a': 'Task A', 'task_b': 'Task B', 'multitask': 'Multitask'}

for ax, loss_type, idx in zip(axes, ['Train Loss', 'Val Loss'], [1, 2]):
    for task in ['task_a', 'task_b', 'multitask']:
        steps, train, val = parse_log(f'logs/{task}.log')
        data = train if idx == 1 else val
        if steps:
            ax.plot(steps, data, color=colors[task], label=labels[task], linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(loss_type, fontsize=12)
    ax.set_title(f'SFT {loss_type} vs. Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Part 2 - SFT Training Curves', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('sft_training_curves.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved: sft_training_curves.png')
