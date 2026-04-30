"""
prepare_task_b.py  [v3 — median split heuristic]
=================
Creates Task B (verse vs prose) dataset from the Shakespeare corpus.
Uses median line-length split to guarantee balanced classes.

Run from repo root:
  python prepare_task_b.py
"""

import os
import re
import pickle
import random
import numpy as np

SHAKESPEARE_PATH = os.path.join('data', 'shakespeare_char', 'input.txt')
DATA_DIR         = os.path.join('data', 'task_b')
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

with open(SHAKESPEARE_PATH, 'r', encoding='utf-8') as f:
    raw = f.read()

print(f"Loaded Shakespeare: {len(raw):,} characters")

# Split into speech blocks on blank lines
blocks = re.split(r'\n{2,}', raw.strip())
blocks = [b.strip() for b in blocks if b.strip()]
print(f"Raw blocks: {len(blocks)}")

def get_content_lines(block):
    lines = block.split('\n')
    content = []
    for l in lines:
        l = l.strip()
        if not l:
            continue
        # Skip speaker names: short ALL CAPS lines
        if l.upper() == l and len(l) < 35:
            continue
        content.append(l)
    return content

def avg_line_length(block):
    lines = get_content_lines(block)
    if not lines:
        return 0
    return np.mean([len(l) for l in lines])

def block_to_example(block, label):
    lines = get_content_lines(block)
    if not lines:
        return None
    n = min(random.randint(3, 5), len(lines))
    start = random.randint(0, max(0, len(lines) - n))
    passage = ' '.join(lines[start:start+n])
    passage = re.sub(r'\s+', ' ', passage).strip()
    return f"[CLASSIFY] {passage} [ANSWER] {label} [END]"

# Filter blocks with at least 3 content lines
valid_blocks = [(b, avg_line_length(b)) for b in blocks if len(get_content_lines(b)) >= 3]
print(f"Valid blocks (>=3 content lines): {len(valid_blocks)}")

# Sort by avg line length
valid_blocks.sort(key=lambda x: x[1])

# Print some stats to understand the distribution
avgs = [x[1] for x in valid_blocks]
print(f"Avg line length — min:{min(avgs):.1f}  median:{np.median(avgs):.1f}  max:{max(avgs):.1f}")

# Median split: bottom half = VERSE, top half = PROSE
mid = len(valid_blocks) // 2
verse_blocks = [b for b, _ in valid_blocks[:mid]]
prose_blocks  = [b for b, _ in valid_blocks[mid:]]

verse_examples = [block_to_example(b, 'VERSE') for b in verse_blocks]
prose_examples  = [block_to_example(b, 'PROSE')  for b in prose_blocks]

# Remove None entries
verse_examples = [e for e in verse_examples if e]
prose_examples  = [e for e in prose_examples  if e]

print(f"Verse examples : {len(verse_examples)}")
print(f"Prose examples : {len(prose_examples)}")

# Balance and cap at 600 per class
min_count = min(len(verse_examples), len(prose_examples), 600)
random.shuffle(verse_examples)
random.shuffle(prose_examples)
verse_examples = verse_examples[:min_count]
prose_examples  = prose_examples[:min_count]

all_examples = verse_examples + prose_examples
random.shuffle(all_examples)
print(f"Total balanced : {len(all_examples)} ({min_count} per class)")

split_idx      = int(len(all_examples) * 0.85)
train_examples = all_examples[:split_idx]
test_examples  = all_examples[split_idx:]
print(f"Train: {len(train_examples)}  Test: {len(test_examples)}")

with open(os.path.join(DATA_DIR, 'train.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_examples) + '\n')
with open(os.path.join(DATA_DIR, 'test.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_examples) + '\n')

train_text = '\n'.join(train_examples) + '\n'
val_text   = '\n'.join(test_examples)  + '\n'
all_text   = train_text + val_text
chars      = sorted(set(all_text))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]

train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids   = np.array(encode(val_text),   dtype=np.uint16)

train_ids.tofile(os.path.join(DATA_DIR, 'train.bin'))
val_ids.tofile(  os.path.join(DATA_DIR, 'val.bin'))

meta = {'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}
with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nSaved to {DATA_DIR}: train.bin, val.bin, meta.pkl")
print(f"\nSample VERSE (shorter lines):\n{verse_examples[0]}\n")
print(f"Sample PROSE (longer lines):\n{prose_examples[0]}\n")

v = sum(1 for e in train_examples if '[ANSWER] VERSE' in e)
p = sum(1 for e in train_examples if '[ANSWER] PROSE' in e)
print(f"Train balance: VERSE={v}, PROSE={p}")
