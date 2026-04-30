"""
prepare_task_b.py
=================
1. Parses the raw Shakespeare text to extract verse vs. prose passages.
2. Saves train.txt and test.txt to data/task_b/.
3. Encodes them to train.bin, val.bin, meta.pkl for nanoGPT fine-tuning.

Format:
  [CLASSIFY] <3-5 lines of text> [ANSWER] VERSE [END]
  [CLASSIFY] <3-5 lines of text> [ANSWER] PROSE [END]

Heuristic for verse vs prose:
  - Verse: most lines are 40-80 chars (roughly iambic pentameter),
            consistent line lengths (low std dev)
  - Prose: lines vary wildly in length or are very long (>90 chars)

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

TRAIN_SPLIT = 0.85
MIN_EXAMPLES_PER_CLASS = 300
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Load Shakespeare ───────────────────────────────────────────────────────────

with open(SHAKESPEARE_PATH, 'r', encoding='utf-8') as f:
    raw = f.read()

print(f"Loaded Shakespeare: {len(raw):,} characters")

# ── Split into paragraphs / passage blocks ─────────────────────────────────────

# Split on blank lines to get natural passage blocks
blocks = re.split(r'\n{2,}', raw.strip())
blocks = [b.strip() for b in blocks if b.strip()]
print(f"Raw blocks: {len(blocks)}")

# ── Classify each block ────────────────────────────────────────────────────────

def classify_block(block):
    """
    Returns 'VERSE', 'PROSE', or None (if ambiguous / too short).
    Heuristic:
      - Split into lines, filter blank lines and stage directions (ALL CAPS short lines)
      - Verse: median line length 40-75, std dev < 20
      - Prose: median line length > 75 OR std dev > 35
    """
    lines = [l for l in block.split('\n') if l.strip()]
    # Filter out stage directions and speaker names (short ALL CAPS lines)
    lines = [l for l in lines if not (l.isupper() and len(l) < 40)]
    # Need at least 3 lines for a meaningful passage
    if len(lines) < 3:
        return None

    lengths = [len(l.strip()) for l in lines]
    median_len = sorted(lengths)[len(lengths) // 2]
    std_len    = np.std(lengths)

    # Very short lines (< 20 chars) usually fragments — skip
    if median_len < 20:
        return None

    if median_len <= 78 and std_len < 22:
        return 'VERSE'
    elif median_len > 78 or std_len > 32:
        return 'PROSE'
    else:
        return None  # ambiguous middle ground — skip


def block_to_example(block, label):
    """Format a block as a task B example, using 3-5 lines."""
    lines = [l.strip() for l in block.split('\n') if l.strip()]
    lines = [l for l in lines if not (l.isupper() and len(l) < 40)]
    # Take 3-5 lines from a random position in the block
    n = min(random.randint(3, 5), len(lines))
    start = random.randint(0, max(0, len(lines) - n))
    passage = '\n'.join(lines[start:start+n])
    return f"[CLASSIFY] {passage} [ANSWER] {label} [END]"


# ── Build dataset ──────────────────────────────────────────────────────────────

verse_examples = []
prose_examples = []

for block in blocks:
    label = classify_block(block)
    if label == 'VERSE':
        verse_examples.append(block_to_example(block, 'VERSE'))
    elif label == 'PROSE':
        prose_examples.append(block_to_example(block, 'PROSE'))

print(f"Verse examples : {len(verse_examples)}")
print(f"Prose examples : {len(prose_examples)}")

# Balance classes
min_count = min(len(verse_examples), len(prose_examples), 600)
random.shuffle(verse_examples)
random.shuffle(prose_examples)
verse_examples = verse_examples[:min_count]
prose_examples = prose_examples[:min_count]

all_examples = verse_examples + prose_examples
random.shuffle(all_examples)

print(f"Total balanced : {len(all_examples)} ({min_count} per class)")

if min_count < MIN_EXAMPLES_PER_CLASS:
    print(f"WARNING: Only {min_count} examples per class — less than recommended {MIN_EXAMPLES_PER_CLASS}")

# ── Train / test split ─────────────────────────────────────────────────────────

split_idx    = int(len(all_examples) * TRAIN_SPLIT)
train_examples = all_examples[:split_idx]
test_examples  = all_examples[split_idx:]

print(f"Train examples : {len(train_examples)}")
print(f"Test  examples : {len(test_examples)}")

# Save txt files
with open(os.path.join(DATA_DIR, 'train.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_examples) + '\n')

with open(os.path.join(DATA_DIR, 'test.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_examples) + '\n')

print(f"Saved train.txt and test.txt to {DATA_DIR}")

# ── Encode to bin ──────────────────────────────────────────────────────────────

train_text = '\n'.join(train_examples) + '\n'
val_text   = '\n'.join(test_examples)  + '\n'

all_text   = train_text + val_text
chars      = sorted(set(all_text))
vocab_size = len(chars)

print(f"Vocab size     : {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s if c in stoi]

train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids   = np.array(encode(val_text),   dtype=np.uint16)

print(f"Train tokens   : {len(train_ids):,}")
print(f"Val tokens     : {len(val_ids):,}")

train_ids.tofile(os.path.join(DATA_DIR, 'train.bin'))
val_ids.tofile(  os.path.join(DATA_DIR, 'val.bin'))

meta = {'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}
with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nSaved to {DATA_DIR}:")
print(f"  train.bin, val.bin, meta.pkl")

# ── Quick sanity check ─────────────────────────────────────────────────────────

print(f"\nSample VERSE example:\n{verse_examples[0]}\n")
print(f"Sample PROSE example:\n{prose_examples[0]}\n")

verse_count = sum(1 for e in train_examples if 'VERSE' in e.split('[ANSWER]')[1])
prose_count = sum(1 for e in train_examples if 'PROSE' in e.split('[ANSWER]')[1])
print(f"Train class balance: VERSE={verse_count}, PROSE={prose_count}")
