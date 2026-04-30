"""
prepare_task_a.py
=================
Converts data/task_a/train.txt and test.txt into train.bin, val.bin, meta.pkl
for character-level nanoGPT fine-tuning on Task A (speaker identification).

Format expected:
  [SPEAKER] <dialogue> [ANSWER] CHARACTER [END]

Run from repo root:
  python prepare_task_a.py
"""

import os
import pickle
import numpy as np

DATA_DIR = os.path.join('data', 'task_a')
TRAIN_TXT = os.path.join(DATA_DIR, 'train.txt')
TEST_TXT  = os.path.join(DATA_DIR, 'test.txt')

# ── Load raw text ──────────────────────────────────────────────────────────────

with open(TRAIN_TXT, 'r', encoding='utf-8') as f:
    train_lines = [l.strip() for l in f if l.strip()]

with open(TEST_TXT, 'r', encoding='utf-8') as f:
    test_lines = [l.strip() for l in f if l.strip()]

print(f"Train examples : {len(train_lines)}")
print(f"Test  examples : {len(test_lines)}")

# Join into single strings (one example per line, newline separated)
train_text = '\n'.join(train_lines) + '\n'
val_text   = '\n'.join(test_lines)  + '\n'

# ── Build character vocabulary ─────────────────────────────────────────────────

all_text = train_text + val_text
chars    = sorted(set(all_text))
vocab_size = len(chars)

print(f"Vocab size     : {vocab_size}")
print(f"Characters     : {''.join(chars)}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Encode and save ────────────────────────────────────────────────────────────

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
print(f"  train.bin ({len(train_ids):,} tokens)")
print(f"  val.bin   ({len(val_ids):,} tokens)")
print(f"  meta.pkl  (vocab_size={vocab_size})")

# ── Quick sanity check ─────────────────────────────────────────────────────────

sample = decode(train_ids[:200].tolist())
print(f"\nFirst 200 chars decoded:\n{sample}")
