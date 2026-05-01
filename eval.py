import os
import sys
import argparse
import pickle
import torch
import numpy as np

# args
parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True, choices=['a', 'b', 'forgetting'])
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--data', default=None)
parser.add_argument('--max_examples', type=int, default=None)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()

# load checkpoint
print(f"loading checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
model_args = ckpt['model_args']
print(f"model args: {model_args}")
print(f"iter: {ckpt.get('iter_num', '?')}  val loss: {ckpt.get('best_val_loss', '?')}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT

model = GPT(GPTConfig(**model_args))
model.load_state_dict(ckpt['model'])
model.eval()
model.to(args.device)
print("model loaded\n")

# load vocab from meta.pkl
def load_meta(task_dir):
    with open(os.path.join(task_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode

# greedy decode — just take argmax at each step
@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens, block_size):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_tok], dim=1)
    return idx

# task a — speaker identification
def evaluate_task_a(data_path, max_examples=None):
    encode, decode = load_meta(os.path.dirname(data_path))

    with open(data_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    if max_examples:
        lines = lines[:max_examples]

    correct = 0
    total = 0
    errors = []

    for example in lines:
        if '[ANSWER]' not in example or '[END]' not in example:
            continue

        # split into prompt and gold label
        prompt_part, rest = example.split('[ANSWER]', 1)
        gold = rest.replace('[END]', '').strip()
        prompt = prompt_part + '[ANSWER] '

        prompt_ids = encode(prompt)
        if not prompt_ids:
            continue

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        out = generate_greedy(model, idx, max_new_tokens=30,
                              block_size=model_args['block_size'])
        generated = decode(out[0].tolist()[len(prompt_ids):])

        # grab everything before [END]
        pred = generated.split('[END]')[0].strip() if '[END]' in generated else generated.strip()

        if pred.upper() == gold.upper():
            correct += 1
        else:
            errors.append((gold, pred))
        total += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"task a — speaker identification")
    print(f"examples: {total}  correct: {correct}  accuracy: {acc:.2f}%")
    print(f"random baseline: 10.00% (10 speakers)")
    if errors:
        print("first 5 errors (gold -> predicted):")
        for g, p in errors[:5]:
            print(f"  {g!r:20s} -> {p!r}")
    return acc

# task b — verse vs prose
def evaluate_task_b(data_path, max_examples=None):
    encode, decode = load_meta(os.path.dirname(data_path))

    with open(data_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    if max_examples:
        lines = lines[:max_examples]

    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    errors = []

    for example in lines:
        if '[ANSWER]' not in example or '[END]' not in example:
            continue

        prompt_part, rest = example.split('[ANSWER]', 1)
        gold = rest.replace('[END]', '').strip().upper()
        prompt = prompt_part + '[ANSWER] '

        if gold not in ['VERSE', 'PROSE']:
            continue

        prompt_ids = encode(prompt)
        if not prompt_ids:
            continue

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        out = generate_greedy(model, idx, max_new_tokens=10,
                              block_size=model_args['block_size'])
        generated = decode(out[0].tolist()[len(prompt_ids):]).upper()

        # check which label appears first
        if 'VERSE' in generated:
            pred = 'VERSE'
        elif 'PROSE' in generated:
            pred = 'PROSE'
        else:
            pred = generated[:5].strip()

        if pred == gold:
            correct += 1
        else:
            errors.append((gold, pred))

        # confusion matrix counts
        if gold == 'VERSE' and pred == 'VERSE': tp += 1
        if gold == 'PROSE' and pred == 'VERSE': fp += 1
        if gold == 'PROSE' and pred == 'PROSE': tn += 1
        if gold == 'VERSE' and pred == 'PROSE': fn += 1
        total += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"task b — verse vs prose")
    print(f"examples: {total}  correct: {correct}  accuracy: {acc:.2f}%")
    print(f"random baseline: 50.00% (binary)")
    print(f"confusion matrix: tp={tp} fp={fp} tn={tn} fn={fn}")
    if errors:
        print("first 5 errors (gold -> predicted):")
        for g, p in errors[:5]:
            print(f"  {g!r:8s} -> {p!r}")
    return acc

# catastrophic forgetting — val loss on original shakespeare data
def evaluate_forgetting():
    val_path = 'data/shakespeare_char/val.bin'
    if not os.path.exists(val_path):
        print(f"cant find {val_path} — need the original shakespeare val set")
        sys.exit(1)

    encode, decode = load_meta('data/shakespeare_char')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    print(f"val tokens: {len(val_data):,}")

    block_size = model_args['block_size']
    batch_size = 32
    eval_iters = 50  # average over 50 batches

    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            ix = torch.randint(len(val_data) - block_size, (batch_size,))
            x = torch.stack([
                torch.from_numpy(val_data[i:i+block_size].astype(np.int64)) for i in ix
            ]).to(args.device)
            y = torch.stack([
                torch.from_numpy(val_data[i+1:i+1+block_size].astype(np.int64)) for i in ix
            ]).to(args.device)
            _, loss = model(x, y)
            losses.append(loss.item())

    mean_loss = np.mean(losses)
    print(f"shakespeare val loss: {mean_loss:.4f}")
    print(f"(compare against pretrained baseline to measure forgetting)")
    return mean_loss

# run
if args.task == 'a':
    if not args.data:
        parser.error("--data required for task a")
    evaluate_task_a(args.data, args.max_examples)

elif args.task == 'b':
    if not args.data:
        parser.error("--data required for task b")
    evaluate_task_b(args.data, args.max_examples)

elif args.task == 'forgetting':
    evaluate_forgetting()
