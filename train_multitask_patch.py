# train_multitask_patch.py
# Replace get_batch() in train.py with this version.
# Everything else in train.py stays the same.
#
# After patching, run with:
#   cp out/scaling/M_100/ckpt_small.pt out/multitask/ckpt.pt
#   python train.py config/finetune_multitask.py

def get_batch(split):
    # multitask — interleave task a and task b
    if dataset == 'multitask':
        data_a = np.memmap(os.path.join('data', 'task_a', f'{split}.bin'), dtype=np.uint16, mode='r')
        data_b = np.memmap(os.path.join('data', 'task_b', f'{split}.bin'), dtype=np.uint16, mode='r')

        xs, ys = [], []
        for _ in range(batch_size):
            data = data_a if np.random.random() < 0.5 else data_b
            ix = np.random.randint(0, len(data) - block_size)
            x = torch.from_numpy(data[ix : ix + block_size].astype(np.int64))
            y = torch.from_numpy(data[ix+1 : ix + block_size + 1].astype(np.int64))
            xs.append(x)
            ys.append(y)

        idx = torch.randperm(batch_size)
        x = torch.stack(xs)[idx]
        y = torch.stack(ys)[idx]

        # FIX: use pin_memory for CUDA just like the single-task branch
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    # original single-task loader — unchanged
    else:
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
