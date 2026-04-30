# config/finetune_multitask.py — config for experiment 2, joint task a + task b training

# system
device = 'cuda'   # FIX: was 'cpu'
compile = False

# I/O
out_dir = 'out/multitask'
eval_interval = 100
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

# init from M_100 checkpoint
# cp out/scaling/M_100/ckpt_small.pt out/multitask/ckpt.pt
init_from = 'resume'

# signals the patched get_batch() to use multitask loader
dataset = 'multitask'

# must match M_100 architecture exactly
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
bias = False
vocab_size = 65

# same hyperparams as single-task runs for fair comparison
batch_size = 32
max_iters = 2000
learning_rate = 5e-5
decay_lr = False
warmup_iters = 0
min_lr = 5e-5

dropout = 0.1
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
dtype = 'float16'

iter_num = 0
best_val_loss = 1e9
