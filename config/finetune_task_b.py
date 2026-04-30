# config/finetune_task_b.py

# system
device = 'cuda'   # FIX: was 'cpu'
compile = False

# I/O
out_dir = 'out/task_b'
eval_interval = 50
eval_iters = 40
always_save_checkpoint = False

# logging
wandb_log = False  # set to True if you want W&B
wandb_project = 'shakespeare_sft'
wandb_run_name = 'task_b_m100_finetune'
log_interval = 10

# data
dataset = 'task_b'
block_size = 256

# model — must match M_100 architecture
n_layer = 6
n_head = 6
n_embd = 384
bias = False
vocab_size = 65

# init from M_100 checkpoint
init_from = 'resume'
dropout = 0.1

# training
max_iters = 2000
iter_num = 0
learning_rate = 5e-5
decay_lr = False
min_lr = 5e-5
batch_size = 32
gradient_accumulation_steps = 1

# optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
dtype = 'float16'
