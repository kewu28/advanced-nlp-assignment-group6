# config/finetune_task_a.py

# system
device = 'cpu'
compile = False

# I/O
out_dir = 'out/task_a'
eval_interval = 50
eval_iters = 40
always_save_checkpoint = False

# logging
wandb_log = True
wandb_project = 'shakespeare_sft'
wandb_run_name = 'task_a_m100_finetune'
log_interval = 10

# data
dataset = 'task_a'
block_size = 256

# model
init_from = 'resume'
dropout = 0.1

# training
max_iters = 2000
iter_num = 0
learning_rate = 5e-5
decay_lr = False
batch_size = 32
gradient_accumulation_steps = 1

# optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
