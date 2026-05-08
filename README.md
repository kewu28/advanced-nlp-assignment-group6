---
All experiments were run on Lightning AI with a T4 GPU.

---

## Running the Notebook

Open `advanced_nlp_group6.ipynb` and run cells top to bottom. The notebook is self-contained and covers:

1. Data preparation (dataset subsets, Task A/B encoding)
2. Model training (scaling grid, SFT, LoRA)
3. Evaluation and results tables
4. Plots and discussion

Make sure `scaling_results.json`, `lora_results.json`, and the pretrained checkpoint at `out/scaling/M_100/ckpt_small.pt` are present before running Parts 2 and 3.

---

## Key Results

| Model | Dataset | Val Loss |
|-------|---------|----------|
| XS | 100% | 2.3675 |
| S | 100% | **1.8630** |
| M | 100% | 2.0393 |
| L | 100% | 3.2590 |

| Setup | Task A Acc. | Task B Acc. | Shakespeare Val Loss |
|-------|-------------|-------------|---------------------|
| Pretrained | — | — | 2.04 |
| Single-task A | 19.0% | — | 5.75 |
| Single-task B | — | 68.89% | 6.25 |
| Multi-task | 16.0% | 63.33% | 8.44 |

| Method | Trainable Params | Val Loss | Forgetting |
|--------|-----------------|----------|------------|
| Full Fine-Tuning | 10,745,088 | 2.2053 | +0.165 |
| LoRA rank=1 | 9,216 | 2.0480 | +0.008 |
| LoRA rank=4 | 36,864 | **2.0454** | +0.005 |
| LoRA rank=16 | 147,456 | 2.0543 | +0.014 |
