#!/bin/bash
# Re-run vanilla RLOO with Nick's hyperparameters but longer
# (original run stopped at step ~30 due to time).
#
# Nick's config (extracted from his checkpoint path):
#   lr=1e-5, batch_size=128, group_size=8, ent=0.001, kl=0.001,
#   lr_schedule=constant, warmup=0.0, temp=1.0, top_p=1.0, top_k=-1
#
# We bump num_training_steps from ~30 to 100 to give the joint extension
# a long-form RLOO baseline to compare against.
#
# Set WANDB_API_KEY in your shell before running: export WANDB_API_KEY=...

modal run --detach modal_train.py rloo \
  --learning_rate=1e-5 \
  --batch_size=128 \
  --group_size=8 \
  --gradient_accumulation_steps=64 \
  --entropy_coefficient=0.001 \
  --kl_divergence_coefficient=0.001 \
  --lr_schedule=constant \
  --warmup_ratio=0.0 \
  --temperature=1.0 \
  --top_p=1.0 \
  --top_k=-1 \
  --max_tokens=1024 \
  --num_training_steps=100 \
  --save_every_n_steps=20 \
  --gpu_memory_utilization=0.3 \
  --wandb_name=rloo_long
