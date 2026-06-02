#!/bin/bash
# Set WANDB_API_KEY in your shell before running: export WANDB_API_KEY=...
modal run --detach modal_train.py joint \
  --num_problems_per_step=64 \
  --group_size_p=8 \
  --gradient_accumulation_steps=16 \
  --num_training_steps=100 \
  --save_every_n_steps=20 \
  --learning_rate=1e-5 \
  --c_learning_rate=1e-5 \
  --kl_divergence_coefficient=0.001 \
  --c_kl_divergence_coefficient=0.001 \
  --entropy_coefficient=0.001 \
  --difficulty_band_low=0.05 \
  --difficulty_band_high=0.75 \
  --max_tokens=512 \
  --gpu_memory_utilization=0.5 \
  --wandb_name=joint_v7
