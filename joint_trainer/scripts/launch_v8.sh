#!/bin/bash
# v8 — same hyperparams as v7 plus four fixes for the diversity-collapse issue:
#   1. Shaped C reward (tent function peaking at p_hat=0.4) — replaces binary band
#   2. 50/50 mix: half of P's training problems come from the fixed Countdown set
#   3. P length penalty (alpha=0.1) — discourages rambling without hard truncation
#   4. max_tokens=1024 in training (was 512) — wider budget paired with the penalty
#
# Set WANDB_API_KEY in your shell before running: export WANDB_API_KEY=...

modal run --detach modal_train.py joint \
  --num_problems_per_step=16 \
  --group_size_p=8 \
  --gradient_accumulation_steps=8 \
  --num_training_steps=100 \
  --save_every_n_steps=20 \
  --learning_rate=1e-5 \
  --c_learning_rate=5e-7 \
  --kl_divergence_coefficient=0.001 \
  --c_kl_divergence_coefficient=0.02 \
  --entropy_coefficient=0.001 \
  --max_tokens=1024 \
  --gpu_memory_utilization=0.5 \
  --c_reward_peak=0.4 \
  --p_mix_fraction=0.5 \
  --p_length_penalty_alpha=0.1 \
  --wandb_name=joint_v8
