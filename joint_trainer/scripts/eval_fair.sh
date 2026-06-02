#!/bin/bash
# Fair re-eval: SFT, v6 step 16, v7 latest all with max_tokens=2048 (was 1024).
# Eval critic found 269/271 v7 failures hit the 1024-token cap mid-reasoning.
# Must re-eval all three to enable apples-to-apples comparison.

# Set WANDB_API_KEY in your shell before running: export WANDB_API_KEY=...

BUDGET_ARGS="--max_tokens=2048 --max_model_len=4096 --max_num_batched_tokens=8192"

# Eval 1: SFT baseline
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=asingh15/qwen-sft-countdown-defaultproj \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_sft_baseline_2048 \
  $BUDGET_ARGS

# Eval 2: Joint v6 step 16 (the v6 peak)
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v6/epoch_0_step_16/model \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_joint_v6_step_16_2048 \
  $BUDGET_ARGS

# Eval 3: Joint v7 latest
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v7/latest_checkpoint/model \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_joint_v7_latest_2048 \
  $BUDGET_ARGS
