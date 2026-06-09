#!/bin/bash
# Eval v6, v9, and RLOO baseline at max_tokens=2048 for fair comparison.
# Plus SFT baseline at the same budget for the fourth comparison point.
#
# Set WANDB_API_KEY and HF_TOKEN in your shell before running.

set -euo pipefail

BUDGET_ARGS="--max_tokens=2048 --max_model_len=4096 --max_num_batched_tokens=8192"

echo "==> Eval 1/4: SFT baseline"
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=asingh15/qwen-sft-countdown-defaultproj \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_sft_baseline_2048 \
  $BUDGET_ARGS

echo "==> Eval 2/4: Joint v6 step 16 (the v6 peak)"
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v6/epoch_0_step_16/model \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_joint_v6_step_16_2048 \
  $BUDGET_ARGS

echo "==> Eval 3/4: Joint v9 latest checkpoint"
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v9/latest_checkpoint/model \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_joint_v9_latest_2048 \
  $BUDGET_ARGS

echo "==> Eval 4/4: RLOO baseline step 30 (Nick's reference)"
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_neb_lr1e-5_bs128_gs8_ent0.001_kl0.001_lrconstant_warmup0.0_temp1.0_topp1.0_topk-1/epoch_0_step_30/model \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_rloo_step_30_2048 \
  $BUDGET_ARGS

echo
echo "==> All 4 eval jobs launched in --detach mode."
echo "    Each takes ~15-20 min on H100 (at 2048 max_tokens it's a bit slower)."
echo "    Outputs land at /vol/evaluation/eval_results/eval_*_2048.json"
