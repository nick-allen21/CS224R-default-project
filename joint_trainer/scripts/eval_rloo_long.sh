#!/bin/bash
# Eval the salvaged rloo_long checkpoints at max_tokens=2048 (fair eval budget).
# Salvage details: copied from running container ta-01KT599SSVT7VFMMCAX99FAY58 at step ~84.
#
# Set WANDB_API_KEY and HF_TOKEN in your shell before running.

set -euo pipefail

BUDGET_ARGS="--max_tokens=2048 --max_model_len=4096 --max_num_batched_tokens=8192"
BASE=/vol/checkpoints/rloo_checkpoints/rloo_default_project/rloo_long

echo "==> Eval rloo_long step 80"
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=${BASE}/epoch_0_step_80/model \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_rloo_long_step_80_2048 \
  $BUDGET_ARGS

echo "==> Eval rloo_long latest (~step 84+)"
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=${BASE}/latest_checkpoint/model \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_rloo_long_latest_2048 \
  $BUDGET_ARGS

echo
echo "==> Both eval jobs launched in --detach mode."
echo "    Each ~20 min on H100. Outputs land at /vol/evaluation/eval_results/eval_rloo_long_*_2048.json"
