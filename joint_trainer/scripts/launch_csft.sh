#!/bin/bash
# Mini-SFT the conjecturer on its own (problem-generation) dataset.
#
# Dataset: finn-staeblein/conjecturer_sft_mini (720 train / 80 test).
# Each row: query=conjecturer prompt, completion=<problem>numbers=[..],target=N</problem>
#
# Output: /vol/checkpoints/conjecturer_sft_mini/conjecturer_sft_mini/conj_sft_lr1e-5_ep2/epoch_1
# (sft.py auto-creates output_dir/wandb_project/wandb_name/epoch_{N})
#
# Set secrets first:  export WANDB_API_KEY=...  HF_TOKEN=...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export DATASET_NAME="finn-staeblein/conjecturer_sft_mini"
export OUTPUT_DIR="/vol/checkpoints/conjecturer_sft_mini"
export MODAL_TIMEOUT_SECONDS=3600

bash joint_trainer/train_conjecturer_sft.sh
