#!/bin/bash
# Full-scale SFT for the conjecturer: 100k examples (vs mini's 720).
#
# Set WANDB_API_KEY and HF_TOKEN in your shell before running.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export DATASET_NAME="finn-staeblein/conjecturer_sft_full"
export OUTPUT_DIR="/vol/checkpoints/conjecturer_sft_full"
export WANDB_PROJECT="conjecturer_sft_full"
export WANDB_NAME_PREFIX="conj_sft_full"
export BATCH_SIZE=32
export EPOCHS=1
export LRS=1e-5
export MAX_RESPONSE_LENGTH=128
export MODAL_TIMEOUT_SECONDS=14400

bash joint_trainer/train_conjecturer_sft.sh
