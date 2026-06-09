#!/bin/bash
# Mini-SFT for the conjecturer (C) — backup plan.
#
# Trains Qwen 2.5 0.5B Base on a small set of (conjecturer_prompt, problem)
# pairs so the model learns to emit problems in the
# ``<problem> numbers=[...], target=T </problem>`` format. Modeled on
# ``sft_trainer/train_sft_modal.sh``.
#
# BEFORE RUNNING:
#   1. Build the dataset locally:
#        python joint_trainer/build_conjecturer_sft_data.py
#      This writes ``data/conjecturer_sft/`` via ``Dataset.save_to_disk``.
#
#   2. IMPORTANT — ``sft_trainer/sft_dataset.py`` calls
#      ``datasets.load_dataset(dataset_name, split=...)``, which does NOT
#      read a directory produced by ``save_to_disk``. You have two options:
#
#        (a) Push the dataset to HF Hub, then point DATASET_NAME at the repo:
#              huggingface-cli login
#              python -c "from datasets import load_from_disk; \
#                load_from_disk('data/conjecturer_sft').push_to_hub( \
#                'YOUR_HF_USER/conjecturer_sft_mini', private=True)"
#              export DATASET_NAME=YOUR_HF_USER/conjecturer_sft_mini
#
#        (b) Or modify ``sft_dataset.py`` to use ``load_from_disk`` when the
#            path exists locally. (Out of scope for this script — DCC 1
#            "don't touch the existing SFT trainer code".)
#
#   3. Required env vars:
#        export WANDB_API_KEY=...
#        export HF_TOKEN=...  # for push_to_hub above and gated pulls
#
# Useful Modal overrides:
#   export MODAL_GPU=H100!
#   export MODAL_VOLUME_NAME=default-proj-training
#   export MODAL_TIMEOUT_SECONDS=3600  # this run is small, an hour is plenty

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-3600}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

# Tiny fine-tune — goal is to teach the format, not to optimize.
batch_size="${BATCH_SIZE:-8}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
gradient_clipping="${GRADIENT_CLIPPING:-1.0}"
weight_decay="${WEIGHT_DECAY:-0.01}"
warmup_ratio="${WARMUP_RATIO:-0.05}"
max_prompt_length="${MAX_PROMPT_LENGTH:-512}"
max_response_length="${MAX_RESPONSE_LENGTH:-128}"  # completions are short
output_dir="${OUTPUT_DIR:-/vol/checkpoints/conjecturer_sft_mini}"
model_name="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
# Default to a HF repo placeholder — override after push_to_hub.
dataset_name="${DATASET_NAME:-YOUR_HF_USER/conjecturer_sft_mini}"
wandb_project="${WANDB_PROJECT:-conjecturer_sft_mini}"
save_model="${SAVE_MODEL:-1}"
gradient_checkpointing="${GRADIENT_CHECKPOINTING:-1}"

# ~100-200 update steps with 720 train rows and batch_size=8 -> ~90 updates/epoch,
# so 2 epochs gives ~180 updates. Adjust EPOCHS if you want a longer run.
read -r -a lrs <<< "${LRS:-1e-5}"
read -r -a epochs <<< "${EPOCHS:-2}"

num_lrs=${#lrs[@]}
num_epochs=${#epochs[@]}

if [[ $num_lrs -ne $num_epochs ]]; then
    echo "Error: LRS and EPOCHS must have the same number of values"
    exit 1
fi

if [[ "$dataset_name" == YOUR_HF_USER/* ]]; then
    echo "WARNING: DATASET_NAME is still the placeholder ($dataset_name)."
    echo "         Either push data/conjecturer_sft/ to HF Hub and re-export"
    echo "         DATASET_NAME, or patch sft_dataset.py to use load_from_disk."
    echo "         See the header comment of this script for instructions."
fi

for i in "${!lrs[@]}"; do
    curr_lr="${lrs[$i]}"
    curr_epochs="${epochs[$i]}"
    wandb_name="${WANDB_NAME_PREFIX:-conj_sft_lr${curr_lr}_ep${curr_epochs}}"

    command=(
        modal run --detach "$PROJECT_ROOT/modal_train.py"
        sft
        --model_name "$model_name"
        --dataset_name "$dataset_name"
        --output_dir "$output_dir"
        --max_prompt_length "$max_prompt_length"
        --max_response_length "$max_response_length"
        --batch_size "$batch_size"
        --gradient_accumulation_steps "$gradient_accumulation_steps"
        --gradient_clipping "$gradient_clipping"
        --num_epochs "$curr_epochs"
        --learning_rate "$curr_lr"
        --weight_decay "$weight_decay"
        --warmup_ratio "$warmup_ratio"
        --wandb_project "$wandb_project"
        --wandb_name "$wandb_name"
        --save_model "$save_model"
        --gradient_checkpointing "$gradient_checkpointing"
    )

    printf 'Executing command: '
    printf '%q ' "${command[@]}"
    printf '\n'
    "${command[@]}"
done
