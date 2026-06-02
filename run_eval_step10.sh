#!/bin/bash
set -e

CKPT=/vol/checkpoints/rloo_checkpoints/rloo_training/rloo_neb_lr1e-5_bs128_gs8_ent0.001_kl0.001_lrconstant_warmup0.0_temp1.0_topp1.0_topk-1/epoch_0_step_10/model

modal run modal_train.py eval -- \
    --model_path "$CKPT" \
    --eval_dataset asingh15/countdown_tasks_3to4 \
    --output_dir /vol/evaluation/eval_results \
    --output_name rloo_step10 \
    --num_responses 16
