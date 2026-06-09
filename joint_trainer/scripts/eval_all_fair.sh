#!/bin/bash
# Fair, apples-to-apples re-eval of EVERY model in ONE consistent harness.
#
# Why this matters: the current result files are NOT comparable.
#   * v7's eval used a 1024-token cap and 34% of its responses never emitted
#     </answer> (truncated mid-reasoning) -> its pass@k is artificially low.
#   * RLOO/IPO were evaluated in a separate batch from the joint runs, and two
#     different RLOO eval files disagree (pass@1 = 0.246 vs 0.453).
# A poster claim of "beats RLOO" is only valid if SFT, IPO, RLOO and the joint
# models are all sampled the SAME way. This script does exactly that.
#
# All evals share:
#   max_tokens=2048, max_model_len=4096  (removes the truncation artifact)
#   temperature=0.6, top_p=0.95, top_k=20, num_responses=16  (countdown_eval.py defaults)
#   eval_dataset=asingh15/countdown_tasks_3to4 (test split)
#
# Set secrets first:  export WANDB_API_KEY=...  HF_TOKEN=...
#
# ---- FILL THESE IN with your actual checkpoint paths on the Modal volume ----
# Find them with:  modal volume ls default-proj-training /checkpoints
RLOO_CKPT="${RLOO_CKPT:-/vol/checkpoints/rloo_checkpoints/REPLACE_WITH_RLOO_RUN/epoch_0_step_100/model}"
IPO_CKPT="${IPO_CKPT:-/vol/checkpoints/REPLACE_WITH_IPO_PATH/model}"
V6_CKPT="${V6_CKPT:-/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v6/epoch_0_step_16/model}"
# v9 checkpoints (created by launch_v9.sh). Eval several and pick the best.
V9_STEP20="${V9_STEP20:-/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v9/epoch_0_step_20/model}"
V9_STEP30="${V9_STEP30:-/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v9/epoch_0_step_30/model}"
V9_LATEST="${V9_LATEST:-/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v9/latest_checkpoint/model}"
# ----------------------------------------------------------------------------

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"

BUDGET="--max_tokens=2048 --max_model_len=4096 --max_num_batched_tokens=8192"
OUTDIR="/vol/evaluation/eval_results_fair"

run_eval () {  # $1 = model_path, $2 = output_name
  echo ">>> eval $2"
  caffeinate -i modal run --detach modal_train.py eval \
    --model_path="$1" \
    --eval_dataset=asingh15/countdown_tasks_3to4 \
    --output_dir="$OUTDIR" \
    --output_name="$2" \
    $BUDGET
}

# Baselines
run_eval "asingh15/qwen-sft-countdown-defaultproj" "fair_sft"
run_eval "$RLOO_CKPT" "fair_rloo"
run_eval "$IPO_CKPT"  "fair_ipo"
# Joint
run_eval "$V6_CKPT"   "fair_joint_v6_step16"
run_eval "$V9_STEP20" "fair_joint_v9_step20"
run_eval "$V9_STEP30" "fair_joint_v9_step30"
run_eval "$V9_LATEST" "fair_joint_v9_latest"

echo "All eval jobs launched (detached). Download results from $OUTDIR when done:"
echo "  modal volume get default-proj-training /evaluation/eval_results_fair ./eval_results_fair"
echo "Then compute pass@k locally with:  python joint_trainer/scripts/passk_table.py eval_results_fair/*.json"
