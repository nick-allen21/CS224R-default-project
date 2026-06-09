#!/bin/bash
# Fair-eval the joint_v10 (Anchored-shaped (fixed)) PROVER checkpoints at
# max_tokens=2048, identical protocol to SFT/RLOO/v6/v9 (eval_v6_v9_rloo.sh) so
# v10 can enter the results table. We eval three checkpoints and pick the best,
# since the peak need not be the latest step (v6 peaked at step 16).
#
# The v10 run trained 43 steps with save_every_n_steps=20, so these should exist:
#   epoch_0_step_20, epoch_0_step_40, latest_checkpoint (~step 43).
#
# VERIFY the checkpoints exist first:
#   modal volume ls default-proj-training \
#     /checkpoints/joint_prover/joint_rloo_default_project/joint_v10
#
# Set WANDB_API_KEY and HF_TOKEN in your shell before running.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"

BUDGET="--max_tokens=2048 --max_model_len=4096 --max_num_batched_tokens=8192"
BASE=/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v10

# NOTE: this run collapsed at ~step 56 but RECOVERED by ~step 85, then crashed at
# step 99 (so there is no epoch_0_step_100). The recovered, fully-trained model is
# in latest_checkpoint (~step 99) -- this is the fair v10 number (recovered AND
# ~100 steps, matching v9). step_20/step_40 were already evaluated (pre-collapse);
# epoch_0_step_60/80 are collapsed/mid-recovery, skip them.
for CKPT in latest_checkpoint; do
  echo "==> Eval v10 ${CKPT} @2048"
  caffeinate -i modal run --detach modal_train.py eval \
    --model_path="${BASE}/${CKPT}/model" \
    --output_dir=/vol/evaluation/eval_results \
    --output_name="eval_joint_v10_${CKPT}_2048" \
    $BUDGET
done

echo
echo "Launched 3 eval jobs (--detach), ~15-20 min each on H100."
echo "Outputs: /vol/evaluation/eval_results/eval_joint_v10_*_2048.json"
echo
echo "When done, download to the local results dir so we can score them:"
echo "  for c in epoch_0_step_20 epoch_0_step_40 latest_checkpoint; do"
echo "    modal volume get default-proj-training \\"
echo "      /evaluation/eval_results/eval_joint_v10_\${c}_2048.json \\"
echo "      ./eval_results_joint/"
echo "  done"
