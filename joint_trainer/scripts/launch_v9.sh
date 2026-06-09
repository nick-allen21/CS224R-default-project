#!/bin/bash
# v9 — "stable v6 hyperparameters at v7 scale, with v8 anti-collapse fixes".
#
# Why v7 collapsed (diagnosed from W&B + eval files):
#   * c_lr=1e-5 (20x v6's 5e-7) + weak KL anchor (0.001) let C drift to
#     all-easy problems by ~step 30 -> p_hat -> 1.0 -> C reward -> 0 -> dead gradient.
#   * entropy=0.001 let P mode-collapse -> pass@16 fell from 0.78 to 0.62.
#   * 100 steps gave the collapse plenty of time to set in.
#
# v9 changes vs v7:
#   1. Shaped C reward (tent peaking at p_hat=0.4) instead of binary band.   [v8]
#   2. 50/50 data mix: half of P's problems come from the fixed Countdown    [v8]
#      set, so even if C collapses P keeps improving like RLOO (graceful
#      degradation -> we should never do worse than RLOO on the real data).
#   3. P length penalty (alpha=0.1) to curb the rambling that tripped the    [v8]
#      eval token cap.
#   4. c_learning_rate=1e-6 (was 1e-5)  -> slows C drift toward easy.
#   5. c_kl_divergence_coefficient=0.02 (was 0.001) -> firmer SFT anchor.
#   6. entropy_coefficient=0.01 (was 0.001) -> preserves P diversity/coverage.
#   7. num_training_steps=100, save_every_n_steps=20 -> checkpoints at
#      20/40/60/80/100. v6 peaked at step 16 so do NOT assume last checkpoint
#      is best; eval each and pick the winner.
#   8. max_tokens=1024 in training (was 512 in early v7) paired with 2048 eval.
#
# Self-healing guards (so an unattended run can't die the way v8 did, where
# c/parseable_rate fell to 0 by ~step 5 and the whole pipeline flatlined):
#   * c_min_parseable_rate=0.5 -> if C drifts off the SFT distribution and
#     <50% of its outputs are parseable, the destabilising update is DISCARDED
#     and C is rolled back to the last healthy checkpoint (ping-pong saves keep
#     it intact). Watch c/collapse_guard_triggered in W&B.
#   * p_min_fixed_problems=8 -> P always trains on >=8 real problems even if C
#     stalls, so the method degrades gracefully to RLOO instead of halting.
#
# Watch these W&B metrics live; stop early if collapse starts:
#   c/p_hat_mean  -> creeping toward 1.0 == curriculum collapsing
#   c/frac_too_easy -> fraction of C problems P already solves
#   c/reward_mean -> should stay well above 0
#   p/accuracy    -> P's solve rate
#
# Set secrets first:  export WANDB_API_KEY=...  HF_TOKEN=...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"

caffeinate -i modal run --detach modal_train.py joint \
  --num_problems_per_step=16 \
  --group_size_p=8 \
  --gradient_accumulation_steps=8 \
  --num_training_steps=100 \
  --save_every_n_steps=20 \
  --learning_rate=1e-5 \
  --c_learning_rate=1e-6 \
  --kl_divergence_coefficient=0.001 \
  --c_kl_divergence_coefficient=0.02 \
  --entropy_coefficient=0.01 \
  --max_tokens=1024 \
  --max_model_len=2048 \
  --gpu_memory_utilization=0.5 \
  --shaped_c_reward \
  --c_reward_peak=0.4 \
  --p_mix_fraction=0.5 \
  --p_length_penalty_alpha=0.1 \
  --p_min_fixed_problems=8 \
  --c_min_parseable_rate=0.5 \
  --c_save_dir=/vol/checkpoints/joint_conjecturer \
  --p_save_dir=/vol/checkpoints/joint_prover \
  --wandb_name=joint_v9
