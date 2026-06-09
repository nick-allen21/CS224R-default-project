#!/bin/bash
# v10 — v9 hyperparameters, but the conjecturer (C) is initialized from the
# FULL conjecturer SFT instead of the prover SFT model.
#
# Key difference vs v8/v9:
#   Those runs never passed --c_model_name, so C started from the *prover* SFT
#   (asingh15/qwen-sft-countdown-defaultproj, the joint.py CLI default). v10
#   starts C from finn-staeblein/conjecturer_sft_full (~100k+ rows of
#   conjecturer-prompt -> <problem> pairs), so C emits well-formed, solvable
#   problems from step 0 — higher initial parseable rate, less risk of tripping
#   the c_min_parseable_rate collapse guard, and a curriculum that's actually
#   on-distribution from the start.
#
# One other deviation from v9: num_problems_per_step 16 -> 32. The full-SFT
# conjecturer is 100% parseable but only ~20% solvable at init, so at 16
# problems/step only ~3 slots/step carry a nonzero (solvable) C reward, making
# C's gradient noisy. Doubling to 32 raises that to ~6 solvable samples/step,
# de-noising the C update WITHOUT raising c_learning_rate (which stays at 1e-6;
# bigger steps on a noisy gradient is what collapsed v7). The reward structure
# already pushes C toward solvable problems; this just gives it more signal.
#
# Everything else mirrors launch_v9.sh (shaped C reward, 50/50 data mix, length
# penalty, self-healing guards). See launch_v9.sh for the rationale on each flag.
#
# Run test_csft.sh FIRST and confirm parseable/solvable rates look healthy
# before launching this.
#
# Set secrets first:  export WANDB_API_KEY=...  HF_TOKEN=...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"

C_INIT="/vol/checkpoints/conjecturer_sft_full/conjecturer_sft_full/conj_sft_full/epoch_0/model"

caffeinate -i modal run --detach modal_train.py joint \
  --c_model_name="$C_INIT" \
  --num_problems_per_step=32 \
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
  --wandb_name=joint_v10
