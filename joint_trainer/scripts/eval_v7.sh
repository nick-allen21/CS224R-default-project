#!/bin/bash
# Evaluates the two joint_v7 checkpoints against the held-out Countdown test set.
# Results saved to /vol/evaluation/eval_results/ on the Modal volume.

# Set WANDB_API_KEY in your shell before running: export WANDB_API_KEY=...

# Eval 1: joint_v7 step 20 (permanent checkpoint)
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v7/epoch_0_step_20/model \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_joint_v7_step_20

# Eval 2: joint_v7 latest checkpoint (whatever was last saved before cancel)
caffeinate -i modal run --detach modal_train.py eval \
  --model_path=/vol/checkpoints/joint_prover/joint_rloo_default_project/joint_v7/latest_checkpoint/model \
  --output_dir=/vol/evaluation/eval_results \
  --output_name=eval_joint_v7_latest
