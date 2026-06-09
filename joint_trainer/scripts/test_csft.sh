#!/bin/bash
# Quick conjtest of the newly-SFT'd conjecturer.
# Confirms parseable + solvable rates before committing to v10.
#
# Points at the FULL-SFT conjecturer (finn-staeblein/conjecturer_sft_full,
# ~100k+ rows) — vs the old mini (720 rows). Update --num_samples for a
# tighter rate estimate if needed.
#
# Set WANDB_API_KEY and HF_TOKEN in your shell before running.

modal run modal_train.py conjtest \
  --model_path=/vol/checkpoints/conjecturer_sft_full/conjecturer_sft_full/conj_sft_full/epoch_0/model \
  --prompt_style=chat \
  --num_samples=20
