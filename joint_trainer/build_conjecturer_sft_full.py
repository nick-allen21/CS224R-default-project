"""Build a FULL-size SFT dataset for the conjecturer.

Same structure as build_conjecturer_sft_data.py but uses every available
row from asingh15/countdown_tasks_3to4 (or up to MAX_SAMPLES). Pushes
directly to HF Hub so we don't have to mess with save_to_disk + push later.
"""

from __future__ import annotations
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset, DatasetDict, load_dataset
from joint_trainer.conjecturer_prompt import CONJECTURER_USER


SOURCE_DATASET = "asingh15/countdown_tasks_3to4"
HF_REPO_ID = "finn-staeblein/conjecturer_sft_full"
MAX_SAMPLES = 400_000
TEST_FRACTION = 0.005        # 0.5% test ~ 2000 rows is plenty
SEED = 42


def format_completion(nums, target):
    nums_str = ", ".join(str(int(n)) for n in nums)
    return f"<problem> numbers=[{nums_str}], target={int(target)} </problem>"


def main():
    print(f"[1/5] Loading source dataset: {SOURCE_DATASET} (train split)")
    src = load_dataset(SOURCE_DATASET, split="train")
    print(f"      source rows: {len(src):,}")

    n_take = min(MAX_SAMPLES, len(src))
    print(f"[2/5] Sampling {n_take:,} rows (seed={SEED})")
    rng = random.Random(SEED)
    indices = rng.sample(range(len(src)), n_take)
    sampled = src.select(indices)

    print("[3/5] Formatting (query, completion) records")
    records = [
        {"query": CONJECTURER_USER, "completion": format_completion(row["nums"], row["target"])}
        for row in sampled
    ]

    rng.shuffle(records)
    n_test = max(1, int(len(records) * TEST_FRACTION))
    test_records = records[:n_test]
    train_records = records[n_test:]
    print(f"      train: {len(train_records):,} | test: {len(test_records):,}")

    train_ds = Dataset.from_list(train_records)
    test_ds = Dataset.from_list(test_records)
    ds = DatasetDict({"train": train_ds, "test": test_ds})

    print(f"[4/5] Pushing to HF hub: {HF_REPO_ID}")
    ds.push_to_hub(HF_REPO_ID, private=True)

    print("[5/5] Done. Dataset available at https://huggingface.co/datasets/" + HF_REPO_ID)


if __name__ == "__main__":
    main()
