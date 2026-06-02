"""Build a mini SFT dataset to teach a base model to act as the conjecturer.

Backup plan for the joint trainer: if Path 2 (raw base Qwen 2.5 0.5B as the
conjecturer C) fails to reliably emit problems in our ``<problem> ... </problem>``
format, we want a tiny fine-tuned model that has learned the format from a
handful of (prompt, problem) pairs.

This script materialises that dataset locally. It does NOT train anything.

Pipeline:
    1. Load ``asingh15/countdown_tasks_3to4`` (train split) — the same dataset
       the SFT solver was trained on. Each row has ``nums`` (list[int]) and
       ``target`` (int).
    2. Sample ``N_SAMPLES`` rows with a fixed seed.
    3. For each row, format:
         - ``query``      = ``CONJECTURER_USER`` (the plain user-message string;
                             ``sft_dataset.py`` applies the chat template itself)
         - ``completion`` = ``<problem> numbers=[...], target=T </problem>``
    4. Split into train/test (~90/10) so the existing SFT trainer (which
       expects both splits) works unchanged.
    5. ``save_to_disk`` to ``data/conjecturer_sft/`` and verify by reloading.

Run from the project root:

    python joint_trainer/build_conjecturer_sft_data.py

CPU-only, ~seconds. Requires network access for the initial HF download.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# Allow ``python joint_trainer/build_conjecturer_sft_data.py`` to import
# sibling modules from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from joint_trainer.conjecturer_prompt import CONJECTURER_USER


SOURCE_DATASET = "asingh15/countdown_tasks_3to4"
OUTPUT_DIR = PROJECT_ROOT / "data" / "conjecturer_sft"
N_SAMPLES = 800
TEST_FRACTION = 0.1
SEED = 42


def format_completion(nums: list[int], target: int) -> str:
    """Return the canonical ``<problem> ... </problem>`` completion string."""
    nums_str = ", ".join(str(int(n)) for n in nums)
    return f"<problem> numbers=[{nums_str}], target={int(target)} </problem>"


def build_records(source_rows) -> list[dict]:
    """Convert source rows into (query, completion) records."""
    records: list[dict] = []
    for row in source_rows:
        nums = row["nums"]
        target = row["target"]
        records.append(
            {
                "query": CONJECTURER_USER,
                "completion": format_completion(nums, target),
            }
        )
    return records


def main() -> None:
    print(f"[1/5] Loading source dataset: {SOURCE_DATASET} (train split)")
    src = load_dataset(SOURCE_DATASET, split="train")
    print(f"      source rows: {len(src):,}")

    print(f"[2/5] Sampling {N_SAMPLES} rows (seed={SEED})")
    rng = random.Random(SEED)
    n_take = min(N_SAMPLES, len(src))
    indices = rng.sample(range(len(src)), n_take)
    sampled = src.select(indices)

    print("[3/5] Formatting (query, completion) records")
    records = build_records(sampled)

    # Train/test split — sft_dataset.get_dataloaders requires both.
    rng.shuffle(records)
    n_test = max(1, int(len(records) * TEST_FRACTION))
    test_records = records[:n_test]
    train_records = records[n_test:]
    print(f"      train: {len(train_records)} | test: {len(test_records)}")

    train_ds = Dataset.from_list(train_records)
    test_ds = Dataset.from_list(test_records)
    ds = DatasetDict({"train": train_ds, "test": test_ds})

    print(f"[4/5] Writing dataset to {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR.parent, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR))

    print("[5/5] Reloading from disk to verify")
    reloaded = load_from_disk(str(OUTPUT_DIR))
    assert set(reloaded.keys()) == {"train", "test"}, reloaded.keys()
    assert set(reloaded["train"].column_names) >= {"query", "completion"}, (
        reloaded["train"].column_names
    )
    example = reloaded["train"][0]

    print("\n=== Dataset summary ===")
    print(f"path:        {OUTPUT_DIR}")
    print(f"num train:   {len(reloaded['train'])}")
    print(f"num test:    {len(reloaded['test'])}")
    print(f"columns:     {reloaded['train'].column_names}")
    print("\n--- Example query ---")
    print(example["query"])
    print("\n--- Example completion ---")
    print(example["completion"])
    print("\n[ok] dataset built and verified.")


if __name__ == "__main__":
    main()
