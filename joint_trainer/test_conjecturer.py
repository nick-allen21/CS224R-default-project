"""Smoke test: does the SFT model produce parseable + solvable problems
from the conjecturer prompt?

Usage:
    python joint_trainer/test_conjecturer.py --num_samples 20

Reports:
    - parseable rate (regex extraction succeeded)
    - solvable rate (brute-force solver finds a solution)
    - first few raw outputs for inspection
    - distribution of (num_nums, num_unique_problems)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from joint_trainer.conjecturer_prompt import build_conjecturer_prompt, parse_problem
from joint_trainer.conjecturer_prompt_base import build_completion_prompt
from joint_trainer.solver import is_solvable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="asingh15/qwen-sft-countdown-defaultproj")
    parser.add_argument(
        "--prompt_style",
        choices=["chat", "completion"],
        default="chat",
        help="chat: SFT-style chat template (default). completion: raw "
             "few-shot completion prompt for the base model.",
    )
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Default 1.0 for chat, 0.7 for completion.",
    )
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--print_first_n", type=int, default=5)
    args = parser.parse_args()

    if args.temperature is None:
        args.temperature = 0.7 if args.prompt_style == "completion" else 1.0

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if args.prompt_style == "completion":
        prompt = build_completion_prompt()
        stop_tokens = ["</problem>", "\n\n"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        prompt = build_conjecturer_prompt(tokenizer)
        stop_tokens = ["</problem>"]

    print("=" * 80)
    print("CONJECTURER PROMPT")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.num_samples,
        stop=stop_tokens,
        include_stop_str_in_output=True,
    )

    outputs = llm.generate([prompt], sampling_params)
    completions = [o.text for o in outputs[0].outputs]

    # In completion mode the model is prefilled with "<problem> numbers=["
    # which is NOT part of its output. Re-attach the prefill so the regex
    # parser (which expects the full "<problem> numbers=[...]" pattern) works.
    if args.prompt_style == "completion":
        prefill_tail = "<problem> numbers=["
        parse_inputs = [prefill_tail + c for c in completions]
    else:
        parse_inputs = completions

    print("\n" + "=" * 80)
    print(f"FIRST {args.print_first_n} RAW COMPLETIONS")
    print("=" * 80)
    for i, c in enumerate(completions[: args.print_first_n]):
        print(f"\n--- Sample {i} ---")
        print(c)

    parsed = [parse_problem(c) for c in parse_inputs]
    n_parseable = sum(1 for p in parsed if p is not None)

    solvable_results = []
    for p in parsed:
        if p is None:
            solvable_results.append(None)
        else:
            nums, target = p
            solvable_results.append(is_solvable(nums, target))
    n_solvable = sum(1 for s in solvable_results if s is True)

    problem_strs = [
        f"nums={p[0]}, T={p[1]}" if p is not None else "UNPARSED"
        for p in parsed
    ]
    counts = Counter(problem_strs)

    n_total = len(completions)
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total samples:    {n_total}")
    print(f"Parseable:        {n_parseable}/{n_total} ({100*n_parseable/n_total:.0f}%)")
    print(f"Solvable:         {n_solvable}/{n_total} ({100*n_solvable/n_total:.0f}%)")
    print(f"Unique outputs:   {len(set(problem_strs))}/{n_total}")
    print(f"\nDistribution of generated problems:")
    for problem_str, count in counts.most_common():
        print(f"  [{count}x] {problem_str}")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    if n_parseable / n_total >= 0.8:
        print("GREEN: parseable rate >= 80%. Prompt is usable.")
    elif n_parseable / n_total >= 0.4:
        print("YELLOW: parseable rate 40-80%. Prompt needs iteration but is workable.")
    else:
        print("RED: parseable rate < 40%. Prompt needs major rework.")


if __name__ == "__main__":
    main()
