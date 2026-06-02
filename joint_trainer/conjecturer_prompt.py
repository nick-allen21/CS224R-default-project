"""Conjecturer prompt + parser.

The conjecturer asks the SFT-warmed model to generate new Countdown
problems instead of solving them. We mirror the SFT chat template so
the model sees a familiar format.
"""

import re
from typing import Optional


CONJECTURER_SYSTEM = "You are a helpful assistant."

# Mirrors the SFT user prompt structure: long instruction, ends with an
# "Assistant: ..." priming line so the model continues in-role.
CONJECTURER_USER = (
    "A conversation between User and Assistant. The User asks the Assistant "
    "to create a new Countdown arithmetic problem. The Assistant generates "
    "a problem (a set of numbers and a target).\n"
    "User: Generate a new, solvable Countdown problem. Use 3 or 4 distinct "
    "positive integers between 1 and 99 and a target between 10 and 99. "
    "The problem must be solvable using +, -, *, / with each number used "
    "exactly once. Briefly think in <think> </think> tags, then return the "
    "problem in <problem> </problem> tags, for example "
    "<problem> numbers=[3, 4, 6, 8], target=24 </problem>.\n"
    "Here are example problems for reference (do NOT solve them):\n"
    "<problem> numbers=[85, 31, 37, 4], target=76 </problem>\n"
    "<problem> numbers=[79, 54, 4, 18], target=25 </problem>\n"
    "<problem> numbers=[3, 4, 6, 8], target=24 </problem>\n"
    "Generate a new problem (different from the examples).\n"
    "Assistant: Let me create a new problem step by step."
)


def build_conjecturer_prompt(tokenizer) -> str:
    """Return the full chat-templated string ready to feed to vLLM."""
    messages = [
        {"role": "system", "content": CONJECTURER_SYSTEM},
        {"role": "user", "content": CONJECTURER_USER},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


_PROBLEM_RE = re.compile(
    r"<problem>\s*numbers\s*=\s*\[([^\]]+)\]\s*,\s*target\s*=\s*(\d+)\s*</problem>",
    re.IGNORECASE,
)


def parse_problem(text: str) -> Optional[tuple[list[int], int]]:
    """Extract (numbers, target) from generated text.

    Uses the LAST match in the text (model may produce multiple, e.g. echo
    examples then generate). Returns None if no parseable problem found
    or if it violates basic constraints (3-4 numbers, all in [1, 99]).
    """
    matches = list(_PROBLEM_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    nums_str, target_str = m.group(1), m.group(2)
    try:
        numbers = [int(x.strip()) for x in nums_str.split(",")]
        target = int(target_str)
    except ValueError:
        return None
    if len(numbers) < 3 or len(numbers) > 4:
        return None
    if not all(1 <= n <= 99 for n in numbers):
        return None
    if target < 1 or target > 999:
        return None
    return numbers, target


if __name__ == "__main__":
    # Smoke tests for the parser (no model needed).
    cases = [
        ("<problem> numbers=[3, 4, 6, 8], target=24 </problem>", ([3, 4, 6, 8], 24)),
        ("blah <problem>numbers=[1,2,3],target=10</problem> blah", ([1, 2, 3], 10)),
        ("<problem>numbers=[3, 4, 6, 8], target=24</problem>\n<problem>numbers=[1, 2, 5], target=12</problem>", ([1, 2, 5], 12)),
        ("no problem tag here", None),
        ("<problem>numbers=[1, 2], target=3</problem>", None),  # too few numbers
        ("<problem>numbers=[1, 2, 3, 4, 5], target=15</problem>", None),  # too many
        ("<problem>numbers=[100, 2, 3], target=20</problem>", None),  # number too big
    ]
    all_ok = True
    for inp, expected in cases:
        got = parse_problem(inp)
        ok = got == expected
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'}: parse({inp[:60]!r}) -> {got}")
    print("\nAll parser tests pass." if all_ok else "\nSOME TESTS FAILED")
