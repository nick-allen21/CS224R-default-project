"""Completion-style conjecturer prompt for the RAW base model.

The base Qwen 2.5 0.5B model has not been fine-tuned for Countdown, so we
cannot rely on the chat template / instruction-following ability. Instead
we prefill it into the right format with a few-shot completion prompt
that ends mid-token sequence: ``<problem> numbers=[`` — the model just
needs to continue.

Re-exports ``parse_problem`` from the existing chat-style module so the
regex parser is shared (same output format).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from joint_trainer.conjecturer_prompt import parse_problem  # noqa: E402,F401  (re-export)


def build_completion_prompt() -> str:
    """Return a plain-text few-shot completion prompt.

    The prompt ends with ``<problem> numbers=[`` so the base model is
    primed to emit the rest of a problem in the expected format. Do NOT
    apply a chat template to this string.
    """
    return (
        "Here are example Countdown problems. Each is a set of numbers and a target.\n"
        "<problem> numbers=[85, 31, 37, 4], target=76 </problem>\n"
        "<problem> numbers=[79, 54, 4, 18], target=25 </problem>\n"
        "<problem> numbers=[3, 4, 6, 8], target=24 </problem>\n"
        "<problem> numbers=[42, 9, 11], target=53 </problem>\n"
        "<problem> numbers=[12, 7, 50, 2], target=64 </problem>\n"
        "<problem> numbers=[6, 25, 8], target=39 </problem>\n"
        "<problem> numbers=["
    )


if __name__ == "__main__":
    p = build_completion_prompt()
    print(p)
    assert p.endswith("<problem> numbers=["), "prompt must end with the prefill anchor"
    print("\n[ok] prompt ends with '<problem> numbers=['")
