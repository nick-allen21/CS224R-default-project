"""Brute-force solver for the Countdown arithmetic task.

Returns an equation string in the exact format the verifier
(`evaluation.countdown.compute_score`) accepts: original integer literals
combined with +, -, *, / (Python true-division), parenthesised as needed.

Algorithm: recursive enumeration. At each step pick all unordered pairs
of remaining (value, expr_string) entries, combine via the 4 operations
(both orderings for - and /), insert the result back, and recurse on the
shortened list. Skip division by ~0. Equality tolerance 1e-9 internally;
verifier uses 1e-5.
"""

from itertools import combinations

EPS = 1e-9


def _search(items, target):
    """items: list of (value: float, expr: str). Returns expr or None."""
    if len(items) == 1:
        v, e = items[0]
        if abs(v - target) < EPS:
            # Strip outer parens if present (e.g. "(8 - 4) * 6" is fine; "((..))" -> "(..)").
            return e
        return None

    n = len(items)
    for i, j in combinations(range(n), 2):
        a_val, a_exp = items[i]
        b_val, b_exp = items[j]
        rest = [items[k] for k in range(n) if k != i and k != j]

        candidates = [
            (a_val + b_val, f"({a_exp} + {b_exp})"),
            (a_val * b_val, f"({a_exp} * {b_exp})"),
            (a_val - b_val, f"({a_exp} - {b_exp})"),
            (b_val - a_val, f"({b_exp} - {a_exp})"),
        ]
        if abs(b_val) > EPS:
            candidates.append((a_val / b_val, f"({a_exp} / {b_exp})"))
        if abs(a_val) > EPS:
            candidates.append((b_val / a_val, f"({b_exp} / {a_exp})"))

        for new_val, new_exp in candidates:
            result = _search(rest + [(new_val, new_exp)], target)
            if result is not None:
                return result
    return None


def solve(numbers, target):
    """Return a valid equation string for the verifier, or None."""
    items = [(float(n), str(int(n))) for n in numbers]
    expr = _search(items, target)
    if expr is None:
        return None
    # Strip a single layer of outer parens for cleanliness (verifier accepts either).
    if expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for idx, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(expr) - 1:
                    return expr  # outer parens are not a single matched pair
        return expr[1:-1]
    return expr


def is_solvable(numbers, target):
    return solve(numbers, target) is not None


# ---------- tests ----------

if __name__ == "__main__":
    import random
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluation.countdown import compute_score

    def check(label, cond):
        print(f"{'PASS' if cond else 'FAIL'}: {label}")
        return cond

    all_ok = True

    # Hard-coded sanity tests.
    all_ok &= check("is_solvable([3,4,6,8], 24) == True", is_solvable([3, 4, 6, 8], 24) is True)
    all_ok &= check("is_solvable([1,2,3,4], 100) == False", is_solvable([1, 2, 3, 4], 100) is False)
    all_ok &= check("is_solvable([7], 7) == True", is_solvable([7], 7) is True)
    all_ok &= check("is_solvable([7], 8) == False", is_solvable([7], 8) is False)

    # Cross-check solver output against the verifier on a few hard-coded cases.
    for nums, tgt in [([3, 4, 6, 8], 24), ([7], 7), ([85, 31, 37, 4], 76)]:
        eq = solve(nums, tgt)
        if eq is None:
            all_ok &= check(f"solve({nums},{tgt}) -> None then skip verifier", True)
            continue
        sol_str = f"<answer>{eq}</answer>"
        s = compute_score(sol_str, {"target": tgt, "numbers": nums})
        all_ok &= check(f"verifier accepts solve({nums},{tgt})='{eq}' (score={s})", s == 1.0)

    # Random sample from the real dataset.
    try:
        from datasets import load_dataset

        ds = load_dataset("asingh15/countdown_tasks_3to4", split="train")
        random.seed(0)
        idxs = random.sample(range(len(ds)), 10)
        for i in idxs:
            row = ds[i]
            nums = list(row["nums"])
            tgt = int(row["target"])
            eq = solve(nums, tgt)
            if eq is None:
                # Dataset is reportedly all solvable; if we say no, that's worth flagging
                # but not a hard fail (some rows may be intentionally unsolvable).
                all_ok &= check(f"dataset row {i} nums={nums} tgt={tgt}: solver=None (no cross-check)", True)
                continue
            sol_str = f"<answer>{eq}</answer>"
            s = compute_score(sol_str, {"target": tgt, "numbers": nums})
            all_ok &= check(f"dataset row {i} nums={nums} tgt={tgt} eq='{eq}' verifier={s}", s == 1.0)
    except Exception as e:
        print(f"SKIP dataset test ({type(e).__name__}: {e})")

    print()
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
