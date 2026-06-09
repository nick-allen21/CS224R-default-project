"""Compute pass@k for any number of Countdown eval JSON(L) files and print a
comparison table (+ optional plot). Handles both eval formats:
  * files with a 'scores' column (countdown_eval.py output), and
  * files with only a 'response' list (re-scored on the fly via compute_score).

Usage:
    python joint_trainer/scripts/passk_table.py FILE1.json FILE2.json ...
    python joint_trainer/scripts/passk_table.py eval_results_fair/*.json --plot out.png

The Chen et al. (2021) unbiased estimator is used:
    pass@k = mean_i [ 1 - C(n - c_i, k) / C(n, k) ]
where c_i = number of correct samples for problem i and n = samples per problem.
"""
import json
import sys
import os
from math import comb

# Make 'evaluation' importable when run from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.countdown import compute_score


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def passk_one(c, n, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def passk(correct_counts, n, k):
    return sum(passk_one(c, n, k) for c in correct_counts) / len(correct_counts)


def correct_counts_and_n(recs):
    counts, n_per, noans, total = [], None, 0, 0
    for r in recs:
        if "scores" in r:
            scores = r["scores"]
        else:
            resp = r["response"]
            gt = r.get("ground_truth") or {"numbers": r["nums"], "target": r["target"]}
            scores = [compute_score(x, gt) for x in resp]
        n_per = len(scores)
        counts.append(sum(1 for s in scores if s >= 1.0))
        if "response" in r:
            for x in r["response"]:
                total += 1
                if "</answer>" not in x:
                    noans += 1
    trunc = (noans / total) if total else float("nan")
    return counts, n_per, trunc


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    plot_path = None
    if "--plot" in sys.argv:
        plot_path = sys.argv[sys.argv.index("--plot") + 1]
        args = [a for a in args if a != plot_path]
    ks = [1, 4, 8, 16]
    rows = []
    print(f"{'model':40s} {'n':>4s} {'K':>3s} " + " ".join(f"p@{k:<5d}" for k in ks) + "  no-ans%")
    print("-" * 92)
    for path in args:
        name = os.path.basename(path).replace(".json", "")
        recs = load_jsonl(path)
        counts, n, trunc = correct_counts_and_n(recs)
        vals = {k: passk(counts, n, k) for k in ks if k <= n}
        rows.append((name, n, vals, trunc))
        line = f"{name:40s} {len(recs):>4d} {n:>3d} "
        line += " ".join(f"{vals.get(k, float('nan')):.3f} " for k in ks)
        line += f"  {trunc*100:4.0f}%" if trunc == trunc else "   n/a"
        print(line)

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        for name, n, vals, _ in rows:
            xs = sorted(vals.keys())
            plt.plot(xs, [vals[k] for k in xs], marker="o", label=name)
        plt.xlabel("k")
        plt.ylabel("pass@k")
        plt.title("pass@k (fair eval, max_tokens=2048)")
        plt.legend(fontsize=7)
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nsaved {plot_path}")


if __name__ == "__main__":
    main()
