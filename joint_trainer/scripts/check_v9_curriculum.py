"""Did v9's conjecturer actually train, or was it starved by the grad_accum bug?

Pulls joint_v9's per-step curriculum-health metrics from W&B and reports whether
C ever received a real reward signal. This decides whether the paper's central
"joint curriculum" claim is supported (C trained) or needs reframing (C starved).

Usage (W&B key must be set):
    export WANDB_API_KEY=...
    python joint_trainer/scripts/check_v9_curriculum.py            # defaults to joint_v9
    python joint_trainer/scripts/check_v9_curriculum.py joint_v8   # or any run name
"""

import sys

import wandb

ENTITY = "staeblein-stanford-university"
PROJECT = "joint_rloo_default_project"
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "joint_v9"

KEYS = ["c/valid_rate", "c/reward_mean", "c/parseable_rate",
        "c/p_hat_mean", "c/frac_too_easy"]


def main():
    api = wandb.Api()
    runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}") if r.name == RUN_NAME]
    if not runs:
        print(f"No run named {RUN_NAME!r} in {ENTITY}/{PROJECT}. "
              f"Available: {[r.name for r in api.runs(f'{ENTITY}/{PROJECT}')][:20]}")
        return
    run = runs[0]
    hist = run.history(keys=KEYS, pandas=True)
    if hist.empty:
        print(f"{RUN_NAME}: no history for {KEYS}.")
        return

    print(f"=== {RUN_NAME}  ({len(hist)} logged steps) ===\n")
    for k in KEYS:
        if k not in hist:
            print(f"  {k:20s}  (not logged)")
            continue
        col = hist[k].dropna()
        if col.empty:
            print(f"  {k:20s}  (all NaN)")
            continue
        nonzero = (col > 1e-9).mean()
        print(f"  {k:20s}  mean={col.mean():.4f}  min={col.min():.4f}  "
              f"max={col.max():.4f}  frac_steps_>0={nonzero:.2f}")

    vr = hist.get("c/valid_rate")
    rm = hist.get("c/reward_mean")
    print("\n--- VERDICT ---")
    if vr is not None and rm is not None:
        vr, rm = vr.dropna(), rm.dropna()
        starved = (vr <= 1e-9).mean() if len(vr) else 1.0
        rm_zero = (rm <= 1e-9).mean() if len(rm) else 1.0
        print(f"  Steps with valid_rate==0:  {starved:.0%}")
        print(f"  Steps with reward_mean==0: {rm_zero:.0%}")
        if starved > 0.8 or rm_zero > 0.8:
            print("  => C WAS STARVED. v9's gains are NOT attributable to the "
                  "curriculum. Reframe the paper / get corrected-v10 results.")
        elif starved > 0.3:
            print("  => PARTIAL. C trained on some steps only. Caveat the claim.")
        else:
            print("  => C TRAINED. The joint-curriculum claim is supported.")


if __name__ == "__main__":
    main()
