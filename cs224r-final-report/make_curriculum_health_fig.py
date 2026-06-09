"""Generate figs/curriculum_health.png — the v9 starvation 'smoking gun'.

Pulls per-step c/valid_rate and c/reward_mean for the v9 run from W&B and plots
them, making visible that the conjecturer's signal is zero on ~71% of steps.

Usage (W&B key must be set):
    export WANDB_API_KEY=...
    python cs224r-final-report/make_curriculum_health_fig.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "staeblein-stanford-university"
PROJECT = "joint_rloo_default_project"
RUN_NAME = "joint_v9"
OUT = os.path.join(os.path.dirname(__file__), "figs", "curriculum_health.png")


def main():
    api = wandb.Api()
    runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}") if r.name == RUN_NAME]
    if not runs:
        raise SystemExit(f"No run {RUN_NAME!r} in {ENTITY}/{PROJECT}")
    hist = runs[0].history(keys=["c/valid_rate", "c/reward_mean"], pandas=True)
    step = hist.get("_step", range(len(hist)))

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(step, hist["c/valid_rate"], label="c/valid_rate", lw=1.6)
    ax.plot(step, hist["c/reward_mean"], label="c/reward_mean", lw=1.6)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("value")
    ax.set_title("v9 conjecturer health: signal is zero on ~71% of steps")
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
