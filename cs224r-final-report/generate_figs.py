"""Generate the W&B-derived result figures for the report.

Figure 1 (curriculum_health.png): C's valid (parseable & solvable) rate and the
prover solve-rate p_hat, Anchored-shaped vs Anchored-shaped (fixed). Shows the
starvation (v9 valid_rate -> 0 most steps) and the fix (v10 always positive,
p_hat tracking the 0.4 target).

Figure 2 (v10_collapse_recovery.png): the v10 prover reward over training -- peak,
transient collapse (~step 56), and recovery (~step 85).

Usage (W&B creds in ~/.netrc):  python cs224r-final-report/generate_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENT, PROJ = "staeblein-stanford-university", "joint_rloo_default_project"
FIGS = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGS, exist_ok=True)


def pick(name):
    runs = [r for r in wandb.Api().runs(f"{ENT}/{PROJ}") if r.name == name]
    # the run with the most logged history (full / latest run of that name)
    return max(runs, key=lambda r: r.summary.get("_step", 0))


def hist(run, keys):
    h = run.history(keys=keys, pandas=True)
    return h


def main():
    v9 = pick("joint_v9")
    v10 = pick("joint_v10")
    keys = ["c/valid_rate", "c/p_hat_mean", "p/reward_mean"]
    h9, h10 = hist(v9, keys), hist(v10, keys)
    C9, C10 = "#c0504d", "#4f81bd"  # v9 red, v10 blue

    # ---- Figure 1: curriculum health (2 panels) ----
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.1))
    ax[0].plot(h9["_step"], h9["c/valid_rate"], color=C9, lw=1.5, label="Anchored-shaped")
    ax[0].plot(h10["_step"], h10["c/valid_rate"], color=C10, lw=1.5, label="Anchored-shaped (fixed)")
    ax[0].set_title("C valid (solvable) rate")
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("fraction valid")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].plot(h9["_step"], h9["c/p_hat_mean"], color=C9, lw=1.5, label="Anchored-shaped")
    ax[1].plot(h10["_step"], h10["c/p_hat_mean"], color=C10, lw=1.5, label="Anchored-shaped (fixed)")
    ax[1].axhline(0.4, color="gray", ls="--", lw=0.9, label="target $\\hat{p}=0.4$")
    ax[1].set_title("prover solve-rate $\\hat{p}$ on C's problems")
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("$\\hat{p}$")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "curriculum_health.png"), dpi=200)
    print("wrote curriculum_health.png")

    # ---- Figure 2: v10 collapse + recovery ----
    fig2, a = plt.subplots(figsize=(6.0, 3.2))
    a.plot(h10["_step"], h10["p/reward_mean"], color=C10, lw=1.6, label="prover reward")
    a.plot(h10["_step"], h10["c/p_hat_mean"], color="#9bbb59", lw=1.3, label="$\\hat{p}$ (difficulty)")
    a.axvspan(56, 80, color="gray", alpha=0.15)
    a.text(68, a.get_ylim()[1]*0.92, "collapse", ha="center", fontsize=8, color="gray")
    a.set_title("Anchored-shaped (fixed): collapse and recovery")
    a.set_xlabel("training step"); a.set_ylabel("value")
    a.legend(frameon=False, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGS, "v10_collapse_recovery.png"), dpi=200)
    print("wrote v10_collapse_recovery.png")


if __name__ == "__main__":
    main()
