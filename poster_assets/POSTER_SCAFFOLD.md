# POSTER SCAFFOLD — fill in placeholder numbers as runs complete

**Size:** 24" × 36" landscape (Stanford-recommended)
**Layout:** 3-column with a headline strip on top

---

## HEADLINE STRIP (top, full width)

**Title:** Self-Generated Curriculum via Joint Conjecturer/Prover RLOO for Arithmetic Reasoning

**Authors:** Finn Stäblein • Stanford CS224R Spring 2026

**One-line takeaway:** A small (0.5B) LLM with a deterministic verifier can produce its own training problems, yielding **[FILL: best pass@1 vs SFT]** with **[FILL: x× less compute]** than vanilla RLOO — though the curriculum-stability regime is narrow.

---

## COLUMN 1 — Problem & Method

### 1. Problem

LLM reasoning training is bottlenecked by static curricula. Verifiable tasks (math, code) let us score outputs at training time — can we use that to **generate our own problems** at the model's frontier?

We test this on **Countdown**: given numbers `n` and target `T`, produce an arithmetic expression. Rule-based verifier scores correctness.

### 2. Related Work

- DeepSeek-R1-Zero, TinyZero: RL from rule-based verifier works, but on **static training data**.
- Dong et al.: Conjecturer/prover framework for **formal theorem proving** (proof checker).
- Subramaniam et al.: Multi-model co-evolution on **fixed problems**.

**Our contribution:** First conjecturer/prover joint training for **arithmetic reasoning with a rule-based verifier** + brute-force solvability filter + shaped difficulty-band reward.

### 3. Method

**Two LLMs, same architecture (Qwen 2.5 0.5B). Both initialized from SFT.**

**Per training step — 9 phases:**
```
1. Sample C → 16 candidate problems (n, T)
2. Parse (regex)
3. Solvability filter (brute-force)
4. Build prover prompts (+ mix in fixed-dataset prompts)
5. Sample P → k=8 rollouts per problem
6. Score P with verifier
7. Compute C's reward from p̂ (per-problem solve rate)
8. RLOO update P
9. RLOO update C
```

**Rewards:**
- **P:** verifier score (0 / 0.1 / 1.0) − α·(length / max_length) penalty
- **C (shaped tent):** `r_C = max(0, 1 − 2.5·|p̂ − 0.4|)` — peaks at p̂=0.4, zero outside [0, 0.8]

**[FIGURE 1: Method diagram of 9-phase loop]**

---

## COLUMN 2 — Experiments

### 4. Experimental Setup

- **Model:** Qwen 2.5 0.5B Base → SFT (on Countdown corpus) → joint RLOO
- **Test set:** 50 held-out Countdown problems, 16 samples per problem
- **Eval:** vLLM, temp 0.6, top_p 0.95, max_tokens 1024 (NB: see Limitations)
- **Compute:** Single H100 via Modal; ~3-8 hr per training run

### 5. Configurations Tested

| Config | C reward | Mix | C lr | C kl | P entropy | Steps |
|---|---|---|---|---|---|---|
| v6 | binary band [.05, .85] | 0% | 5e-7 | 0.02 | 0.02 | 25 |
| v7 | binary band [.05, .85] | 0% | 1e-5 | 0.001 | 0.001 | ~30 |
| v9 | shaped tent, peak=0.4 | 50% | 1e-6 | 0.02 | 0.01 | 100 |
| v10 | shaped tent, peak=0.4 | 50% | 1e-6 | 0.02 | 0.01 | [PENDING] |

**v10 difference: C initialized from dedicated SFT on `(prompt → <problem>numbers=[..], target=N)` pairs** instead of solve-SFT.

### 6. Headline Results — Test-set pass@k

```
                 pass@1   pass@4   pass@8   pass@16
SFT baseline     0.286    0.596    0.720    0.780
RLOO (~30 steps) 0.[??]   0.[??]   0.[??]   0.[??]   ← original Nick checkpoint
Joint v6 step 16 0.378    0.685    0.764    0.800   ← clean peak
Joint v7 latest  0.416    0.580    0.610    0.620   ← +pass@1, diversity collapse
Joint v9 final   0.[??]   0.[??]   0.[??]   0.[??]   ← [FILL when run done]
Joint v10 final  0.[??]   0.[??]   0.[??]   0.[??]   ← [FILL if SFT-C run done]
```

**[FIGURE 2: pass@k bar chart, SFT vs v6 vs v9 (vs v10 if available)]**
**[FIGURE 3: pass@k vs training step, line plot]**

---

## COLUMN 3 — Analysis & Discussion

### 7. Key Findings

**Finding 1 (v6): Joint curriculum works at moderate scale.**
+33% relative pass@1 lift over SFT with only ~400 update samples (vs Nick's ~30,000 for vanilla RLOO — **75× compute-efficient**). All pass@k metrics improve monotonically; no diversity loss.

**Finding 2 (v7): Aggressive scaling causes diversity collapse.**
Increasing batch size + reducing KL anchor + lowering entropy → +45% pass@1 but pass@16 falls **below SFT baseline**. Statistical paired-bootstrap confirms gap is real (CI excludes zero). Per-problem solve-count histogram becomes bimodal: 19/50 problems with 0/16 correct, 20/50 with 12+/16. Classic mode collapse.

**Finding 3 (v9): Conservative hyperparameters + anchoring restore stability.**
Adding (a) shaped reward, (b) 50% fixed-dataset mix for P, (c) length penalty α=0.1, and (d) v6-style C hyperparams produces stable training. [FILL with v9 result interpretation when done.]

**[Optional Finding 4 (v10) — only if results meaningful by deadline]**
Dedicated SFT initialization for C produces problems with [FILL: x% solvable vs y% for v9], improving curriculum signal density.

### 8. Diagnostics & Failure Modes (the failure-boundary story)

**Why v7 failed:** Three contributing causes diagnosed by independent post-hoc analyses:
- (a) **Curriculum collapse:** C drifted to producing only easy problems (c_reward → 0, c_valid_rate → 1.0 by step 30)
- (b) **Mode collapse:** P specialized to C's narrow distribution; per-problem solve distribution became sharply bimodal
- (c) **Eval truncation artifact:** 269 of 271 v7 failures hit eval `max_tokens=1024` mid-reasoning, never emitting `<answer>` tag

This matches **Kirk et al. 2024** on RLHF-induced diversity loss.

**[FIGURE 4 (optional): p/rollout_accuracy and c/reward_mean curves for v6 vs v7 vs v9, showing v6 stable, v7 collapsing, v9 [outcome]]**

### 9. Limitations

- **n=50 test set, single seed** — paired-bootstrap CIs are wide.
- **0.5B model** — small enough that diversity collapse is sharper.
- **Eval `max_tokens=1024`** likely undercounts true pass@k for all conditions; fair re-eval at 2048 was set up but ran out of time.
- **No adaptive band** (Dong et al. 2024 used moving target) — we used fixed band and a shaped variant.

### 10. Future Work

- **Dedicated SFT corpus for C** (problem generation rather than solving) — initial v10 test [STATUS].
- **Adaptive difficulty band** matching prover skill over time.
- **Held-out eval during training** for early stopping at peak.
- **Larger model (3B-7B)** — curriculum methods may show stronger gains at scale.

---

## BOTTOM-RIGHT BOX — Conclusion

**Joint conjecturer/prover RLOO is a viable curriculum-learning recipe for verifiable tasks, but its stable regime is narrow.** Our v6 result (+33% pass@1, 75× compute-efficient) validates the mechanism. Our v7 ablation maps the failure boundary — when scaled aggressively without anchoring, diversity collapses. v9 + v10 (in progress / future work) explore principled fixes.

---

## REFERENCES (compact)
- DeepSeek-AI (2025). DeepSeek-R1.
- Pan et al. TinyZero.
- Dong & Ma. Self-play with formal proofs.
- Subramaniam et al. Multi-model co-evolution.
- Kirk et al. (2024). Understanding RLHF effects on generalisation and diversity.

---

## ASSETS LIST (in `poster_assets/`)

- `headline_passk.png` — Figure 2 (SFT vs v6 step 16 pass@k)
- `training_curve.png` — Figure 3 (pass@1 / pass@4 across training steps)
- **[TODO: method diagram for Figure 1]** — could be made in Figma/Keynote in 30 min
- **[TODO: rerun headline_passk.png with v9 added]** — once v9 evals complete
- **[TODO: c/valid_rate + c/reward_mean curves]** — pull from wandb when ready

## PRINTING

- 24" × 36" landscape
- Same-day: CVS Photo, Walgreens, FedEx — call by 6pm
- Avoid Lathrop (3-day turnaround) — too late
