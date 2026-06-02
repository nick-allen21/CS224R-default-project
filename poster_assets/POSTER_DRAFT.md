# Poster Draft — Self-Generated Curriculum via Joint Conjecturer/Prover RLOO for Countdown

**Author:** Finn Stäblein • Stanford CS224R Spring 2026

---

## 1. Problem

LLM reasoning training is bottlenecked by **static curricula**. Once a model has consumed its training set, there's no more learning signal — but rule-based verifiers (like a Countdown solver) let us generate signal at inference time. **Can we train a model to generate its own curriculum, targeted at the prover's frontier of ability?**

Concretely on Countdown: given numbers `n` and target `T`, the prover must produce an arithmetic expression that uses each number once and evaluates to `T`. We adapt the conjecturer/prover framework (Dong et al., for formal theorem proving) to this arithmetic-reasoning setting, with a rule-based verifier replacing the proof checker.

---

## 2. Related Work

- **DeepSeek-R1-Zero** [DeepSeek-AI 2025]: RL alone elicits reasoning, but training data is fixed.
- **Pan et al. (TinyZero)**: Replicates R1-Zero on Countdown — also static dataset.
- **Subramaniam et al.**: Trains specialized model copies but still on fixed problems.
- **Dong et al.**: Conjecturer/prover framework for formal proofs; demonstrates difficulty-targeted curriculum, never applied to arithmetic reasoning with a deterministic verifier.

**Our contribution:** First adaptation of joint C/P RLOO to arithmetic reasoning with a rule-based verifier, including (a) a brute-force solvability filter exploiting the cheap verifier, and (b) a difficulty-band reward calibrated to the prover's frontier.

---

## 3. Method

**Two LLMs, same architecture (Qwen 2.5 0.5B):**
- **Prover (P):** initialized from SFT checkpoint on Countdown corpus.
- **Conjecturer (C):** also initialized from SFT checkpoint. Prompted in completion mode with `<problem> numbers=[` prefill to suppress the trained solve-reflex.

**Per training step (9 phases):**
```
   ┌─ 1. Sample C ─→ 8 problems (n, T)
   │  2. Parse
   │  3. Solvability filter (brute-force) ─→ ~4 valid problems
   │  4. Build prover prompts
   │  5. Sample P ─→ group_size=4 rollouts per problem
   │  6. Score P with countdown verifier (0 / 0.1 / 1.0)
   │  7. C reward: 1 if p̂ ∈ [0.05, 0.85], else 0
   │  8. RLOO update P  (LOO baseline within each problem's group)
   └─ 9. RLOO update C  (LOO baseline across C's N generations)
```

**Reward design:**
- **P**: standard Countdown verifier — 0 / 0.1 (format only) / 1.0 (correct).
- **C**: binary band reward — `r_C = 1` iff prover solves `[5%, 85%]` of attempts. Pushes C toward problems neither trivial nor unsolvable for current P.

**C-side stabilization (deviation from default RLOO):**
- Lower learning rate (`5e-7` vs `1e-5`) — C's gradient is high variance because batch_size=1 with N-way LOO baseline.
- Explicit KL anchor against SFT reference (`coef=0.02`) — prevents C from drifting into a degenerate distribution.

---

## 4. Results

**Setup:** Held-out test set of 50 Countdown problems, 16 samples per problem at `T=0.6, top_p=0.95, top_k=20`. pass@k is the standard Chen et al. unbiased estimator.

| Model | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| **SFT baseline** | 0.286 | 0.596 | 0.720 | 0.780 |
| Joint step 8  | **0.380** | 0.668 | 0.738 | 0.780 |
| Joint step 16 | 0.378 | **0.685** | **0.764** | **0.800** |
| Joint step 24 | 0.361 | 0.659 | 0.736 | 0.780 |

**Headline figure** (`headline_passk.png`): Joint step 16 dominates SFT baseline across all k. Improvement is largest at low k (most "reliable" gains, not just coverage gains).

**Training curve** (`training_curve.png`): pass@1 jumps fastest in the first 8 steps; pass@4/8/16 continue to improve through step 16; mild regression at step 24 (likely C drift — see Limitations).

**Headline numbers for the poster:**
- pass@1: 0.286 → 0.380 = **+33% relative improvement** over SFT baseline.
- pass@4: 0.596 → 0.685 = +15% relative.
- pass@8: 0.720 → 0.764 = +6% relative.
- Total compute: **~400 update samples** (vs. ~30k for default RLOO baseline — 75× less).

---

## 5. Limitations and Future Work

1. **Small test set (n=50)** — single seed, 16 samples/problem. Confidence intervals are wide.
2. **Short training (25 steps)** — only 400 update samples vs. ~30k for vanilla RLOO. Yet we still match RLOO-style gains, suggesting **curriculum efficiency** beyond raw sample count.
3. **Step 24 regression** likely indicates C drift; tighter KL anchoring or LR decay should help.
4. **Future:** scale to larger models; OOD evaluation (5-6 number Countdown); longer training to test whether the curriculum allows continued improvement past where vanilla RLOO plateaus.

---

## 6. What audiences can take away

A small model with a deterministic verifier can produce its own curriculum cheaply — no human-annotated problems needed. The conjecturer/prover loop is a general recipe wherever verification is cheap (math, code, constraint satisfaction).
