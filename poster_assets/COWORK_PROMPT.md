## Cowork Prompt — CS224R Poster Generation (FINAL)

Copy-paste this verbatim into Cowork.

---

I need you to design a research poster for a Stanford CS224R class project. **The poster will be printed by tiling onto multiple A4 sheets and glued together** — design needs to work as a single coherent layout even after being cut/glued.

### Poster size + tiling spec

- **Final assembled size: 36" wide × 24" tall — LANDSCAPE.** This is the CS224R-recommended size that fits the easels at the poster session.
- **Tiling: 4 wide × 2 tall = 8 A4 sheets in landscape orientation.** I will tile-print using "Poster" mode in Acrobat — it auto-scales.
- Design at **300 DPI** for print quality.
- **Leave a ~0.5" overlap zone** on each internal seam so glue lines won't cut content.
- **Avoid placing critical text within 0.25" of any tile boundary.** Imagine a 4×2 grid overlaid; keep titles, key numbers, axis labels away from those grid lines (every ~9" horizontally, ~12" vertically).
- Export as a **single PDF at exactly 36" × 24" (landscape)**. Save to `/Users/finnstablein/code/CS224R-default-project/poster_assets/poster_draft.pdf`.

---

### Title + headline

**Title:** Self-Generated Curriculum via Joint Conjecturer/Prover RLOO for Arithmetic Reasoning

**Authors:** Finn Stäblein, Nick Allen • Stanford CS224R Spring 2026

**Headline takeaway (subtitle, ≥36pt):** A small (0.5B) LLM with a deterministic verifier can train its own curriculum — yielding **+111% relative pass@1 over SFT with ~6× less compute than vanilla RLOO**, while preserving sample diversity.

---

### Section 1: Problem (Column 1, top)

LLM reasoning training is bottlenecked by static curricula. Verifiable tasks (math, code) let us score outputs at training time — can we use that to **generate our own problems** at the model's frontier of ability?

We test this on **Countdown**: given a list of numbers `n` (e.g., `[3, 4, 6, 8]`) and a target `T` (e.g., `24`), produce an arithmetic expression using +, −, ×, ÷ that uses each number exactly once and equals T. A rule-based Python verifier scores correctness.

### Section 2: Related Work (Column 1)

- **DeepSeek-R1-Zero, TinyZero (2025):** RL from rule-based verifier produces strong reasoning, but on static training data.
- **Dong et al.:** Conjecturer/prover framework for **formal theorem proving** with a proof checker.
- **Subramaniam et al.:** Multi-model co-evolution on fixed problems.
- **Kirk et al. (2024):** Documents diversity loss as a known failure of RLHF.

**Our novel contribution:** First conjecturer/prover joint training applied to **arithmetic reasoning with a rule-based verifier**, plus a brute-force solvability filter and a shaped difficulty-band reward.

### Section 3: Method (Column 1, with Figure 1 — method diagram)

**Two LLMs — same Qwen 2.5 0.5B architecture, both warm-started from SFT.**

**Per training step (9 phases):**
```
1. Sample C → 16 candidate Countdown problems (numbers, target)
2. Parse problems via regex
3. Solvability filter (brute-force solver) → ~8 valid
4. Build prover prompts (+ mix in 8 fixed-dataset problems)
5. Sample P → 8 rollouts per problem
6. Score P with the verifier (+ length penalty)
7. Compute C's reward from p̂ = P's solve rate per problem
8. RLOO update P
9. RLOO update C
```

**Reward design (v9):**
- **P:** verifier score (0 / 0.1 / 1.0) − α·(length / max_length) penalty
- **C (shaped tent):** `r_C = max(0, 1 − 2.5·|p̂ − 0.4|)` — peaks at moderate difficulty (p̂=0.4), zero outside [0, 0.8]

**Why both shaped + mixed data + length penalty:** an earlier configuration (v7) collapsed because (a) binary reward + drift killed C's gradient; (b) P over-specialized to C's narrow distribution; (c) eval token cap penalized long chains-of-thought. All three issues addressed in v9.

**Figure 1 — please generate** a clean 9-phase loop diagram. Boxes labeled 1–9, arrows between them, color-code C-phases (1–3, 9) blue and P-phases (4–8) orange. Minimal and readable from 6 feet.

---

### Section 4: Experimental Setup (Column 2, top)

- **Model:** Qwen 2.5 0.5B Base → SFT (on `Asap7772/cog_behav_all_strategies` Countdown corpus) → joint RLOO
- **Test set:** 50 held-out Countdown problems, 16 samples per problem at temp 0.6, top_p 0.95, top_k 20
- **Eval:** vLLM, **max_tokens=2048** (fair budget — see Limitations)
- **Compute:** Single H100 via Modal; v9 trained 100 steps in ~3–4 hr

### Section 5: Configurations (Column 2 — small table)

| Run | C reward | Mix | Update samples | Notes |
|---|---|---|---|---|
| v6 (joint) | binary band | 0% C/fixed | ~400 | minimal-viable joint training |
| **v9 (joint)** | **shaped tent** | **50% / 50%** | **~13,000** | **+ length penalty, conservative C hyperparams** |
| RLOO (single-policy) | n/a | 100% fixed | ~80,000 | vanilla RLOO baseline at 80 steps |

### Section 6: Results — fair-eval at max_tokens=2048, n=50 test problems, k=16 samples (Column 2)

```
                          pass@1   pass@4   pass@8   pass@16
SFT baseline              0.244    0.534    0.669    0.760
RLOO (~80 steps)          0.547    0.694    0.718    0.740   ← best pass@1
Joint v6 step 16          0.345    0.651    0.748    0.800   ← best pass@16
Joint v9 latest           0.515    0.654    0.704    0.760   ← v9 — comparable to RLOO at 6× less compute
```

**Headline numbers:**
- **v9 pass@1: 0.244 → 0.515** = **+111% relative over SFT**
- **v9 reaches comparable performance to RLOO at ~6× less compute** (13k vs 80k update samples)
- **Joint methods preserve diversity:** v6 (0.800) > v9 = SFT (0.760) > RLOO (0.740) on pass@16
- **Vanilla RLOO does NOT exceed the untrained SFT baseline on pass@16** despite ~80,000 update samples — characteristic RLHF-style diversity loss. Joint training matches or exceeds SFT on pass@16.

### Section 7: Figures (Column 2)

- **Figure 2 (`headline_passk.png`):** pass@k bar chart, SFT vs RLOO vs v6 step 16 vs v9 — note that the existing PNG only has SFT vs v6, you may need to redraw to include v9. If unable to redraw with new data, use the existing PNG as-is and label clearly which models are shown.
- **Figure 3 (`training_curve.png`):** Use existing — pass@1 and pass@4 across v6 training steps 0/8/16/24. Same — existing PNG is fine.

---

### Section 8: Key Findings (Column 3)

**Finding 1 — Joint curriculum matches vanilla RLOO at ~6× less compute.**
v9 hits pass@1 = 0.515 using ~13,000 update samples. Vanilla RLOO needs ~80,000 samples to reach 0.547 — comparable performance but 6× the training cost. The conjecturer's adaptive problem generation makes more efficient use of each training step.

**Finding 2 — Joint methods preserve sample diversity; vanilla RLOO does not.**
After ~80,000 update samples, RLOO reaches pass@16 = 0.740 — still below the untrained SFT baseline at 0.760. Both joint variants match or exceed SFT diversity (v6 = 0.800, v9 = 0.760). The curriculum signal acts as an implicit regularizer against over-specialization. Matches Kirk et al. 2024 on RLHF-induced diversity loss.

**Finding 3 — The joint-training regime is narrow but stable when properly designed.**
v9's success depended on four design choices: (a) shaped C reward (continuous gradient even when C drifts from the target band), (b) 50% mixed fixed-dataset / curriculum data (anchors P to general task distribution), (c) length penalty α=0.1 (discourages rambling without hard truncation), (d) conservative C-side hyperparameters (c_lr=1e-6, c_kl=0.02). Without these, two-policy joint RL collapses — the conjecturer's gradient dies, or it drifts off the SFT distribution.

### Section 9: Methodological notes (Column 3, compact)

- **Eval token budget matters more than we initially measured.** Initial evals at max_tokens=1024 silently truncated longer chains-of-thought before they could emit `<answer>` — disproportionately penalizing more-trained models. All numbers reported here use max_tokens=2048.
- **Test set size n=50 with 16 samples per problem.** Paired-bootstrap confidence intervals are wide; treat differences <0.05 absolute as suggestive, not conclusive.

### Section 10: Limitations (Column 3)

- **n=50 test set, single seed.** Paired-bootstrap CIs are wide.
- **0.5B model.** Small enough that diversity collapse is sharper than at larger scale.
- **Solver filter accepts non-integer intermediates.** Inherits the verifier's true-division semantics.
- **C-SFT side experiment showed limits of small-data initialization.** A model SFT'd on 720 (prompt, problem) pairs achieved 100% format-parseable but 0% solvable — surface patterns learned without internalizing implicit task constraints.

### Section 11: Future Work (Column 3)

- **Adaptive difficulty band** that shifts as the prover improves (matching Dong et al.)
- **Held-out eval during training** for early stopping at peak
- **Larger model (3B–7B)** — curriculum methods may show stronger gains at scale
- **Dedicated SFT corpus for C with verifier-augmented loss** that penalizes unsolvable outputs

---

### Section 12: Conclusion box (Column 3, bottom-right, highlighted)

**Joint conjecturer/prover RLOO works for arithmetic reasoning with a rule-based verifier.** v9 more than doubles SFT pass@1 while matching vanilla RLOO at ~6× less compute. Both joint variants (v6, v9) preserve sample diversity on pass@16, whereas vanilla RLOO collapses it. The principles transfer: shaped reward, distributional anchoring via mixed data, length penalty, and conservative C-side regularization are necessary ingredients for stable two-policy joint training.

---

### Layout guidance (36" wide × 24" tall landscape)

3-column layout fits the 4×2 tile grid naturally:
- **Top strip (full width, ~3"-4" tall):** Title, authors, headline takeaway in ≥72pt title / ≥36pt subtitle.
- **Column 1 (left, ~12" wide):** Sections 1, 2, 3 (with Figure 1).
- **Column 2 (middle, ~12" wide):** Sections 4, 5, 6, 7 (with Figures 2 and 3).
- **Column 3 (right, ~12" wide):** Sections 8, 9, 10, 11; Section 12 (Conclusion) as a bottom-right box.

Each column ~12" wide which is comfortably more than 1 A4 sheet's height (8.27" in landscape). Column dividers should align roughly with tile seams.

### Tone + style

- Academic but readable from a distance — section headers ≥36pt, body text ≥18pt.
- Use a clean sans-serif (Inter, Source Sans Pro, or similar).
- Honest about limitations — do not over-claim. The diagnostic / failure-boundary story is part of the contribution.
- Stanford colors are fine but don't lean on them; keep it visually clean.

### Figures to embed (existing PNGs)

These exist as PNG files in `/Users/finnstablein/code/CS224R-default-project/poster_assets/`:
- `headline_passk.png` — pass@k bar chart, currently SFT vs v6 step 16 only
- `training_curve.png` — pass@1 / pass@4 across v6 training steps

**If you can redraw `headline_passk.png` to include all four configurations (SFT, RLOO ~80 steps, v6, v9) using the numbers in Section 6, please do.** Otherwise use the existing PNG with appropriate caption.

### Critical constraints

- **No hallucinated numbers.** Only use numbers from this document.
- **No emoji.** Stanford academic context.
- **Render `[PENDING]` rows as gray.** Numbers may arrive shortly; I'll do a text replace.
- **No fancy backgrounds.** White is fine; keep ink consumption realistic for tiled home/office print.
- **Save your output to** `/Users/finnstablein/code/CS224R-default-project/poster_assets/poster_draft.pdf`.

Confirm you understand the size + tiling constraints before starting. Ask clarifying questions if any section is unclear.
