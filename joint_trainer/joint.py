"""High-level joint conjecturer + prover RLOO training orchestration.

This script extends the single-policy RLOO trainer (``rloo_trainer/rloo.py``)
to a two-policy setup where:

1) The conjecturer (C) generates Countdown problems ``(numbers, target)``.
2) The prover (P) tries to solve each problem.
3) Both are updated via RLOO each training step:
     - P's reward is the verifier score (``compute_score``).
     - C's reward is 1.0 iff P's pass-rate ``p_hat`` lands in a
       "difficulty band" (medium-hard problems), else 0.0.

Per training step, 9 phases (mirrors Nick's hot-swap pattern from rloo.py
but doubled):

    1) Sample C (worker hot-swap).
    2) Parse C's outputs to ``(numbers, target)``.
    3) Solvability filter via brute-force solver.
    4) Build P prompts for surviving problems.
    5) Sample P (worker hot-swap).
    6) Reward P (verifier).
    7) Compute C's reward (difficulty band).
    8) Update P (worker hot-swap; skipped if no valid problems).
    9) Update C (worker hot-swap).

One GPU worker is alive at a time; we ``ray.kill`` between phases.

Smoke test::

    # ``batch_size_p`` doesn't apply — number of P rollout problems is
    # ``len(valid_problems)`` which depends on N and the solvability rate.
    python joint_trainer/joint.py \\
        --num_problems_per_step=4 --group_size_p=4 --num_training_steps=2
"""

import os
import shutil
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import random
import ray
import wandb
from transformers import AutoTokenizer

# Make sibling packages importable when running ``python joint_trainer/joint.py``.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.countdown import compute_score
from joint_trainer.conjecturer_prompt import build_conjecturer_prompt, parse_problem
from joint_trainer.conjecturer_prompt_base import build_completion_prompt
from joint_trainer.solver import is_solvable
from rloo_trainer.rloo_update_worker import RLOOUpdateWorker
from rloo_trainer.sampling_worker import SamplingWorker


# Completion-mode prefill anchor — re-prepended before parsing C's outputs
# so the regex parser sees the full ``<problem> numbers=[...]`` pattern.
COMPLETION_PREFILL = "<problem> numbers=["


def build_prover_prompt(tokenizer, numbers, target):
    """Build a Countdown solving prompt in the format used by the RLOO dataset.

    Mirrors the prompt format from ``asingh15/countdown_tasks_3to4``: an SFT
    chat-templated user message asking to solve ``numbers -> target``.
    """
    nums_str = "[" + " ".join(str(int(n)) for n in numbers) + "]"
    user = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the "
        "answer.\n"
        f"User: Using the numbers {nums_str}, create an equation that equals "
        f"{int(target)}. You can use basic arithmetic operations (+, -, *, /) "
        "and each number can only be used once. Show your work in <think> "
        "</think> tags. And return the final answer in <answer> </answer> "
        "tags, for example <answer> (1 + 2) / 3 </answer>.\n"
        "Assistant: Let me solve this step by step."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


class JointTrainer:
    """Coordinates joint RLOO updates for conjecturer + prover policies."""

    def __init__(self, **kwargs):
        """Set attributes from kwargs. See ``_parse_args`` for the full list.

        Accepts ``**kwargs`` so the joint trainer config stays in lockstep
        with ``_parse_args``. Defaults live in ``_parse_args``; the small
        set below is what we need when constructing this class directly
        (e.g. from a notebook).
        """
        defaults = dict(
            c_model_name="Qwen/Qwen2.5-0.5B",
            p_model_name="asingh15/qwen-sft-countdown-defaultproj",
            c_prompt_style="completion",
            num_problems_per_step=8, group_size_p=4,
            difficulty_band_low=0.25, difficulty_band_high=0.75,
            c_learning_rate=1e-6,
            c_save_dir="checkpoints/joint_conjecturer",
            p_save_dir="checkpoints/joint_prover",
            ref_model_name=None,
            wandb_project="joint_rloo_default_project", wandb_name="test",
            lr_schedule="constant", learning_rate=1e-5,
            warmup_ratio=0.05, weight_decay=0.01,
            entropy_coefficient=0.01, kl_divergence_coefficient=0.0,
            num_epochs=1, num_training_steps=250,
            gradient_accumulation_steps=1, gradient_clipping=1.0,
            temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
            max_tokens=1024, max_model_len=2048,
            gpu_memory_utilization=0.9, max_num_batched_tokens=8192,
            enable_chunked_prefill=True, max_num_seqs=64,
            max_table_rows=20, save_every_n_steps=-1,
            # v8: shaped C reward, mixed data, length penalty, larger cap.
            c_kl_divergence_coefficient=0.02,
            shaped_c_reward=True,
            c_reward_peak=0.4,
            p_mix_fraction=0.5,
            p_length_penalty_alpha=0.1,
            fixed_dataset_name="asingh15/countdown_tasks_3to4",
            # Self-healing guards (v9):
            p_min_fixed_problems=0,   # floor of real problems P always trains on
            c_min_parseable_rate=0.0, # below this, roll C back (0 = disabled)
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)
        if self.ref_model_name is None:
            self.ref_model_name = self.p_model_name

        # Lengths used for tokenize_batch padding.
        self.max_prompt_length = 512
        self.max_response_length = self.max_tokens

        # Worker handles — only one of (sampling, update) is alive at a time.
        self.sampling_worker = None
        self.update_worker = None

        # Tokenizers — C and P may have different chat templates / vocabularies.
        self.c_tokenizer = AutoTokenizer.from_pretrained(self.c_model_name)
        self.p_tokenizer = AutoTokenizer.from_pretrained(self.p_model_name)

        # Pre-load the fixed-dataset prompts for the P-side mixing (v8).
        self._fixed_prompts = []
        self._fixed_gts = []
        if self.p_mix_fraction > 0:
            from datasets import load_dataset
            ds = load_dataset(self.fixed_dataset_name, split="train")
            self._fixed_prompts = list(ds["prompt"])
            self._fixed_gts = list(ds["ground_truth"])
            if not self._fixed_prompts:
                raise RuntimeError(
                    f"Fixed dataset '{self.fixed_dataset_name}' loaded empty; "
                    "set --p_mix_fraction=0 to disable mixing or fix the dataset."
                )
            print(f"[init] loaded {len(self._fixed_prompts)} fixed-dataset prompts for P mixing")

        self.wandb = wandb.init(project=self.wandb_project, name=self.wandb_name)
        self.wandb.config.update(vars(self))

    # ----- Worker lifecycle -----
    def _kill_all_workers(self):
        """Tear down whichever worker currently holds the GPU.

        Explicit ``tear_down()`` first to release GPU memory synchronously;
        ``ray.kill()`` alone is async and can leave the next worker OOM-ing
        because vLLM's KV cache (~70GB at gpu_memory_utilization=0.9) hasn't
        been released yet.
        """
        import time
        if self.sampling_worker is not None:
            try:
                ray.get(self.sampling_worker.tear_down.remote(), timeout=30)
            except Exception as e:
                print(f"Warning: sampling_worker.tear_down failed: {e}")
            ray.kill(self.sampling_worker)
            self.sampling_worker = None
        if self.update_worker is not None:
            try:
                ray.get(self.update_worker.tear_down.remote(), timeout=30)
            except Exception as e:
                print(f"Warning: update_worker.tear_down failed: {e}")
            ray.kill(self.update_worker)
            self.update_worker = None
        # Give Ray/CUDA time to fully release before next worker spawns.
        time.sleep(2)

    def _create_sampling_worker(
        self, model_path, group_size, stop_tokens, temperature,
        top_p, top_k, min_p=0.0, max_tokens=None,
    ):
        """Spin up a fresh SamplingWorker with role-specific sampling params."""
        self._kill_all_workers()
        mt = max_tokens if max_tokens is not None else self.max_tokens
        worker = SamplingWorker.remote(
            model_path=model_path,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_batched_tokens=self.max_num_batched_tokens,
            enable_chunked_prefill=self.enable_chunked_prefill,
            max_num_seqs=self.max_num_seqs,
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            max_tokens=mt, group_size=group_size,
            stop_tokens=stop_tokens,
        )
        ray.get(worker.load_checkpoint.remote())
        self.sampling_worker = worker
        return worker

    def _create_update_worker(
        self, model_path, optimizer_path, scheduler_path, group_size,
        learning_rate, ref_model_path, tokenizer_path,
        kl_divergence_coefficient=None,
        gradient_accumulation_steps=None,
    ):
        """Spin up a fresh RLOOUpdateWorker for either C or P."""
        self._kill_all_workers()
        kl_coef = (
            kl_divergence_coefficient
            if kl_divergence_coefficient is not None
            else self.kl_divergence_coefficient
        )
        ga = (
            gradient_accumulation_steps
            if gradient_accumulation_steps is not None
            else self.gradient_accumulation_steps
        )
        self.update_worker = RLOOUpdateWorker.remote(
            model_path=model_path, ref_model_path=ref_model_path,
            tokenizer_path=tokenizer_path,
            optimizer_path=optimizer_path, scheduler_path=scheduler_path,
            batch_size=max(group_size, 1),
            gradient_accumulation_steps=ga,
            gradient_clipping=self.gradient_clipping,
            group_size=group_size,
            entropy_coefficient=self.entropy_coefficient,
            kl_divergence_coefficient=kl_coef,
            lr_schedule=self.lr_schedule, learning_rate=learning_rate,
            weight_decay=self.weight_decay, warmup_ratio=self.warmup_ratio,
            num_training_steps=self.num_training_steps,
        )
        ray.get(self.update_worker.load_checkpoint.remote())
        return self.update_worker

    # ----- Tokenization (copied from rloo.py per design constraint) -----
    def tokenize_batch(self, batch, tokenizer, group_size):
        """Tokenize prompt/response rollouts into arrays for the update worker.

        Same logic as ``RLOOTrainer.tokenize_batch`` but parameterised by
        tokenizer and group_size so it can serve both C and P. Copied (not
        imported) to avoid cross-file coupling.
        """
        all_prompts_repeated = [p for p in batch["prompt"] for _ in range(group_size)]
        all_responses_flattened = [x for sub in batch["response"] for x in sub]
        all_rewards_flattened = [x for sub in batch["rewards"] for x in sub]
        all_sample_log_probs_flattened = [x for sub in batch["sample_log_probs"] for x in sub]
        n = len(all_prompts_repeated)
        assert (
            n == len(all_responses_flattened) == len(all_rewards_flattened)
            == len(all_sample_log_probs_flattened)
        ), f"length mismatch in tokenize_batch: {n=}"

        tokenizer.padding_side = "left"
        tp = tokenizer(all_prompts_repeated, add_special_tokens=False, padding=True,
                       truncation=True, max_length=self.max_prompt_length, return_tensors="np")
        tokenizer.padding_side = "right"
        tr = tokenizer(all_responses_flattened, add_special_tokens=False, padding=True,
                       truncation=True, max_length=self.max_response_length, return_tensors="np")
        p_ids, p_mask = tp["input_ids"], tp["attention_mask"]
        r_ids, r_mask = tr["input_ids"], tr["attention_mask"]
        is_response_token = np.concatenate(
            [np.zeros_like(p_ids), np.ones_like(r_ids)], axis=1
        )
        return {
            "input_ids": np.concatenate([p_ids, r_ids], axis=1),
            "attention_mask": np.concatenate([p_mask, r_mask], axis=1),
            "is_response_token": is_response_token,
            "rewards": np.array(all_rewards_flattened, dtype=np.float32),
            "sample_log_probs": np.array(all_sample_log_probs_flattened, dtype=np.float32),
        }

    # ----- Logging helpers -----
    def _build_table(self, prompts, responses, rewards):
        """Lightweight W&B table; mirrors rloo.py ``_build_generation_table``."""
        if self.max_table_rows <= 0:
            return None
        flat_rows = []
        for prompt, prompt_responses, prompt_rewards in zip(prompts, responses, rewards):
            for response, reward in zip(prompt_responses, prompt_rewards):
                flat_rows.append((prompt, response, float(np.array(reward).item())))
        if not flat_rows:
            return None
        random.shuffle(flat_rows)
        flat_rows = flat_rows[: self.max_table_rows]
        table = wandb.Table(columns=["prompt", "response", "reward"])
        for prompt, response, reward in flat_rows:
            table.add_data(prompt, response, reward)
        return table

    # ----- Checkpointing -----
    def _ckpt_paths(self, last_ckpt_dir, default_model_name):
        """Return (model_path, optimizer_path, scheduler_path) for loading."""
        if last_ckpt_dir is None:
            return default_model_name, None, None
        return (
            os.path.join(last_ckpt_dir, "model"),
            os.path.join(last_ckpt_dir, "optimizer.pt"),
            os.path.join(last_ckpt_dir, "scheduler.pt"),
        )

    def _run_update(self, tokenized):
        """Call ``update_gradient_accumulation`` on the active update worker."""
        return ray.get(
            self.update_worker.update_gradient_accumulation.remote(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                is_response_token=tokenized["is_response_token"],
                rewards=tokenized["rewards"],
                sample_log_probs=tokenized["sample_log_probs"],
            )
        )

    @staticmethod
    def _slot_of(path):
        """Return the ping-pong slot index (0/1) encoded in a C checkpoint dir."""
        if not path:
            return None
        base = os.path.basename(path.rstrip("/"))
        if base.endswith("_0"):
            return 0
        if base.endswith("_1"):
            return 1
        return None

    def _save_checkpoint(self, base_save_dir, epoch, global_step, sub_override=None):
        """Pick save dir, repoint update worker, and save. Returns save_dir.

        ``sub_override`` forces a specific sub-directory name (used by the C
        collapse-rollback ping-pong so the previous healthy checkpoint is never
        overwritten); otherwise the normal save_every / latest_checkpoint
        scheme is used.
        """
        if sub_override is not None:
            sub = sub_override
        elif self.save_every_n_steps > 0 and global_step % self.save_every_n_steps == 0:
            sub = f"epoch_{epoch}_step_{global_step}"
        else:
            sub = "latest_checkpoint"
        save_dir = os.path.join(base_save_dir, self.wandb_project, self.wandb_name, sub)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        ray.get(
            self.update_worker.update_checkpoint_paths.remote(
                model_path=os.path.join(save_dir, "model"),
                optimizer_path=os.path.join(save_dir, "optimizer.pt"),
                scheduler_path=os.path.join(save_dir, "scheduler.pt"),
                load_checkpoint=False,
            )
        )
        ray.get(self.update_worker.save_checkpoint.remote())
        return save_dir

    # ----- C-side sampling: prompt + parse -----
    def _c_prompt_and_stops(self):
        """Return (raw prompt string for vLLM, stop tokens, temperature)."""
        if self.c_prompt_style == "completion":
            return build_completion_prompt(), ["</problem>", "\n\n"], 0.7
        elif self.c_prompt_style == "chat":
            return build_conjecturer_prompt(self.c_tokenizer), ["</problem>"], 1.0
        else:
            raise ValueError(f"Unknown c_prompt_style: {self.c_prompt_style}")

    def _parse_c_outputs(self, raw_completions):
        """Parse C's vLLM outputs into ``(numbers, target)`` (or None per slot).

        In completion mode the prefill ``<problem> numbers=[`` is part of the
        PROMPT (already at the end of ``build_completion_prompt()``), NOT part
        of the model's generated tokens. We prepend it only for the regex
        parser so the ``<problem>...</problem>`` span matches. The "full
        responses" returned for training are the RAW completions (no prefill)
        so that HF's ``seq_logp`` is computed over the same tokens vLLM
        computed ``sample_log_probs`` over — keeping the importance weight
        valid.
        """
        problems, full_responses = [], []
        for c in raw_completions:
            if self.c_prompt_style == "completion":
                parse_text = COMPLETION_PREFILL + c
            else:
                parse_text = c
            problems.append(parse_problem(parse_text))
            full_responses.append(c)  # raw completion only, no prefill
        return problems, full_responses

    # ----- Main training loop -----
    def train(self):
        """Run online joint RLOO training for ``num_training_steps`` updates."""
        c_last_ckpt_dir = None
        c_prev_ckpt_dir = None  # last *healthy* C ckpt, for collapse rollback
        p_last_ckpt_dir = None
        global_step = 0

        for epoch in range(self.num_epochs):
            while global_step < self.num_training_steps:
                print("=" * 100)
                print(f"Joint step | Epoch {epoch} | Global Step {global_step}")
                print("=" * 100)

                # --- Phase 1: Sample C -----------------------------------
                c_mp, _, _ = self._ckpt_paths(c_last_ckpt_dir, self.c_model_name)
                c_prompt, c_stops, c_temp = self._c_prompt_and_stops()
                print(f"[1/9] Sampling C from {c_mp}")
                self._create_sampling_worker(
                    model_path=c_mp, group_size=self.num_problems_per_step,
                    stop_tokens=c_stops, temperature=c_temp, top_p=1.0, top_k=-1,
                )
                # vLLM ``n=num_problems_per_step`` → one prompt, N completions.
                c_resp_nested, c_lp_nested = ray.get(
                    self.sampling_worker.generate.remote([c_prompt])
                )
                c_raw_completions = c_resp_nested[0]
                c_logprobs = c_lp_nested[0]
                assert len(c_raw_completions) == self.num_problems_per_step

                # --- Phase 2: Parse C's outputs --------------------------
                parsed_problems, c_full_responses = self._parse_c_outputs(c_raw_completions)
                n_parseable = sum(p is not None for p in parsed_problems)
                parseable_rate = n_parseable / self.num_problems_per_step
                print(f"[2/9] Parseable: {n_parseable}/{self.num_problems_per_step}")

                # --- Phase 3: Solvability filter -------------------------
                # valid_mask[i] = True iff problem i is parseable AND solvable.
                valid_mask = [False] * self.num_problems_per_step
                for i, p in enumerate(parsed_problems):
                    if p is None:
                        continue
                    try:
                        valid_mask[i] = is_solvable(p[0], p[1])
                    except Exception:
                        valid_mask[i] = False
                n_valid = sum(valid_mask)
                valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
                valid_problems = [parsed_problems[i] for i in valid_indices]
                # v10: do NOT truncate n_valid here. The P update only requires the
                # *total* batch (n_valid + n_fixed) to be a multiple of grad_accum —
                # each problem is one whole RLOO group, so the per-problem grouping is
                # preserved regardless of how n_valid splits. The old "truncate n_valid
                # down to a multiple of ga" logic, combined with low C solvability,
                # drove n_valid -> 0 and starved C of any reward signal (all C reward
                # slots stayed 0 -> zero policy gradient). Padding to a multiple of ga
                # is now done in Phase 4 using fixed-dataset problems.
                ga = max(self.gradient_accumulation_steps, 1)
                print(f"[3/9] Valid (parseable & solvable): {n_valid}/{self.num_problems_per_step}")

                # --- Phase 4: Construct prover prompts -------------------
                # v8: prepend C-generated problems; append fixed-dataset
                # problems for distributional anchoring. C's reward (Phase 7)
                # is still computed ONLY from the C slice (indices 0..n_valid-1).
                import random
                if self.p_mix_fraction > 0 and self._fixed_prompts:
                    n_fixed = int(round(n_valid * self.p_mix_fraction / max(1e-9, 1.0 - self.p_mix_fraction)))
                    # Always keep P training on at least this many real problems so the
                    # method degrades gracefully to RLOO if C ever stalls. (v8 failure mode:
                    # when n_valid->0 the fixed count went to 0 too, halting P entirely.)
                    n_fixed = max(n_fixed, self.p_min_fixed_problems)
                    # v10: pad the *total* P batch up to a multiple of grad_accum using
                    # fixed problems, so every solvable C problem is kept (and thus gets
                    # a C reward) instead of being truncated away. The worker only needs
                    # (n_valid + n_fixed) % ga == 0; n_valid itself need not be divisible.
                    if ga > 1:
                        rem = (n_valid + n_fixed) % ga
                        if rem != 0:
                            n_fixed += ga - rem
                    n_fixed = min(n_fixed, len(self._fixed_prompts))
                else:
                    n_fixed = 0
                # Last-resort fallback: if there aren't enough fixed problems to pad the
                # total to a multiple of ga (e.g. p_mix_fraction=0), truncate n_valid so
                # the P microbatch still holds whole RLOO groups. With p_mix_fraction>0
                # and a large fixed set this never triggers.
                if ga > 1 and (n_valid + n_fixed) % ga != 0:
                    keep = (n_valid // ga) * ga
                    print(f"[4/9] WARN: cannot pad total to multiple of ga={ga}; "
                          f"truncating n_valid {n_valid} -> {keep}")
                    valid_indices = valid_indices[:keep]
                    valid_problems = valid_problems[:keep]
                    n_valid = keep
                if n_fixed > 0:
                    idxs = random.sample(range(len(self._fixed_prompts)), n_fixed)
                    fixed_prompts = [self._fixed_prompts[i] for i in idxs]
                    fixed_gts = [self._fixed_gts[i] for i in idxs]
                else:
                    fixed_prompts, fixed_gts = [], []
                print(f"[4/9] Building {n_valid} C prompts + {n_fixed} fixed-dataset prompts")
                p_prompts = [
                    build_prover_prompt(self.p_tokenizer, nums, t)
                    for (nums, t) in valid_problems
                ] + fixed_prompts
                p_gts = [
                    {"numbers": list(nums), "target": int(t)}
                    for (nums, t) in valid_problems
                ] + fixed_gts
                n_total_p = n_valid + n_fixed

                # --- Phase 5: Sample P (only if we have any problems) ----
                if n_total_p > 0:
                    p_mp, _, _ = self._ckpt_paths(p_last_ckpt_dir, self.p_model_name)
                    print(f"[5/9] Sampling P (group_size_p={self.group_size_p}) from {p_mp}")
                    self._create_sampling_worker(
                        model_path=p_mp, group_size=self.group_size_p,
                        stop_tokens=["</answer>"],
                        temperature=self.temperature, top_p=self.top_p,
                        top_k=self.top_k, min_p=self.min_p,
                    )
                    p_responses, p_sample_log_probs = ray.get(
                        self.sampling_worker.generate.remote(p_prompts)
                    )
                else:
                    print("[5/9] No problems → skipping P sampling")
                    p_responses, p_sample_log_probs = [], []

                # --- Phase 6: Reward P (verifier + length penalty, v8) ---
                # base reward in {0, 0.1, 1.0} from compute_score
                # length penalty: alpha * (resp_token_len / max_tokens), clamped so reward >= 0.
                alpha = self.p_length_penalty_alpha
                p_rewards = []
                for resp_list, gt in zip(p_responses, p_gts):
                    row = []
                    for r in resp_list:
                        base = compute_score(r, gt)
                        if alpha > 0:
                            tok_len = len(self.p_tokenizer(r, add_special_tokens=False)["input_ids"])
                            penalty = alpha * min(1.0, tok_len / max(1, self.max_tokens))
                            row.append(max(0.0, base - penalty))
                        else:
                            row.append(base)
                    p_rewards.append(row)
                flat_p = [r for sub in p_rewards for r in sub]
                p_reward_mean = float(np.mean(flat_p)) if flat_p else 0.0
                p_accuracy = float(np.mean([r >= 1.0 for r in flat_p])) if flat_p else 0.0
                print(f"[6/9] p_reward_mean={p_reward_mean:.4f} p_accuracy={p_accuracy:.4f}")

                # --- Phase 7: Compute C's reward (shaped, v8) -----------
                # Only the C slice of p_rewards informs C's reward.
                # Shaped: r_C = max(0, 1 - 2.5 * |p_hat - peak|) -> tent peaking at peak (default 0.4).
                # Solvability uses p_hat over correctness (>=1.0), ignoring the length penalty.
                c_rewards_per_slot = [0.0] * self.num_problems_per_step
                p_hats = []
                for valid_pos, slot_idx in enumerate(valid_indices):
                    rollouts = p_rewards[valid_pos]
                    # p_hat from rollout correctness (ignore length penalty for difficulty estimation)
                    correctness = [compute_score(r, p_gts[valid_pos]) >= 1.0 for r in p_responses[valid_pos]]
                    p_hat = float(np.mean(correctness)) if correctness else 0.0
                    p_hats.append(p_hat)
                    if self.shaped_c_reward:
                        c_rewards_per_slot[slot_idx] = max(0.0, 1.0 - 2.5 * abs(p_hat - self.c_reward_peak))
                    else:
                        in_band = self.difficulty_band_low <= p_hat <= self.difficulty_band_high
                        c_rewards_per_slot[slot_idx] = 1.0 if in_band else 0.0
                c_reward_mean = float(np.mean(c_rewards_per_slot))
                # Collapse diagnostics (computed over the valid C slice only):
                #   p_hat_mean   -> average prover solve-rate on C's problems.
                #   frac_too_easy-> fraction of C problems P already solves (p_hat>=0.95).
                #   frac_in_band -> fraction landing in the target difficulty window.
                # When the curriculum collapses (v7 failure mode) p_hat_mean -> 1.0,
                # frac_too_easy -> 1.0, and c_reward_mean -> 0. Watch these live and
                # early-stop / pick the best checkpoint before that happens.
                p_hat_mean = float(np.mean(p_hats)) if p_hats else 0.0
                frac_too_easy = float(np.mean([ph >= 0.95 for ph in p_hats])) if p_hats else 0.0
                frac_in_band = (
                    float(np.mean([self.difficulty_band_low <= ph <= self.difficulty_band_high for ph in p_hats]))
                    if p_hats else 0.0
                )
                print(f"[7/9] c_reward_mean={c_reward_mean:.4f} p_hat_mean={p_hat_mean:.3f} "
                      f"frac_too_easy={frac_too_easy:.2f} p_hats={p_hats}")

                # --- Phase 8: Update P (skip if no problems) -------------
                p_metrics = {}
                if n_total_p > 0:
                    print(f"[8/9] Updating P on {n_total_p} problems "
                          f"({n_valid} C + {n_fixed} fixed)")
                    p_tok = self.tokenize_batch(
                        {"prompt": p_prompts, "response": p_responses,
                         "rewards": p_rewards, "sample_log_probs": p_sample_log_probs},
                        self.p_tokenizer, self.group_size_p,
                    )
                    p_mp, p_op, p_sp = self._ckpt_paths(p_last_ckpt_dir, self.p_model_name)
                    self._create_update_worker(
                        model_path=p_mp, optimizer_path=p_op, scheduler_path=p_sp,
                        group_size=self.group_size_p, learning_rate=self.learning_rate,
                        ref_model_path=self.ref_model_name, tokenizer_path=self.p_model_name,
                    )
                    p_metrics = self._run_update(p_tok)
                    p_last_ckpt_dir = self._save_checkpoint(self.p_save_dir, epoch, global_step)
                else:
                    print("[8/9] No valid problems → skipping P update")

                # --- Phase 9: Update C (with collapse guard) -------------
                # Single conjecturer prompt × N generated problems treated as
                # one group of size N so the LOO baseline spans all outputs.
                #
                # Self-healing guard (the v8 fix): if C's parseable_rate falls
                # below c_min_parseable_rate, C has drifted off the SFT
                # distribution that the prefill trick depends on (v8 collapsed
                # to parseable_rate=0 by ~step 5 and never recovered). When that
                # happens we DISCARD the destabilising update and roll C back
                # two-deep to the last healthy checkpoint, then keep going. With
                # a low c_lr this lets C heal instead of dying. Detection lags
                # one step (parseable is measured before the update that damages
                # it), so we must revert to the checkpoint *before* the last
                # update, i.e. c_prev_ckpt_dir. Set c_min_parseable_rate=0 to
                # disable.
                print("[9/9] Updating C")
                c_group_size = self.num_problems_per_step
                if c_group_size < 2:
                    raise ValueError("num_problems_per_step must be >= 2 for RLOO baseline.")
                c_metrics = {}
                guard_triggered = parseable_rate < self.c_min_parseable_rate
                if not guard_triggered:
                    c_prev_ckpt_dir = c_last_ckpt_dir  # remember last healthy C
                    c_tok = self.tokenize_batch(
                        {"prompt": [c_prompt], "response": [c_full_responses],
                         "rewards": [c_rewards_per_slot], "sample_log_probs": [c_logprobs]},
                        self.c_tokenizer, c_group_size,
                    )
                    c_mp, c_op, c_sp = self._ckpt_paths(c_last_ckpt_dir, self.c_model_name)
                    self._create_update_worker(
                        model_path=c_mp, optimizer_path=c_op, scheduler_path=c_sp,
                        group_size=c_group_size, learning_rate=self.c_learning_rate,
                        ref_model_path=self.c_model_name, tokenizer_path=self.c_model_name,
                        kl_divergence_coefficient=self.c_kl_divergence_coefficient,
                        gradient_accumulation_steps=1,  # C's LOO baseline needs whole-batch
                    )
                    c_metrics = self._run_update(c_tok)
                    # Ping-pong C across two slots so the *previous* healthy
                    # checkpoint (c_prev_ckpt_dir) is never overwritten and stays
                    # available as a rollback target. Always write to the slot
                    # NOT currently held by c_last.
                    target_slot = 1 if self._slot_of(c_last_ckpt_dir) == 0 else 0
                    c_last_ckpt_dir = self._save_checkpoint(
                        self.c_save_dir, epoch, global_step,
                        sub_override=f"c_latest_{target_slot}",
                    )
                else:
                    print(f"[9/9] COLLAPSE GUARD: parseable_rate={parseable_rate:.2f} < "
                          f"{self.c_min_parseable_rate:.2f} -> discarding update, rolling C back "
                          f"to last healthy checkpoint: {c_prev_ckpt_dir}")
                    c_last_ckpt_dir = c_prev_ckpt_dir  # revert; discard damaged C

                # --- Logging ---------------------------------------------
                print(f"== step {global_step} done | parseable={n_parseable} "
                      f"valid={n_valid} p_rew={p_reward_mean:.3f} c_rew={c_reward_mean:.3f}")
                c_table = self._build_table(
                    [c_prompt] * len(c_full_responses),
                    [[r] for r in c_full_responses],
                    [[r] for r in c_rewards_per_slot],
                )
                p_table = (
                    self._build_table(p_prompts, p_responses, p_rewards)
                    if n_total_p > 0 else None
                )
                log_dict = {
                    "train/epoch": epoch, "train/global_step": global_step,
                    "c/parseable_rate": n_parseable / self.num_problems_per_step,
                    "c/valid_rate": n_valid / self.num_problems_per_step,
                    "c/reward_mean": c_reward_mean,
                    "c/p_hat_mean": p_hat_mean,
                    "c/frac_too_easy": frac_too_easy,
                    "c/frac_in_band": frac_in_band,
                    "c/collapse_guard_triggered": 1.0 if guard_triggered else 0.0,
                    "p/reward_mean": p_reward_mean, "p/accuracy": p_accuracy,
                    "p/num_c_problems": n_valid,
                    "p/num_fixed_problems": n_fixed,
                    "p/num_total_problems": n_total_p,
                }
                log_dict.update({f"c/{k}": v for k, v in c_metrics.items()})
                log_dict.update({f"p/{k}": v for k, v in p_metrics.items()})
                if c_table is not None:
                    log_dict["samples/c_generations"] = c_table
                if p_table is not None:
                    log_dict["samples/p_generations"] = p_table
                self.wandb.log(log_dict, step=global_step)

                global_step += 1
                if global_step >= self.num_training_steps:
                    break
            if global_step >= self.num_training_steps:
                break

        self._kill_all_workers()
        ray.shutdown()
        self.wandb.finish()


def _parse_args():
    parser = ArgumentParser()
    # (flag, type, default) — single source of truth for CLI defaults.
    typed_args = [
        # joint-specific
        ("--c_model_name", str, "asingh15/qwen-sft-countdown-defaultproj"),
        ("--p_model_name", str, "asingh15/qwen-sft-countdown-defaultproj"),
        ("--num_problems_per_step", int, 8),
        ("--group_size_p", int, 4),
        ("--difficulty_band_low", float, 0.05),
        ("--difficulty_band_high", float, 0.85),
        ("--c_kl_divergence_coefficient", float, 0.02),
        ("--c_learning_rate", float, 1e-6),
        ("--c_save_dir", str, "/vol/checkpoints/joint_conjecturer"),
        ("--p_save_dir", str, "/vol/checkpoints/joint_prover"),
        # shared / P-side RLOO (mirror rloo.py)
        ("--ref_model_name", str, None),
        ("--wandb_project", str, "joint_rloo_default_project"),
        ("--wandb_name", str, "test"),
        ("--lr_schedule", str, "constant"),
        ("--learning_rate", float, 1e-5),
        ("--warmup_ratio", float, 0.0),
        ("--weight_decay", float, 0.01),
        ("--entropy_coefficient", float, 0.01),
        ("--kl_divergence_coefficient", float, 0.0),
        ("--num_epochs", int, 1),
        ("--num_training_steps", int, 250),
        ("--gradient_accumulation_steps", int, 1),
        ("--gradient_clipping", float, 1.0),
        ("--temperature", float, 1.0),
        ("--top_p", float, 1.0),
        ("--top_k", int, -1),
        ("--min_p", float, 0.0),
        ("--max_tokens", int, 1024),
        ("--max_model_len", int, 2048),
        ("--gpu_memory_utilization", float, 0.9),
        ("--max_num_batched_tokens", int, 8192),
        ("--max_num_seqs", int, 64),
        ("--max_table_rows", int, 20),
        ("--save_every_n_steps", int, -1),
        # v8: reward shaping, mixed data, length penalty
        ("--c_reward_peak", float, 0.4),
        ("--p_mix_fraction", float, 0.5),
        ("--p_length_penalty_alpha", float, 0.1),
        ("--fixed_dataset_name", str, "asingh15/countdown_tasks_3to4"),
        # self-healing guards (v9)
        ("--p_min_fixed_problems", int, 0),
        ("--c_min_parseable_rate", float, 0.0),
    ]
    for flag, ty, default in typed_args:
        parser.add_argument(flag, type=ty, default=default)
    parser.add_argument(
        "--c_prompt_style", choices=["chat", "completion"], default="completion"
    )
    parser.add_argument("--shaped_c_reward", action="store_true", default=True)
    parser.add_argument("--no_shaped_c_reward", dest="shaped_c_reward", action="store_false")
    parser.add_argument("--enable_chunked_prefill", action="store_true")
    parser.add_argument("--disable_chunked_prefill", action="store_true")
    args = parser.parse_args()
    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        raise ValueError(
            "Cannot set both --enable_chunked_prefill and --disable_chunked_prefill."
        )
    if args.enable_chunked_prefill:
        args.enable_chunked_prefill = True
    elif args.disable_chunked_prefill:
        args.enable_chunked_prefill = False
    else:
        args.enable_chunked_prefill = True
    del args.disable_chunked_prefill
    return args


if __name__ == "__main__":
    args = _parse_args()
    ray.init()
    trainer = JointTrainer(**vars(args))
    trainer.train()
