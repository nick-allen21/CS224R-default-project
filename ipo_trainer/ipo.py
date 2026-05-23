"""Starter IPO training entrypoint for the class project.

This script wires model loading, data loading, and optimizer setup.
Students are expected to implement `train(...)` for the IPO objective.
"""

import sys
from pathlib import Path

# Allow `python ipo_trainer/ipo.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from ipo_trainer.ipo_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
import copy
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device, use_gradient_checkpointing=True):
    """Load trainable policy model and frozen reference model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enable gradient checkpointing to reduce memory (trades compute for memory)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.train()

    # IPO compares policy preferences to a fixed baseline policy.
    reference_model = copy.deepcopy(model)
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    return model, tokenizer, reference_model

def clear_cache(model):
    """Best-effort GPU/CPU cache cleanup between heavy steps."""
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir):
    """Save model/tokenizer plus optimizer/scheduler states."""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

    torch.save({
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(output_dir, 'train_states.pth'))
    print(f"Model saved to {output_dir}")

def sequence_logps(model, input_ids, attention_mask, response_mask, average_logps):
    """Sum of per-token log-probs over response tokens, returns [batch]."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    pred_logits = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()

    token_logp = -F.cross_entropy(
        pred_logits.view(-1, pred_logits.size(-1)),
        labels.view(-1),
        reduction="none",
    ).view(input_ids.size(0), -1)

    masked = token_logp * response_mask
    if average_logps:
        return masked.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1.0)
    return masked.sum(dim=-1)

def batch_logps(model, batch, device, average_logps):
    """Compute chosen orrejected sequence log-probs for one batch under a model."""
    input_ids_w = batch["input_ids_w"].to(device)
    attention_mask_w = batch["attention_mask_w"].to(device)
    response_mask_w = batch["is_response_token_w"][:, 1:].to(device).float()
    input_ids_l = batch["input_ids_l"].to(device)
    attention_mask_l = batch["attention_mask_l"].to(device)
    response_mask_l = batch["is_response_token_l"][:, 1:].to(device).float()

    chosen_logps = sequence_logps(model, input_ids_w, attention_mask_w, response_mask_w, average_logps)
    rejected_logps = sequence_logps(model, input_ids_l, attention_mask_l, response_mask_l, average_logps)
    return chosen_logps, rejected_logps

def preference_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, loss_type):
    """IPO or DPO loss plus reward margin and accuracy, all from four sequence log-probs."""
    chosen_logratio = policy_chosen - ref_chosen
    rejected_logratio = policy_rejected - ref_rejected
    h = chosen_logratio - rejected_logratio

    if loss_type == 'ipo':
        loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
    elif loss_type == 'dpo':
        loss = -F.logsigmoid(beta * h).mean()
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")

    reward_margin = (beta * h).mean()
    reward_acc = (chosen_logratio > rejected_logratio).float().mean()
    return loss, reward_margin, reward_acc

def train(
    model, 
    tokenizer, 
    reference_model,
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0,
    beta=0.1,
    average_logps=False,
    loss_type='ipo',
):
    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")
        train_loss_sum = 0.0
        train_margin_sum = 0.0
        train_acc_sum = 0.0
        num_updates = 0

        for i, batch in enumerate(train_dataloader):
            policy_chosen, policy_rejected = batch_logps(model, batch, device, average_logps)
            with torch.no_grad():
                ref_chosen, ref_rejected = batch_logps(reference_model, batch, device, average_logps)

            loss, reward_margin, reward_acc = preference_loss(
                policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, loss_type
            )

            (loss / gradient_accumulation_steps).backward()

            should_step = (i + 1) % gradient_accumulation_steps == 0
            is_last_batch = (i + 1) == len(train_dataloader)

            if should_step or is_last_batch:
                if gradient_clipping and gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                num_updates += 1
                train_loss_sum += loss.item()
                train_margin_sum += reward_margin.item()
                train_acc_sum += reward_acc.item()

                wandb.log({
                    "train/loss": loss.item(),
                    "train/reward_margin": reward_margin.item(),
                    "train/reward_accuracy": reward_acc.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                    "epoch": epoch,
                })

        if num_updates:
            print(
                f"train loss: {train_loss_sum / num_updates:.4f} | "
                f"train reward margin: {train_margin_sum / num_updates:.4f} | "
                f"train reward acc: {train_acc_sum / num_updates:.4f}"
            )

        model.eval()
        eval_losses = []
        eval_margins = []
        eval_accs = []

        with torch.no_grad():
            for batch in test_dataloader:
                policy_chosen, policy_rejected = batch_logps(model, batch, device, average_logps)
                ref_chosen, ref_rejected = batch_logps(reference_model, batch, device, average_logps)
                loss, reward_margin, reward_acc = preference_loss(
                    policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, loss_type
                )
                eval_losses.append(loss.item())
                eval_margins.append(reward_margin.item())
                eval_accs.append(reward_acc.item())

        avg_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0.0
        avg_eval_margin = sum(eval_margins) / len(eval_margins) if eval_margins else 0.0
        avg_eval_acc = sum(eval_accs) / len(eval_accs) if eval_accs else 0.0

        print(f"eval loss: {avg_eval_loss:.4f} | eval reward margin: {avg_eval_margin:.4f} | eval reward acc: {avg_eval_acc:.4f}")
        wandb.log({
            "test/loss": avg_eval_loss,
            "test/reward_margin": avg_eval_margin,
            "test/reward_accuracy": avg_eval_acc,
            "epoch": epoch
        })

        if save_model:
            ckpt_dir = os.path.join(output_dir, f"epoch_{epoch}")
            save_checkpoint(model, tokenizer, optimizer, scheduler, ckpt_dir)

        model.train()
        clear_cache(model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='asingh15/countdown_tasks_3to4-dpo')
    parser.add_argument('--output_dir', type=str, default='sft_model')
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_response_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='sft_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--average_logps', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='dpo')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer, reference_model = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

    dataloaders = get_dataloaders(
        dataset_name=args.dataset_name, 
        tokenizer=tokenizer, 
        max_prompt_length=args.max_prompt_length, 
        max_response_length=args.max_response_length, 
        batch_size=args.batch_size, 
        splits=['train', 'test'],
        pin_memory=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dataloader, test_dataloader = dataloaders['train'], dataloaders['test']
    # Scheduler steps happen only after an optimizer step, so account for
    # gradient accumulation when estimating total training steps.
    num_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_name)
    os.makedirs(full_output_dir, exist_ok=True)

    train(
        model, 
        tokenizer, 
        reference_model,
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping,
        args.beta,
        args.average_logps,
        args.loss_type
    )

if __name__ == "__main__":
    main()
