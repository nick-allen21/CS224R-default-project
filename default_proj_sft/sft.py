"""Starter SFT training entrypoint for the class project.

This file is intentionally incomplete. Students are expected to implement
`train(...)` while reusing the data/model setup provided here.
"""

import sys
from pathlib import Path

# Allow `python sft_trainer/sft.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from sft_trainer.sft_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device='cuda', use_gradient_checkpointing=True):
    """Load policy model + tokenizer for SFT training."""
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
    return model, tokenizer

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

def train(
    model,
    tokenizer,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    num_epochs,
    device='cuda',
    save_model=1,
    output_dir='sft_model',
    gradient_accumulation_steps=1,
    gradient_clipping=1.0
):
    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        num_updates = 0

        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            response_mask = batch["is_response_token"][:, 1:].to(device).float()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # run the model forward, for each token position, output a vector with the predicted probablity for each possible token; size: batch_size, sequence_length, vocab_size
            pred_logits = logits[:, :-1, :].contiguous()  # model's prediction for what comes after token i
            labels = input_ids[:, 1:].contiguous()  # actual token after token i

            token_loss = F.cross_entropy(
                pred_logits.view(-1, pred_logits.size(-1)),
                labels.view(-1),
                reduction="none",
            ).view(input_ids.size(0), -1)

            denom = response_mask.sum().clamp(min=1.0)
            loss = (token_loss * response_mask).sum() / denom

            with torch.no_grad():
                preds = pred_logits.argmax(dim=-1)
                acc = ((preds == labels).float() * response_mask).sum() / denom

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
                train_acc_sum += acc.item()

                wandb.log({
                    "train/loss": loss.item(),
                    "train/token_accuracy": acc.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                    "epoch": epoch,
                })

        if num_updates:
            print(
                f"train loss: {train_loss_sum / num_updates:.4f} | "
                f"train token acc: {train_acc_sum / num_updates:.4f}"
            )

        model.eval()
        eval_losses = []
        eval_accs = []

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                response_mask = batch["is_response_token"][:, 1:].to(device).float()

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                pred_logits = logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous()

                token_loss = F.cross_entropy(
                    pred_logits.view(-1, pred_logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                ).view(input_ids.size(0), -1)

                denom = response_mask.sum().clamp(min=1.0)
                loss = (token_loss * response_mask).sum() / denom

                preds = pred_logits.argmax(dim=-1)
                acc = ((preds == labels).float() * response_mask).sum() / denom

                eval_losses.append(loss.item())
                eval_accs.append(acc.item())

        avg_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0.0
        avg_eval_acc = sum(eval_accs) / len(eval_accs) if eval_accs else 0.0

        print(f"eval loss: {avg_eval_loss:.4f} | eval token acc: {avg_eval_acc:.4f}")
        wandb.log({
            "test/loss": avg_eval_loss,
            "test/token_accuracy": avg_eval_acc,
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
    parser.add_argument('--dataset_name', type=str, default='Asap7772/cog_behav_all_strategies')
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
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

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
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping
    )

if __name__ == "__main__":
    main()
