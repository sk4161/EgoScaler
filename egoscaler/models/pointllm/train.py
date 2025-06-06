import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import argparse
from builder import build_model
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from dataset import CustomDataset
import re 
import numpy as np
import math
import deepspeed
import wandb
from egoscaler.models.utils.metrics import (
    initial_displacement_error, final_displacement_error, average_displacement_error, 
    dynamic_time_warping, anglar_distance
)
from egoscaler.models.utils.utils import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def to_device(item: dict, device: str) -> dict:
    """
    Move a dictionary of tensors to the specified device.
    """
    return {key: value.to(device) for key, value in item.items()}

def get_other_parameters(model, submodule_name):
    """
    Yield parameters of the model that do not belong to the specified submodule.
    """
    for name, param in model.named_parameters():
        if submodule_name not in name and param.requires_grad:
            yield param

def main(args):
    """
    Main function for training the model.
    """
    set_seed(42)

    deepspeed.init_distributed()
    if torch.distributed.get_rank() == 0: 
        print(args)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()
    
    # Configure WandB run name based on arguments
    wandb_run_name = f"pointllm-do_norm" if args.do_norm else "pointllm-no_norm"
    wandb_run_name += "-do_standard" if args.do_standard else "-no_standard"
    
    print(f"WandB run name: {wandb_run_name}")    
    if torch.distributed.get_rank() == 0:  # Output only for rank 0
        wandb.init(
            project=f"egoscaler_trajectory_generator",  # Replace with your WandB project name
            config=vars(args),
            name=wandb_run_name
        )
    else:
        wandb.init(mode="disabled")
    
    # Build model and tokenizer
    model, tokenizer, point_backbone_config, mm_use_point_start_end = build_model(args)

    # Prepare datasets and dataloaders
    train_dataset = CustomDataset(args, wandb.run.dir, 'train', tokenizer)
    val_dataset = CustomDataset(args, wandb.run.dir, 'val', tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.bs, 
        num_workers=6, pin_memory=True, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.bs, 
        num_workers=6, pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    if torch.distributed.get_rank() == 0:  # Output only for rank 0
        print(f"Batch size: {args.grad_accum_steps * args.bs}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_trainable_params}")
        
    # Configure DeepSpeed
    deepspeed_config = {
        "train_batch_size": args.bs,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "train_micro_batch_size_per_gpu": math.ceil(args.bs / args.grad_accum_steps / num_gpus),
        "steps_per_print": 100,
        "fp16": {"enabled": False},  # Disable FP16
        "bf16": {"enabled": True},  # Enable BF16
        "zero_optimization": {
            "stage": 1,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8
        },
    }
    
    # Configure optimizer and scheduler
    optimizer_parameters = [
        {
            'params': [param for param in model.parameters() if param.requires_grad],
            'lr': args.lr_llm
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.epochs * len(train_dataloader) / 5),
        num_training_steps=args.epochs * len(train_dataloader)
    )
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=optimizer_parameters,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=deepspeed_config
    )

    # Initialize training variables
    start_epoch = 0
    global_step = 0 
    best_val_loss = float('inf')
    best_ade = float('inf')
    best_checkpoint_paths = {
        "val_loss": os.path.join(wandb.run.dir, "best_model_val_loss.pt"),
        "ade": os.path.join(wandb.run.dir, "best_model_ade.pt")
    }
    latest_checkpoint_path = os.path.join(wandb.run.dir, "latest_model.pt")
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint.get('global_step', 0)
            print(f"Resumed from checkpoint '{args.resume}' at epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")
            sys.exit(1)
        
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        progress = tqdm(total=len(train_dataloader), desc=f'[train] epoch {epoch}', mininterval=30.0)
        for i, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            optimizer.zero_grad()
            
            pcrgbs = batch['pcrgbs']
            tokens = batch['tokens']
            attention_masks = batch['attention_masks']
            prompts = batch['prompts']

            with autocast(dtype=torch.bfloat16):
                outputs = model_engine(
                    input_ids=tokens,
                    attention_mask=attention_masks,
                    point_clouds=pcrgbs,
                    return_dict=True
                )
                
            logits = outputs.logits[:, prompts.shape[1]-1:-1, :]
            tokens = tokens[:, prompts.shape[1]:]
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), 
                tokens.flatten(), 
                ignore_index=tokenizer.pad_token_id
            )
            
            model_engine.backward(loss)
            model_engine.step()
            
            train_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            
            wandb.log({
                "epoch": epoch,
                "step": global_step,
                "learning_rate": current_lr
            })
            
            global_step += 1                        
            progress.update()
        progress.close()
        
        # Validation loop
        val_loss = 0
        ade_scores = []
        fde_scores = []
        dtws = []
        gds = []
        model.eval()
        progress = tqdm(total=len(val_dataloader), desc=f'[eval] epoch {epoch}', mininterval=30.0)
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                batch = to_device(batch, device)
                optimizer.zero_grad()
                
                pcrgbs = batch['pcrgbs']
                tokens = batch['tokens']
                attention_masks = batch['attention_masks']
                prompts = batch['prompts']
                prompt_masks = batch['prompt_masks']
                gt_trajs = batch['trajectories']
                gt_traj_masks = batch['trajectory_masks']
                max_abs = batch['max_abs']
                
                max_new_len = tokens.shape[1] - prompts.shape[1]
                with autocast(dtype=torch.bfloat16):
                    outputs = model_engine.module.generate(
                        input_ids=prompts,
                        attention_mask=prompt_masks,
                        point_clouds=pcrgbs,
                        max_length=max_new_len
                    )
                    
                    gen_tokens = outputs.sequences[:, prompts.shape[1]:]
                    logits = torch.stack(outputs.scores).transpose(0, 1)
                
                tokens = tokens.detach().cpu()
                gen_tokens = gen_tokens.detach().cpu()
                max_abs = max_abs.detach().cpu().numpy()

                if i % 100 == 0 and torch.distributed.get_rank() == 0:
                    print(tokenizer.decode(gen_tokens[0]))
                    
                for gen_token, gt_traj, gt_traj_mask, m_abs in zip(gen_tokens, gt_trajs, gt_traj_masks, max_abs):
                    gen_token = gen_token.tolist()
                    if tokenizer.eos_token_id in gen_token:
                        gen_token = gen_token[:gen_token.index(tokenizer.eos_token_id)]
                    gen_traj: str = tokenizer.decode(gen_token, skip_special_tokens=True)
                    
                    gen_traj: np.ndarray = train_dataset.detokenize_traj(gen_traj, num_bins=args.num_bins, max_abs=m_abs)
                    gt_traj: np.ndarray = gt_traj.detach().cpu().numpy()
                    
                    if gen_traj is None:
                        continue

                    if gen_traj.shape[0] < gt_traj.shape[0]:
                        gap = gt_traj.shape[0] - gen_traj.shape[0]
                        last_frame = gen_traj[-1][None, :] 
                        repeated = np.repeat(last_frame, repeats=gap, axis=0)
                        gen_traj = np.concatenate([gen_traj, repeated], axis=0)

                    ade: float = average_displacement_error(gen_traj[None,:,:], gt_traj[None,:,:])
                    fde: float = final_displacement_error(gen_traj[None,:,:], gt_traj[None,:,:])
                    gd: float = anglar_distance(gen_traj[None, :, :], gt_traj[None, : ,:])
                    
                    ade_scores.append(ade)
                    fde_scores.append(fde)
                    gds.append(gd)

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader) 
        avg_ade = np.mean(ade_scores)
        avg_fde = np.mean(fde_scores)
        avg_gd = np.mean(gds)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "ADE": avg_ade,
            "FDE": avg_fde,
            "GD": avg_gd
        })
        
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"ADE: {avg_ade}")
        print(f"FDE: {avg_fde}")
        print(f"GD: {avg_gd}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_engine.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step
        }
        
        torch.save(checkpoint, latest_checkpoint_path)
        
        if avg_ade < best_ade:
            best_ade = avg_ade
            print(f"New best ADE: {best_ade}. Saving best checkpoint (ADE).")
            best_checkpoint_ade = {
                'epoch': epoch,
                'model_state_dict': model_engine.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'best_ade': best_ade
            }
            torch.save(best_checkpoint_ade, best_checkpoint_paths["ade"])
    
    wandb.finish()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/data/g-liat/yoshida/EgoScaler')
    parser.add_argument('--data_dir', default='/home/yoshida/EgoScaler/egoscaler/models/data')
    
    parser.add_argument('--task', default="desc2traj", help=["desc2traj", "traj2desc", "mix"])    
    parser.add_argument('--model_config', default="pointllm")
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--model_name', default='RunsenXu/PointLLM_7B_v1.2')
    
    parser.add_argument('--max_traj_token', type=int, default=160)
    parser.add_argument('--max_desc_token', type=int, default=20)
    
    parser.add_argument('--num_steps', type=int, default=20)
    
    parser.add_argument('--do_norm', action='store_true')
    parser.add_argument('--do_standard', action='store_true')
    
    parser.add_argument('--unfreeze_pc_encoder', action="store_true")
    parser.add_argument('--unfreeze_language_model', action="store_true")
    
    parser.add_argument('--warmup_steps', default=5000)
    parser.add_argument('--eval_results', default=True)
    parser.add_argument('--eval_first', default=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--lr_llm', default=2e-5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_every', action='store_true')
    
    parser.add_argument('--num_bins', default=256)
    
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    main(args)