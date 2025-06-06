import os
import torch
from tqdm import tqdm
from builder import build_model
from torch.utils.data import DataLoader
from dataset import CustomDataset
from egoscaler.models.utils.utils import set_seed
import deepspeed
import json
import numpy as np
from egoscaler.models.utils.metrics import (
    initial_displacement_error, final_displacement_error, average_displacement_error, 
    dynamic_time_warping, anglar_distance
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def to_device(item: dict, device: str) -> dict:
    """
    Move a dictionary of tensors to the specified device.

    Parameters:
    - item (dict): Dictionary containing tensors.
    - device (str): Target device ('cuda' or 'cpu').

    Returns:
    - dict: Dictionary with tensors moved to the target device.
    """
    return {key: value.to(device) for key, value in item.items()}

def evaluate(args, split: str):
    """
    Evaluate the model on the specified dataset split.

    Parameters:
    - args: Arguments containing evaluation configurations.
    - split (str): Dataset split (e.g., 'val', 'test').
    """
    set_seed(42)
    deepspeed.init_distributed()

    if torch.distributed.get_rank() == 0:
        print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build model and tokenizer
    model, tokenizer, _, _ = build_model(args)

    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model_ade.pt")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize DeepSpeed inference engine
    model_engine = deepspeed.init_inference(
        model=model,
        tensor_parallel={"tp_size": 1},
        dtype=torch.bfloat16,
    )

    # Prepare dataset and dataloader
    dataset = CustomDataset(args, args.checkpoint_dir, split, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.bs,
        num_workers=6, pin_memory=True,
        collate_fn=dataset.collate_fn
    )

def evaluate(args, split):
    set_seed(42)
    deepspeed.init_distributed()
    if torch.distributed.get_rank() == 0: 
        print(args)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()
    
    model, tokenizer, point_backbone_config, mm_use_point_start_end = build_model(args)
    
    checkpoint = torch.load(f"{args.checkpoint_dir}/best_model_ade.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model_engine = deepspeed.init_inference(
        model=model,
        tensor_parallel={"tp_size": 1},
        dtype=torch.bfloat16,
    )
    
    dataset = CustomDataset(args, args.checkpoint_dir, split, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.bs,
        num_workers=6, pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    
    # Evaluation loop
    model_engine.eval()
    progress = tqdm(total=len(dataloader), desc=f"[eval] {split}", mininterval=30.0)
    generated_trajs = {}
    ade_scores = []
    fde_scores = []
    gds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, device)

            pcrgbs = batch['pcrgbs']
            prompts = batch['prompts']
            prompt_masks = batch['prompt_masks']
            gt_trajs = batch['trajectories']
            max_abs = batch['max_abs']

            max_new_len = batch['tokens'].shape[1] - prompts.shape[1]

            outputs = model_engine.module.generate(
                input_ids=prompts,
                attention_mask=prompt_masks,
                point_clouds=pcrgbs,
                max_length=max_new_len
            )
            
            gen_tokens = outputs.sequences[:, prompts.shape[1]:]
            gen_tokens = gen_tokens.detach().cpu()
            max_abs = max_abs.detach().cpu().numpy()

            for image_id, gen_token, gt_traj, m_abs in zip(batch['image_ids'], gen_tokens, gt_trajs, max_abs):
                gen_token = gen_token.tolist()
                if tokenizer.eos_token_id in gen_token:
                    gen_token = gen_token[:gen_token.index(tokenizer.eos_token_id)]
                gen_traj = tokenizer.decode(gen_token, skip_special_tokens=True)
                gen_traj = dataset.detokenize_traj(gen_traj, num_bins=args.num_bins, max_abs=m_abs)
                gt_traj = gt_traj.detach().cpu().numpy()

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

                generated_trajs[image_id.item()] = gen_traj.tolist()

            progress.update()

    progress.close()

    # Calculate averages
    avg_ade = np.mean(ade_scores)
    avg_fde = np.mean(fde_scores)
    avg_gd = np.mean(gds)

    print(f"ADE: {avg_ade}")
    print(f"FDE: {avg_fde}")
    print(f"GD: {avg_gd}")

    # Save generated trajectories
    output_path = os.path.join(args.checkpoint_dir, f"{split}_gen_trajs.json")
    with open(output_path, 'w') as f:
        json.dump(generated_trajs, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/data/g-liat/yoshida/EgoScaler')
    parser.add_argument('--data_dir', default='/home/yoshida/EgoScaler/egoscaler/models/data')
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--task', default="desc2traj", help=["desc2traj", "traj2desc", "mix"])
    parser.add_argument('--model_config', default="pointllm")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--model_name', default='RunsenXu/PointLLM_7B_v1.2')
    parser.add_argument('--max_traj_token', type=int, default=160)
    parser.add_argument('--max_desc_token', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--num_bins', default=256)

    args = parser.parse_args()
    evaluate(args, "test")