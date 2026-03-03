"""Merge distributed checkpoints into a single model file.

Usage:
    python -m asteroid.ft.merge_checkpoints --checkpoint-dir ./my_checkpoints --iteration 500
    python -m asteroid.ft.merge_checkpoints --checkpoint-dir ./my_checkpoints --iteration 500 --output model.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)


def discover_checkpoints(checkpoint_dir: Path, iteration: int) -> Dict[int, Path]:
    """Find all rank checkpoints for a given iteration."""
    pattern = f"ckpt_*_i{iteration}_r*.pt"
    checkpoints: Dict[int, Path] = {}
    
    # Check both flat structure and rank-N subdirectories
    for ckpt_path in checkpoint_dir.rglob(pattern):
        filename = ckpt_path.name
        # Parse rank from filename: ckpt_e0_i100_r0.pt -> rank 0
        try:
            rank = int(filename.split("_r")[-1].replace(".pt", ""))
            checkpoints[rank] = ckpt_path
        except (ValueError, IndexError):
            logger.warning(f"Could not parse rank from {filename}")
            continue
    
    return dict(sorted(checkpoints.items()))


def merge_checkpoints(
    checkpoint_dir: str | Path,
    iteration: int,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Merge checkpoints from all ranks into a single state dict.
    
    Args:
        checkpoint_dir: Directory containing rank-N subdirectories or flat checkpoints
        iteration: Training iteration to merge (e.g., 100, 200, 500)
        output_path: Optional path to save merged model
        
    Returns:
        Merged state dict with full model weights
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = discover_checkpoints(checkpoint_dir, iteration)
    
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found for iteration {iteration} in {checkpoint_dir}")
    
    print(f"Found {len(checkpoints)} rank checkpoint(s) for iteration {iteration}:")
    for rank, path in checkpoints.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  rank-{rank}: {path.name} ({size_mb:.1f} MB)")
    
    # Load and merge
    merged_state: Dict[str, Any] = OrderedDict()
    merged_model_state: Dict[str, torch.Tensor] = OrderedDict()
    merged_optimizer_state: Dict[str, Any] = {}
    
    config = None
    total_layers = 0
    
    for rank, ckpt_path in checkpoints.items():
        print(f"\nLoading rank-{rank}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # Extract metadata from first checkpoint
        if config is None:
            config = ckpt.get("config", {})
            merged_state["epoch"] = ckpt.get("epoch", 0)
            merged_state["iteration"] = ckpt.get("iteration", iteration)
            merged_state["config"] = config
        
        # Get stage info
        start_layer = ckpt.get("start_layer", 0)
        end_layer = ckpt.get("end_layer", 0)
        is_first = ckpt.get("is_first", rank == 0)
        is_last = ckpt.get("is_last", rank == len(checkpoints) - 1)
        
        print(f"  Stage info: layers {start_layer}-{end_layer}, "
              f"first={is_first}, last={is_last}")
        
        # Merge model state dict (handle both old 'model' and new 'model_state_dict' keys)
        model_state = ckpt.get("model_state_dict") or ckpt.get("model", {})
        
        # If no start_layer info, try to estimate from rank and total ranks
        if start_layer == 0 and end_layer == 0 and len(checkpoints) > 1:
            # Old checkpoint without stage info - estimate layer boundaries
            # Count blocks in this stage's state dict
            block_indices = set()
            for key in model_state.keys():
                if key.startswith("blocks."):
                    idx = int(key.split(".")[1])
                    block_indices.add(idx)
            num_blocks = len(block_indices)
            # Estimate global indices based on rank position
            start_layer = rank * num_blocks  # rough estimate
            end_layer = start_layer + num_blocks
            print(f"  (Estimated from state dict: {num_blocks} blocks, global layers {start_layer}-{end_layer})")
        
        for key, value in model_state.items():
            # Rename keys to use global layer indices
            # e.g., blocks.0.attn.qkv.weight -> blocks.{start_layer+0}.attn.qkv.weight
            if key.startswith("blocks."):
                parts = key.split(".", 2)
                local_idx = int(parts[1])
                global_idx = start_layer + local_idx
                new_key = f"blocks.{global_idx}.{parts[2]}"
                merged_model_state[new_key] = value
                total_layers = max(total_layers, global_idx + 1)
            else:
                # embedding, pos_embedding, head, etc.
                merged_model_state[key] = value
        
        # Merge optimizer state (optional - complex due to param groups)
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state:
            merged_optimizer_state[f"rank_{rank}"] = opt_state
    
    merged_state["model_state_dict"] = merged_model_state
    if merged_optimizer_state:
        merged_state["optimizer_states"] = merged_optimizer_state
    
    print(f"\n✓ Merged {len(checkpoints)} stages into unified model")
    print(f"  Total transformer blocks: {total_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in merged_model_state.values()):,}")
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged_state, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved to: {output_path} ({size_mb:.1f} MB)")
    
    return merged_state


def list_available_iterations(checkpoint_dir: str | Path) -> list[int]:
    """List all available checkpoint iterations in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    iterations = set()
    
    for ckpt_path in checkpoint_dir.rglob("ckpt_*_i*_r*.pt"):
        filename = ckpt_path.name
        try:
            # Parse iteration: ckpt_e0_i100_r0.pt -> 100
            parts = filename.split("_i")
            if len(parts) >= 2:
                iter_str = parts[1].split("_")[0]
                iterations.add(int(iter_str))
        except (ValueError, IndexError):
            continue
    
    return sorted(iterations)


def main():
    parser = argparse.ArgumentParser(
        description="Merge distributed Asteroid checkpoints into a single model file"
    )
    parser.add_argument(
        "--checkpoint-dir", "-d",
        type=str,
        required=True,
        help="Directory containing checkpoint files (with rank-N subdirs or flat)"
    )
    parser.add_argument(
        "--iteration", "-i",
        type=int,
        default=None,
        help="Iteration to merge (default: latest available)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for merged model (default: <checkpoint_dir>/merged_i<iter>.pt)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available iterations and exit"
    )
    
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    # List mode
    if args.list:
        iterations = list_available_iterations(checkpoint_dir)
        if iterations:
            print(f"Available iterations in {checkpoint_dir}:")
            for it in iterations:
                ckpts = discover_checkpoints(checkpoint_dir, it)
                print(f"  iteration {it}: {len(ckpts)} rank(s)")
        else:
            print(f"No checkpoints found in {checkpoint_dir}")
        return 0
    
    # Determine iteration
    iteration = args.iteration
    if iteration is None:
        available = list_available_iterations(checkpoint_dir)
        if not available:
            print(f"Error: No checkpoints found in {checkpoint_dir}")
            return 1
        iteration = available[-1]
        print(f"Using latest iteration: {iteration}")
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = checkpoint_dir / f"merged_i{iteration}.pt"
    
    # Merge
    try:
        merge_checkpoints(checkpoint_dir, iteration, output_path)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
