#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract Rank-1 update dynamics along an RL training trajectory.

For each checkpoint step:
  - Load base model and checkpoint weights
  - Compute per-layer parameter difference ΔW = W_step - W_base
  - For selected layers (typically linear layers in MLP / attention):
      * Flatten ΔW to [out_dim, in_dim]
      * Run low-rank SVD (q=1) to get top-1 component:
            ΔW ≈ s1 * u1 @ v1^T
      * Compute norms:
            ||ΔW||_F, ||ΔW_rank1||_F, rank1_fraction = ||ΔW_rank1|| / ||ΔW||
      * Store u1, s1, v1, shape for later reconstruction

Outputs:
  1) One file per step with Rank-1 updates:
       {output_dir}/rank1_step_{step}.pt
     Each file is a dict: layer_name -> {
         "u":  tensor [out_dim],
         "s":  scalar tensor,
         "v":  tensor [in_dim],
         "shape": original ΔW shape (tuple)
     }
     where out_dim * in_dim == prod(shape).

  2) A summary Parquet with per-step, per-layer stats:
       {output_dir}/rank1_dynamics.parquet
     Columns:
       - base_model_path
       - ckpt_root
       - step
       - layer_name
       - param_shape
       - out_dim, in_dim
       - fro_norm_full
       - fro_norm_rank1
       - rank1_fraction
       - s1
       - mean_abs_delta
       - max_abs_delta

Usage example (your GRPO run):

  python scripts/extract_rank1_dynamics.py \
    --base_model_path /data/wenhesun/model/Qwen/Qwen3-8B \
    --ckpt_root /data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params \
    --steps 100,300,600,1000,1500,2000 \
    --output_dir /data/wenhesun/Predictive-Eval/data/alpharl/rank1_qwen3_8b_grpo \
    --include_pattern "model.layers" \
    --exclude_pattern "ln|norm|embedding|lm_head" \
    --dtype float32

You can also set --steps auto to automatically scan global_step_* under ckpt_root.
"""

import argparse
import re
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM


def parse_steps(steps_str: str, ckpt_root: str) -> List[int]:
    """
    Parse steps from:
      - 'auto': scan ckpt_root/global_step_* and extract integers
      - '100,300,600': comma-separated integers
    """
    if steps_str == "auto":
        root = Path(ckpt_root)
        steps = []
        for p in root.glob("global_step_*"):
            m = re.search(r"global_step_(\d+)", p.name)
            if m:
                steps.append(int(m.group(1)))
        steps = sorted(set(steps))
        if not steps:
            raise ValueError(f"No global_step_* found under {ckpt_root}")
        print(f"[INFO] auto-discovered steps: {steps}")
        return steps

    steps: List[int] = []
    for part in steps_str.split(","):
        part = part.strip()
        if not part:
            continue
        steps.append(int(part))
    steps = sorted(set(steps))
    print(f"[INFO] explicit steps: {steps}")
    return steps


def compile_pattern(pat: Optional[str]) -> Optional[re.Pattern]:
    if pat is None or pat == "":
        return None
    return re.compile(pat)


def param_selected(
    name: str,
    tensor: torch.Tensor,
    include_pat: Optional[re.Pattern],
    exclude_pat: Optional[re.Pattern],
    min_ndim: int = 2,
) -> bool:
    """
    Decide whether this parameter should be considered for Rank-1 SVD.

    Heuristics:
      - dimension >= min_ndim (we care about matrices, not vectors)
      - match include_pattern (if provided)
      - NOT match exclude_pattern (if provided)
      - and typically 'weight' in name
    """
    if tensor.ndim < min_ndim:
        return False
    if "weight" not in name:
        return False
    if include_pat is not None and include_pat.search(name) is None:
        return False
    if exclude_pat is not None and exclude_pat.search(name) is not None:
        return False
    return True


def load_model_state(
    model_path: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load a causal LM using HF transformers and return its state_dict (CPU tensors).
    """
    print(f"[INFO] Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={ "": device } if device != "auto" else None,
    )
    # Move to CPU to make sure all tensors are CPU for diff/SVD
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    # free model to release GPU/CPU RAM
    del model
    torch.cuda.empty_cache()
    return state


def compute_rank1_for_layer(delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Given ΔW for a layer (any shape with ndim >= 2),
    flatten to [out_dim, in_dim], perform q=1 low-rank SVD, and return:

      u: [out_dim]
      s: scalar tensor
      v: [in_dim]
      stats: dict with norms, fraction, etc.
    """
    # Move to float32 for numerical stability
    dW = delta.to(torch.float32)
    # Flatten last dimensions into in_dim
    out_dim = dW.shape[0]
    in_dim = int(np.prod(dW.shape[1:]))
    W = dW.reshape(out_dim, in_dim)  # [O, I]

    # Frobenius norm of full ΔW
    fro_full = torch.linalg.norm(W)
    # If ΔW is (almost) zero, we can early-return
    if fro_full.item() == 0.0:
        u = torch.zeros(out_dim, dtype=torch.float32)
        v = torch.zeros(in_dim, dtype=torch.float32)
        s = torch.tensor(0.0, dtype=torch.float32)
        stats = {
            "out_dim": out_dim,
            "in_dim": in_dim,
            "fro_norm_full": 0.0,
            "fro_norm_rank1": 0.0,
            "rank1_fraction": 0.0,
            "s1": 0.0,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }
        return u, s, v, stats

    # Use low-rank SVD to get top-1 component
    # torch.svd_lowrank(W, q=1) returns U [O,1], S [1], V [I,1]
    # For PyTorch <2.0, replace with torch.pca_lowrank or full SVD if needed.
    U, S, V = torch.svd_lowrank(W, q=1)
    u1 = U[:, 0]         # [O]
    s1 = S[0]            # scalar
    v1 = V[:, 0]         # [I]

    # Rank-1 reconstruction
    rank1 = (u1.unsqueeze(1) @ v1.unsqueeze(0)) * s1   # [O, I]
    fro_rank1 = torch.linalg.norm(rank1)

    rank1_fraction = (fro_rank1 / (fro_full + 1e-8)).item()
    mean_abs = dW.abs().mean().item()
    max_abs = dW.abs().max().item()

    stats = {
        "out_dim": out_dim,
        "in_dim": in_dim,
        "fro_norm_full": fro_full.item(),
        "fro_norm_rank1": fro_rank1.item(),
        "rank1_fraction": rank1_fraction,
        "s1": s1.item(),
        "mean_abs_delta": mean_abs,
        "max_abs_delta": max_abs,
    }
    return u1.detach().cpu(), s1.detach().cpu(), v1.detach().cpu(), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="HF path to base model (e.g. /data/.../Qwen3-8B)",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        required=True,
        help="Root directory containing RL checkpoints global_step_xxx",
    )
    parser.add_argument(
        "--steps",
        type=str,
        required=True,
        help="Comma-separated list of steps, e.g. '100,300,600' or 'auto' to scan ckpt_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write rank1_step_{step}.pt and rank1_dynamics.parquet",
    )
    parser.add_argument(
        "--include_pattern",
        type=str,
        default="model.layers",
        help="Regex to select parameter names (e.g. 'model.layers'). Default focuses on transformer blocks.",
    )
    parser.add_argument(
        "--exclude_pattern",
        type=str,
        default="ln|norm|embedding|lm_head",
        help="Regex to exclude parameter names (e.g. layernorm / embeddings / lm_head).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype to load weights; internally delta & SVD will use float32.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load models on ('cpu', 'cuda', or 'auto'). "
             "Note: we immediately move state_dict to CPU for ΔW & SVD.",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="Optional: maximum number of layers/parameters to process per model (for quick tests).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ parse steps ------------------
    steps = parse_steps(args.steps, args.ckpt_root)

    # ------------------ patterns --------------------
    include_pat = compile_pattern(args.include_pattern)
    exclude_pat = compile_pattern(args.exclude_pattern)

    # ------------------ dtype -----------------------
    if args.dtype == "float32":
        load_dtype = torch.float32
    elif args.dtype == "float16":
        load_dtype = torch.float16
    else:
        load_dtype = torch.bfloat16

    # ------------------ load base state -------------
    base_state = load_model_state(
        args.base_model_path,
        dtype=load_dtype,
        device="cpu" if args.device == "auto" else args.device,
    )

    # Filter candidate params once using base_state
    selected_param_names: List[str] = []
    for name, tensor in base_state.items():
        if param_selected(name, tensor, include_pat, exclude_pat, min_ndim=2):
            selected_param_names.append(name)
    selected_param_names = sorted(selected_param_names)
    if args.max_layers is not None:
        selected_param_names = selected_param_names[: args.max_layers]

    print(f"[INFO] Selected {len(selected_param_names)} parameters for Rank-1 SVD.")
    if not selected_param_names:
        raise ValueError("No parameters selected; check include_pattern / exclude_pattern settings.")

    # ------------------ main loop over steps --------
    dynamics_records: List[Dict[str, Any]] = []

    for step in steps:
        ckpt_path = os.path.join(args.ckpt_root, f"global_step_{step}")
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Checkpoint dir not found for step={step}: {ckpt_path}")

        print(f"\n[STEP] Processing step={step} at {ckpt_path}")

        # load checkpoint state
        ckpt_state = load_model_state(
            ckpt_path,
            dtype=load_dtype,
            device="cpu" if args.device == "auto" else args.device,
        )

        # dict for rank1 updates for this step
        rank1_updates: Dict[str, Dict[str, Any]] = {}

        for name in selected_param_names:
            if name not in ckpt_state:
                print(f"[WARN] param {name} not found in checkpoint state; skip.")
                continue

            base_param = base_state[name]
            step_param = ckpt_state[name]
            if base_param.shape != step_param.shape:
                print(f"[WARN] shape mismatch for {name}: base {base_param.shape}, "
                      f"step {step_param.shape}; skip.")
                continue

            delta = step_param - base_param  # ΔW
            u1, s1, v1, stats = compute_rank1_for_layer(delta)

            # store rank1 components
            rank1_updates[name] = {
                "u": u1.to(torch.float16),   # reduce disk size
                "s": s1.to(torch.float32),
                "v": v1.to(torch.float16),
                "shape": tuple(base_param.shape),
            }

            rec = {
                "base_model_path": args.base_model_path,
                "ckpt_root": args.ckpt_root,
                "step": int(step),
                "layer_name": name,
                "param_shape": str(tuple(base_param.shape)),
            }
            rec.update(stats)
            dynamics_records.append(rec)

        # Save per-step rank1 updates
        rank1_path = out_dir / f"rank1_step_{step}.pt"
        torch.save(rank1_updates, rank1_path)
        print(f"[INFO] Saved Rank-1 updates for step={step} to {rank1_path}")
        print(f"[INFO] Processed {len(rank1_updates)} layers at step={step}")

        # Free ckpt_state
        del ckpt_state
        torch.cuda.empty_cache()

    # ------------------ save summary parquet --------
    df_dyn = pd.DataFrame.from_records(dynamics_records)
    dyn_path = out_dir / "rank1_dynamics.parquet"
    df_dyn.to_parquet(dyn_path, index=False)
    print(f"\n[OK] Wrote Rank-1 dynamics summary ({len(df_dyn)} rows) to {dyn_path}")


if __name__ == "__main__":
    main()