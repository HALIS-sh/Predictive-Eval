#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Rank-1 updates (from extract_rank1_dynamics.py) back to the base model,
save Rank-1-only checkpoints, and evaluate them with an external eval script.

Typical usage (for your GRPO run):

  python scripts/apply_rank1_and_eval.py \
    --base_model_path /data/wenhesun/model/Qwen/Qwen3-8B \
    --rank1_dir /data/wenhesun/Predictive-Eval/data/alpharl/rank1_qwen3_8b_grpo \
    --steps 100,300,600,1000,1500,2000 \
    --model_out_root /data/wenhesun/Predictive-Eval/checkpoints/rank1_qwen3_8b_grpo \
    --eval_script /data/wenhesun/AlphaRL/data_eval.py \
    --eval_output_dir /data/wenhesun/Predictive-Eval/data/alpharl/rank1_eval \
    --metric_key acc \
    --eval_extra_args "--batch_size 16 --dataset math500" \
    --experiment_id grpo_dapo_math_17k_qwen3_8b \
    --base_model_name Qwen3-8B \
    --algo grpo \
    --task math \
    --base_score 0.25 \
    --rl_curves_path /data/wenhesun/Predictive-Eval/data/rl_eval/rl_curves.parquet \
    --output_path /data/wenhesun/Predictive-Eval/data/alpharl/rank1_eval/alpha_rl_rank1_curves.parquet

Requirements:

1. You have already run extract_rank1_dynamics.py, so that:
   - rank1_dir/rank1_step_{step}.pt exist
   - rank1_dir/rank1_dynamics.parquet exists (not required here, just FYI)

2. eval_script must support CLI:
     python eval_script --model_path <model_dir> --output_path <json_path> [extra_args...]
   and write a JSON file with a metric you specify via --metric_key, e.g.:
     {"acc": 0.42, "pass@1": 0.40, ...}

This script will output:
  - For each step:
      * a new HF checkpoint directory: {model_out_root}/rank1_step_{step}
      * an eval metrics JSON:        {eval_output_dir}/metrics_rank1_step_{step}.json
  - A summary Parquet:
      alpha_rl_rank1_curves.parquet
    with columns:
      experiment_id, base_model_name, algo, task,
      step, rank1_score, metric_key, eval_json_path,
      (optional) rl_true_score, base_score, frac_recovered
"""

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM


# ------------------------
# Small utilities
# ------------------------

def parse_steps(steps_str: str, rank1_dir: str) -> List[int]:
    """
    Parse steps from:
      - 'auto': scan rank1_dir/rank1_step_*.pt
      - '100,300,600': comma-separated integers
    """
    if steps_str == "auto":
        root = Path(rank1_dir)
        steps: List[int] = []
        for p in root.glob("rank1_step_*.pt"):
            m = re.search(r"rank1_step_(\d+)\.pt", p.name)
            if m:
                steps.append(int(m.group(1)))
        steps = sorted(set(steps))
        if not steps:
            raise ValueError(f"No rank1_step_*.pt found under {rank1_dir}")
        print(f"[INFO] auto-discovered steps from rank1_dir: {steps}")
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


def run_eval(
    eval_script: str,
    model_path: str,
    metric_json_path: str,
    eval_extra_args: str = "",
    force: bool = False,
):
    """
    Call external eval_script with:
        python eval_script --model_path <model_path> --output_path <metric_json_path> [extra_args...]

    If metric_json_path exists and force=False, skip re-eval.
    """
    metric_json = Path(metric_json_path)
    if metric_json.exists() and not force:
        print(f"[INFO] Metrics already exist at {metric_json}, skip eval.")
        return

    cmd = [
        "python",
        eval_script,
        "--model_path",
        model_path,
        "--output_path",
        str(metric_json),
    ]
    if eval_extra_args:
        cmd.extend(shlex.split(eval_extra_args))

    print(f"[INFO] Running eval: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_metric_from_json(json_path: str, metric_key: str) -> float:
    """
    Read JSON and extract metric_key.
    metric_key can be dotted, e.g. 'metrics.acc' or a top-level key 'acc'.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read metrics JSON from {json_path}: {e}")
        return float("nan")

    cur: Any = data
    for part in metric_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            print(f"[WARN] metric_key '{metric_key}' not found in {json_path}")
            return float("nan")

    try:
        return float(cur)
    except Exception as e:
        print(f"[WARN] Value for metric_key '{metric_key}' not float-like in {json_path}: {e}")
        return float("nan")


def load_rl_true_scores(
    rl_curves_path: Optional[str],
    experiment_id: Optional[str],
    task: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Optionally load rl_curves.parquet and filter by experiment_id, task.
    We expect columns: [experiment_id, task, step, rl_score] at least.
    """
    if rl_curves_path is None:
        return None
    rl_path = Path(rl_curves_path)
    if not rl_path.is_file():
        print(f"[WARN] rl_curves_path {rl_path} not found; will not compute frac_recovered.")
        return None

    df_rl = pd.read_parquet(rl_path)
    expected = {"experiment_id", "task", "step", "rl_score"}
    if not expected.issubset(df_rl.columns):
        print(f"[WARN] rl_curves.parquet missing columns {expected - set(df_rl.columns)}; skip RL true score lookup.")
        return None

    if experiment_id is not None:
        df_rl = df_rl[df_rl["experiment_id"] == experiment_id]
    if task is not None:
        df_rl = df_rl[df_rl["task"] == task]

    return df_rl


# ------------------------
# Rank-1 application
# ------------------------

def apply_rank1_to_base_model(
    base_model_path: str,
    rank1_path: str,
    save_dir: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
):
    """
    Load base model from base_model_path, apply rank1 updates from rank1_path,
    and save to save_dir as a HF checkpoint.

    rank1_path (torch file) should be a dict:
      param_name -> {
          "u": tensor [out_dim] (float16/float32),
          "s": scalar tensor,
          "v": tensor [in_dim],
          "shape": tuple(...)
      }
    produced by extract_rank1_dynamics.py.
    """
    print(f"[INFO] Loading base model from {base_model_path} to apply Rank-1 from {rank1_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=None if device == "cpu" else { "": device },
    )

    rank1_updates: Dict[str, Dict[str, Any]] = torch.load(rank1_path, map_location="cpu")

    # Apply updates
    with torch.no_grad():
        name2param = dict(model.named_parameters())
        applied_count = 0

        for name, info in rank1_updates.items():
            if name not in name2param:
                print(f"[WARN] Parameter {name} from rank1 file not found in model; skip.")
                continue

            param = name2param[name]
            u = info["u"].to(torch.float32)  # [O]
            s = info["s"].to(torch.float32)  # scalar
            v = info["v"].to(torch.float32)  # [I]
            shape = tuple(info["shape"])

            # Reconstruct ΔW_rank1 in flattened 2D
            rank1_flat = torch.outer(u, v) * s  # [O, I]
            # Reshape to original shape
            rank1 = rank1_flat.view(shape).to(param.dtype)

            if param.shape != rank1.shape:
                print(f"[WARN] Shape mismatch for {name}: param {param.shape}, rank1 {rank1.shape}; skip.")
                continue

            param += rank1.to(param.device)
            applied_count += 1

    print(f"[INFO] Applied Rank-1 updates to {applied_count} parameters.")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"[INFO] Saved Rank-1 model to {save_path}")

    # Clear to free memory
    del model
    torch.cuda.empty_cache()


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    # Model & Rank1 inputs
    ap.add_argument("--base_model_path", type=str, required=True,
                    help="HF path to base model (e.g. /data/.../Qwen3-8B)")
    ap.add_argument("--rank1_dir", type=str, required=True,
                    help="Directory containing rank1_step_{step}.pt from extract_rank1_dynamics.py")
    ap.add_argument("--steps", type=str, required=True,
                    help="Comma-separated list of steps (e.g. '100,300') or 'auto' to infer from rank1_dir.")
    ap.add_argument("--model_out_root", type=str, required=True,
                    help="Root directory to save Rank-1-applied HF models, subdirs = rank1_step_{step}")

    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float32", "float16", "bfloat16"],
                    help="dtype used when loading base model for applying rank1 updates.")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Device for applying rank1 updates ('cpu', 'cuda'). Eval will be separate via eval_script.")

    # Eval script
    ap.add_argument("--eval_script", type=str, required=True,
                    help="Path to external eval script (e.g. AlphaRL/data_eval.py)")
    ap.add_argument("--eval_output_dir", type=str, required=True,
                    help="Directory to save per-step metrics JSON for Rank-1 models.")
    ap.add_argument("--metric_key", type=str, default="acc",
                    help="Metric key in eval JSON (e.g. 'acc', 'pass@1', 'metrics.acc').")
    ap.add_argument("--eval_extra_args", type=str, default="",
                    help="Extra CLI args passed to eval_script, e.g. \"--batch_size 16 --dataset math500\".")
    ap.add_argument("--force_eval", action="store_true",
                    help="If set, re-run eval even if metrics JSON already exists.")

    # Meta info and RL true curves (optional)
    ap.add_argument("--experiment_id", type=str, default=None,
                    help="Logical experiment id for this RL run, for logging.")
    ap.add_argument("--base_model_name", type=str, default=None,
                    help="Logical name of base model (e.g. Qwen3-8B).")
    ap.add_argument("--algo", type=str, default=None,
                    help="RL algorithm name (e.g. grpo, dpo).")
    ap.add_argument("--task", type=str, default=None,
                    help="Task name (e.g. math, coding).")
    ap.add_argument("--base_score", type=float, default=None,
                    help="Base model's score on the same eval metric (for fraction recovered).")
    ap.add_argument("--rl_curves_path", type=str, default=None,
                    help="Optional: rl_curves.parquet with columns [experiment_id, task, step, rl_score].")

    # Output summary
    ap.add_argument("--output_path", type=str, required=True,
                    help="Output Parquet path for summary of Rank-1 eval results.")

    args = ap.parse_args()

    # ---------------- dtype/device ----------------
    if args.dtype == "float32":
        torch_dtype = torch.float32
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    steps = parse_steps(args.steps, args.rank1_dir)

    model_out_root = Path(args.model_out_root)
    model_out_root.mkdir(parents=True, exist_ok=True)

    eval_output_dir = Path(args.eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally load true RL curves
    df_rl_true = load_rl_true_scores(args.rl_curves_path, args.experiment_id, args.task)

    records: List[Dict[str, Any]] = []

    for step in steps:
        print(f"\n[STEP] Applying Rank-1 and evaluating step={step}")
        rank1_path = Path(args.rank1_dir) / f"rank1_step_{step}.pt"
        if not rank1_path.is_file():
            raise FileNotFoundError(f"Rank-1 file not found for step={step}: {rank1_path}")

        # 1) Apply Rank-1 to base, save new HF model
        rank1_model_dir = model_out_root / f"rank1_step_{step}"
        apply_rank1_to_base_model(
            base_model_path=args.base_model_path,
            rank1_path=str(rank1_path),
            save_dir=str(rank1_model_dir),
            dtype=torch_dtype,
            device=args.device,
        )

        # 2) Run eval via external script
        metric_json_path = eval_output_dir / f"metrics_rank1_step_{step}.json"
        run_eval(
            eval_script=args.eval_script,
            model_path=str(rank1_model_dir),
            metric_json_path=str(metric_json_path),
            eval_extra_args=args.eval_extra_args,
            force=args.force_eval,
        )

        # 3) Extract metric
        rank1_score = extract_metric_from_json(str(metric_json_path), args.metric_key)

        rec: Dict[str, Any] = {
            "experiment_id": args.experiment_id,
            "base_model_name": args.base_model_name,
            "algo": args.algo,
            "task": args.task,
            "step": int(step),
            "rank1_score": rank1_score,
            "metric_key": args.metric_key,
            "rank1_model_dir": str(rank1_model_dir),
            "metric_json_path": str(metric_json_path),
        }

        # 4) Optionally: compare to true RL score & base score
        rl_true_score = None
        frac_recovered = None

        if df_rl_true is not None:
            row = df_rl_true[df_rl_true["step"] == step]
            if len(row) > 0:
                rl_true_score = float(row.iloc[0]["rl_score"])

        if rl_true_score is not None and args.base_score is not None:
            base = args.base_score
            denom = (rl_true_score - base)
            if abs(denom) < 1e-8:
                frac_recovered = float("nan")
            else:
                frac_recovered = (rank1_score - base) / denom

        rec["rl_true_score"] = rl_true_score
        rec["base_score"] = args.base_score
        rec["frac_recovered"] = frac_recovered

        print(f"[RESULT] step={step} | rank1_score={rank1_score:.4f} | "
              f"rl_true={rl_true_score} | base={args.base_score} | "
              f"frac_recovered={frac_recovered}")

        records.append(rec)

    # 5) Write summary parquet
    df_out = pd.DataFrame.from_records(records)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(f"\n[OK] Wrote Rank-1 eval summary ({len(df_out)} rows) to {out_path}")


if __name__ == "__main__":
    main()