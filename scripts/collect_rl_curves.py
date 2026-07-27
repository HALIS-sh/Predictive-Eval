#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect true RL performance curves along a GRPO / RL training trajectory.

For each checkpoint directory:
  - Call an external evaluation script (e.g. Alpha-RL's data_eval.py)
    with the checkpoint as --model_path, and an --output_path JSON file.
  - Parse the JSON file to extract a metric (e.g. "acc", "pass@1", "reward").
  - Store (experiment_id, base_model_name, algo, task, step, rl_score) as a row.

Output:
  rl_curves.parquet  (long-format table)

Typical usage for your case:

  python scripts/collect_rl_curves.py \
    --experiment_id grpo_dapo_math_17k_qwen3_8b \
    --base_model_name Qwen3-8B \
    --algo grpo \
    --task math \
    --ckpt_root /data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params \
    --steps 100,300,600,1000,1500,2000 \
    --eval_script /data/wenhesun/AlphaRL/data_eval.py \
    --eval_output_dir /data/wenhesun/Predictive-Eval/data/rl_eval/grpo_qwen3_8b_math \
    --metric_key acc \
    --eval_extra_args "--batch_size 16 --dataset math500" \
    --output_path /data/wenhesun/Predictive-Eval/data/rl_eval/rl_curves.parquet

Requirements for eval_script:
  - Must accept at least:
        --model_path  <checkpoint_dir_or_model_name>
        --output_path <json_path_to_write_metrics>
  - It should write a JSON object, e.g.:
        {"acc": 0.42, "pass@1": 0.40, ...}
"""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def parse_steps(steps_str: str) -> List[int]:
    """Parse comma-separated steps, e.g. '100,300,600'."""
    out: List[int] = []
    for part in steps_str.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def run_eval_for_step(
    step: int,
    ckpt_root: str,
    eval_script: str,
    eval_output_dir: str,
    metric_json_path: str,
    eval_extra_args: str = "",
    force: bool = False,
):
    """
    Call external eval_script on checkpoint global_step_{step}.

    eval_script is expected to support:

        python eval_script \
            --model_path <model_dir> \
            --output_path <metric_json_path> \
            [eval_extra_args...]

    If metric_json_path already exists and force=False, we skip re-running eval.
    """
    metric_json = Path(metric_json_path)
    if metric_json.exists() and not force:
        print(f"[INFO] Metrics for step={step} already exist at {metric_json}, skip eval.")
        return

    model_path = os.path.join(ckpt_root, f"global_step_{step}")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Checkpoint directory not found for step={step}: {model_path}")

    Path(eval_output_dir).mkdir(parents=True, exist_ok=True)

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

    print(f"[INFO] Running eval for step={step}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_metric_from_json(
    json_path: str,
    metric_key: str,
) -> float:
    """
    Read a JSON file and extract the metric at `metric_key`.

    metric_key can be:
      - a top-level key, e.g. "acc" or "pass@1"
      - or a dotted path, e.g. "metrics.acc"

    Returns float('nan') on failure.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read JSON from {json_path}: {e}")
        return float("nan")

    # support dotted keys
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
        print(f"[WARN] metric value for key '{metric_key}' not float-like in {json_path}: {e}")
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_id", type=str, required=True,
                    help="Logical name for this RL run, e.g. grpo_math_qwen3_8b.")
    ap.add_argument("--base_model_name", type=str, required=True,
                    help="Base model logical name, e.g. Qwen3-8B.")
    ap.add_argument("--algo", type=str, default="grpo",
                    help="RL algorithm name, e.g. grpo, dpo, ppo.")
    ap.add_argument("--task", type=str, default="math",
                    help="Task name, e.g. math / coding / logic.")
    ap.add_argument("--ckpt_root", type=str, required=True,
                    help="Root directory containing global_step_xxx subdirs.")
    ap.add_argument("--steps", type=str, required=True,
                    help="Comma-separated list of steps, e.g. '100,300,600'.")
    ap.add_argument("--eval_script", type=str, required=True,
                    help="Path to evaluation script (e.g. Alpha-RL's data_eval.py).")
    ap.add_argument("--eval_output_dir", type=str, required=True,
                    help="Directory to store per-step JSON metric files.")
    ap.add_argument("--metric_key", type=str, default="acc",
                    help="Key in JSON metrics to use as rl_score, e.g. 'acc' or 'metrics.acc'.")
    ap.add_argument("--eval_extra_args", type=str, default="",
                    help="Additional CLI args passed to eval_script (quoted string).")
    ap.add_argument("--force", action="store_true",
                    help="If set, re-run eval even if JSON metric already exists.")
    ap.add_argument("--output_path", type=str, required=True,
                    help="Output Parquet path for rl_curves.")
    args = ap.parse_args()

    steps = parse_steps(args.steps)
    print(f"[INFO] Steps to evaluate: {steps}")

    records: List[Dict[str, Any]] = []

    for step in steps:
        metric_json_path = os.path.join(
            args.eval_output_dir,
            f"metrics_step_{step}.json"
        )

        # 1) run eval for this step (if needed)
        run_eval_for_step(
            step=step,
            ckpt_root=args.ckpt_root,
            eval_script=args.eval_script,
            eval_output_dir=args.eval_output_dir,
            metric_json_path=metric_json_path,
            eval_extra_args=args.eval_extra_args,
            force=args.force,
        )

        # 2) parse metrics and extract rl_score
        rl_score = extract_metric_from_json(metric_json_path, args.metric_key)

        rec = {
            "experiment_id": args.experiment_id,
            "base_model_name": args.base_model_name,
            "algo": args.algo,
            "task": args.task,
            "step": int(step),
            "rl_score": rl_score,
            "metric_key": args.metric_key,
            "metric_json_path": metric_json_path,
        }
        records.append(rec)
        print(f"[INFO] step={step}: rl_score={rl_score:.4f}")

    df = pd.DataFrame.from_records(records)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()