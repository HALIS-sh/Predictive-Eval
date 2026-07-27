#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect CSV-style loss features along a GRPO / RL training trajectory.

For each checkpoint:
  1) Run teacher-forcing inference on probe.jsonl (via run_inference.py),
     producing a step-specific parquet.
  2) Aggregate per-task loss features for this checkpoint, e.g.:
       - avg sample-level mean_nll
       - avg sample-level mean_nll_answer_only
       - token-level mean NLL (all tokens / answer tokens)
       - token-level mean NLL on a task-specific vocab (optional)
  3) (Optional) If you provide a csv_fits.json trained before,
     compute a CSV "ability score" for the target task using the
     same regression weights.

Output: a Parquet with one row per (step, task):

  experiment_id, base_model_name, step, task,
  num_probes,
  avg_sample_mean_nll,
  avg_sample_mean_nll_answer,
  token_mean_nll_all,
  token_mean_nll_answer,
  token_mean_nll_vocab_all,
  token_mean_nll_vocab_answer,
  (optional) csv_pred_score

Typical usage (for your GRPO run):

  python scripts/collect_csv_along_grpo.py \
    --experiment_id grpo_dapo_math_17k_qwen3_8b \
    --base_model_name Qwen3-8B \
    --ckpt_root /data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params \
    --steps 100,300,600,1000,1500,2000 \
    --probe_path /data/wenhesun/Predictive-Eval/data/probe/probe.jsonl \
    --run_infer_script scripts/run_inference.py \
    --infer_dir /data/wenhesun/Predictive-Eval/data/infer/grpo_qwen3_8b_math \
    --vocab_path /data/wenhesun/Predictive-Eval/data/csv/vocab.json \
    --csv_fit_path /data/wenhesun/Predictive-Eval/data/csv/csv_fits.json \
    --csv_task math \
    --filter_task math \
    --output_path /data/wenhesun/Predictive-Eval/data/rl_csv/rl_csv_curves.parquet

你也可以先不传 --csv_fit_path，仅仅收集各种 loss 特征。
"""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


def parse_steps(steps_str: str) -> List[int]:
    """Parse comma-separated steps, e.g. '100,300,600'."""
    out = []
    for part in steps_str.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def load_vocab(vocab_path: Optional[str], task: Optional[str]) -> Optional[set]:
    """
    Load task-specific vocab tokens from vocab.json (if provided).

    We assume vocab.json structure roughly like:
      {
        "math": [{"token": "x", ...}, ...],
        "coding": [...],
        ...
      }
    """
    if vocab_path is None or task is None:
        return None
    if not os.path.exists(vocab_path):
        print(f"[WARN] vocab_path {vocab_path} does not exist, ignore vocab features.")
        return None
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_obj = json.load(f)
    if task not in vocab_obj:
        print(f"[WARN] task {task} not in vocab.json keys, ignore vocab features.")
        return None
    tokens = set()
    for item in vocab_obj[task]:
        # 兼容不同schema: 可能是 {"token": "..."} 或直接字符串
        if isinstance(item, dict):
            tok = item.get("token")
        else:
            tok = item
        if tok is not None:
            tokens.add(tok)
    print(f"[INFO] Loaded {len(tokens)} vocab tokens for task={task} from {vocab_path}")
    return tokens


def load_csv_fit(csv_fit_path: Optional[str], task: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Load CSV regression fit info for a specific task, if provided.

    We assume csv_fits.json structure like:
      {
        "math": {
          "features": [...],
          "coef": [...],
          "intercept": float,
          "metrics": {...}
        },
        "coding": {...},
        ...
      }
    """
    if csv_fit_path is None or task is None:
        return None
    if not os.path.exists(csv_fit_path):
        print(f"[WARN] csv_fit_path {csv_fit_path} does not exist, ignore csv_pred_score.")
        return None
    with open(csv_fit_path, "r", encoding="utf-8") as f:
        fits = json.load(f)
    if task not in fits:
        print(f"[WARN] task {task} not found in csv_fits.json, ignore csv_pred_score.")
        return None
    fit = fits[task]
    print(
        f"[INFO] Loaded CSV fit for task={task}: "
        f"{len(fit.get('features', []))} features."
    )
    return fit


def run_inference_for_step(
    step: int,
    ckpt_root: str,
    probe_path: str,
    infer_path: str,
    run_infer_script: str,
    batch_size: int,
    max_length: int,
    answer_tag: str,
):
    """
    Call run_inference.py for a given global_step checkpoint, if parquet not exist.
    """
    infer_path = Path(infer_path)
    infer_path.parent.mkdir(parents=True, exist_ok=True)

    if infer_path.exists():
        print(f"[INFO] Inference parquet exists for step={step}, skip running inference.")
        return

    model_path = os.path.join(ckpt_root, f"global_step_{step}")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {model_path}")

    cmd = [
        "python",
        run_infer_script,
        "--model_path",
        model_path,
        "--probe_path",
        probe_path,
        "--output_path",
        str(infer_path),
        "--batch_size",
        str(batch_size),
        "--max_length",
        str(max_length),
        "--answer_tag",
        answer_tag,
    ]
    print(f"[INFO] Running inference for step={step}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def agg_csv_features_for_task(
    df_task: pd.DataFrame,
    vocab_tokens: Optional[set] = None,
) -> Dict[str, float]:
    """
    Aggregate CSV-style loss features for a single (checkpoint, task).

    df_task: subset of inference parquet with df["task"] == this task.

    Returns a dict of features:
      - avg_sample_mean_nll
      - avg_sample_mean_nll_answer
      - token_mean_nll_all
      - token_mean_nll_answer
      - token_mean_nll_vocab_all
      - token_mean_nll_vocab_answer
    """
    out: Dict[str, float] = {}
    if df_task.empty:
        # Use NaN to indicate missing
        out["avg_sample_mean_nll"] = float("nan")
        out["avg_sample_mean_nll_answer"] = float("nan")
        out["token_mean_nll_all"] = float("nan")
        out["token_mean_nll_answer"] = float("nan")
        out["token_mean_nll_vocab_all"] = float("nan")
        out["token_mean_nll_vocab_answer"] = float("nan")
        return out

    # (1) sample-level aggregates (直接利用 run_inference 的 mean_nll 列)
    if "mean_nll" in df_task.columns:
        out["avg_sample_mean_nll"] = float(df_task["mean_nll"].mean())
    else:
        out["avg_sample_mean_nll"] = float("nan")

    if "mean_nll_answer_only" in df_task.columns:
        out["avg_sample_mean_nll_answer"] = float(df_task["mean_nll_answer_only"].mean())
    else:
        out["avg_sample_mean_nll_answer"] = float("nan")

    # (2) token-level aggregates（扫描 nll / is_answer_mask / tokens）
    all_nll: List[float] = []
    ans_nll: List[float] = []
    vocab_all_nll: List[float] = []
    vocab_ans_nll: List[float] = []

    use_vocab = vocab_tokens is not None

    for _, row in df_task.iterrows():
        nll_list = row["nll"]
        ans_mask = row["is_answer_mask"]
        toks = row["tokens"]

        if not isinstance(nll_list, (list, tuple)) or not isinstance(ans_mask, (list, tuple)):
            continue
        if use_vocab and not isinstance(toks, (list, tuple)):
            continue

        for i, nll in enumerate(nll_list):
            if nll is None or (isinstance(nll, float) and math.isnan(nll)):
                continue
            m_ans = ans_mask[i] if i < len(ans_mask) else 0

            all_nll.append(float(nll))
            if m_ans:
                ans_nll.append(float(nll))

            if use_vocab:
                tok = toks[i] if i < len(toks) else ""
                if tok in vocab_tokens:
                    vocab_all_nll.append(float(nll))
                    if m_ans:
                        vocab_ans_nll.append(float(nll))

    def safe_mean(xs: List[float]) -> float:
        if not xs:
            return float("nan")
        return float(sum(xs) / len(xs))

    out["token_mean_nll_all"] = safe_mean(all_nll)
    out["token_mean_nll_answer"] = safe_mean(ans_nll)
    if use_vocab:
        out["token_mean_nll_vocab_all"] = safe_mean(vocab_all_nll)
        out["token_mean_nll_vocab_answer"] = safe_mean(vocab_ans_nll)
    else:
        out["token_mean_nll_vocab_all"] = float("nan")
        out["token_mean_nll_vocab_answer"] = float("nan")

    return out


def maybe_apply_csv_fit(
    features: Dict[str, float],
    csv_fit: Dict[str, Any],
) -> float:
    """
    If csv_fit is provided, compute a CSV ability score using the same regression.

    csv_fit:
      {
        "features": [...],
        "coef": [...],
        "intercept": float,
        ...
      }

    We expect features dict to contain all feature names used in csv_fit["features"].
    """
    feat_names = csv_fit.get("features", [])
    coef = np.asarray(csv_fit.get("coef", []), dtype=float)
    intercept = float(csv_fit.get("intercept", 0.0))

    if not feat_names or coef.size != len(feat_names):
        print("[WARN] csv_fit features/coef mismatch; skip csv_pred_score.")
        return float("nan")

    x_vals = []
    for name in feat_names:
        if name not in features:
            print(f"[WARN] feature {name} missing in current features; use NaN.")
            x_vals.append(float("nan"))
        else:
            x_vals.append(features[name])
    x = np.asarray(x_vals, dtype=float)

    # 若有 NaN，可以简单用均值填补或直接返回 NaN；这里选择简单策略：
    if np.isnan(x).any():
        return float("nan")

    score = float(np.dot(coef, x) + intercept)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_id", type=str, required=True,
                    help="Logical name for this GRPO/GRPO+RL run, e.g. grpo_math_qwen3_8b.")
    ap.add_argument("--base_model_name", type=str, required=True,
                    help="Base model logical name, e.g. Qwen3-8B.")
    ap.add_argument("--ckpt_root", type=str, required=True,
                    help="Root directory containing global_step_xxx subdirs.")
    ap.add_argument("--steps", type=str, required=True,
                    help="Comma-separated list of steps, e.g. '100,300,600'.")
    ap.add_argument("--probe_path", type=str, required=True,
                    help="Path to probe.jsonl.")
    ap.add_argument("--run_infer_script", type=str, default="scripts/run_inference.py",
                    help="Path to run_inference.py (CLI script).")
    ap.add_argument("--infer_dir", type=str, required=True,
                    help="Directory to store per-step inference parquet files.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--answer_tag", type=str, default="###")

    ap.add_argument("--vocab_path", type=str, default=None,
                    help="Optional vocab.json path for task-specific vocab features.")
    ap.add_argument("--csv_fit_path", type=str, default=None,
                    help="Optional csv_fits.json path to compute csv_pred_score.")
    ap.add_argument("--csv_task", type=str, default=None,
                    help="Task name for which csv_fit is defined (e.g. 'math').")
    ap.add_argument("--filter_task", type=str, default=None,
                    help="If set, only aggregate rows with df['task'] == filter_task "
                         "(e.g. rl is math-only).")
    ap.add_argument("--output_path", type=str, required=True,
                    help="Output Parquet path for aggregated CSV features along steps.")
    args = ap.parse_args()

    steps = parse_steps(args.steps)
    print(f"[INFO] Steps to process: {steps}")

    vocab_tokens = load_vocab(args.vocab_path, args.filter_task or args.csv_task)
    csv_fit = load_csv_fit(args.csv_fit_path, args.csv_task)

    records: List[Dict[str, Any]] = []

    infer_dir = Path(args.infer_dir)
    infer_dir.mkdir(parents=True, exist_ok=True)

    for step in steps:
        infer_path = infer_dir / f"step_{step}.probe.parquet"

        # 1) run inference for this step (if parquet not already produced)
        run_inference_for_step(
            step=step,
            ckpt_root=args.ckpt_root,
            probe_path=args.probe_path,
            infer_path=str(infer_path),
            run_infer_script=args.run_infer_script,
            batch_size=args.batch_size,
            max_length=args.max_length,
            answer_tag=args.answer_tag,
        )

        # 2) load inference parquet
        if not infer_path.exists():
            raise FileNotFoundError(f"[FATAL] expected inference parquet not found: {infer_path}")
        df = pd.read_parquet(infer_path)

        if args.filter_task is not None:
            tasks = [args.filter_task]
        else:
            tasks = sorted(df["task"].unique())

        for task in tasks:
            df_task = df[df["task"] == task].copy()
            num_probes = int(len(df_task))
            feat = agg_csv_features_for_task(df_task, vocab_tokens=vocab_tokens)

            # (optional) csv_pred_score
            csv_score = float("nan")
            if csv_fit is not None and (args.csv_task == task or args.csv_task is None):
                csv_score = maybe_apply_csv_fit(feat, csv_fit)

            rec = {
                "experiment_id": args.experiment_id,
                "base_model_name": args.base_model_name,
                "step": int(step),
                "task": task,
                "num_probes": num_probes,
            }
            rec.update(feat)
            rec["csv_pred_score"] = csv_score

            records.append(rec)

        print(f"[INFO] Done step={step}, aggregated tasks={tasks}")

    out_df = pd.DataFrame.from_records(records)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[OK] wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()