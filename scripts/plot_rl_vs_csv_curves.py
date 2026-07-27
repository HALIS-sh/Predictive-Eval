#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot RL true curves vs CSV-style curves, and compute per-run regression R^2.

Inputs:
  - rl_curves.parquet           (from collect_rl_curves.py)
      columns: [experiment_id, base_model_name, algo, task, step, rl_score, ...]
  - rl_csv_curves.parquet       (from collect_csv_along_grpo.py)
      columns: [experiment_id, base_model_name, step, task, csv_pred_score, ...other loss features...]

We merge on (experiment_id, task, step) and, for each (experiment_id, task):
  - Sort by step
  - Plot:
        step vs rl_score      (solid line)
        step vs csv_pred_score (dashed line)
    Save to: <output_dir>/<experiment_id>__<task>.png
  - Fit linear regression:
        rl_score = a + b * csv_pred_score
    and compute:
        R^2, slope b, intercept a, Pearson r

Outputs:
  - One PNG plot per (experiment_id, task)
  - summary.csv in output_dir with columns:
      experiment_id, task, num_points, r2, pearson_r, slope, intercept

Usage example:

  python scripts/plot_rl_vs_csv_curves.py \
    --rl_curves_path /data/wenhesun/Predictive-Eval/data/rl_eval/rl_curves.parquet \
    --csv_curves_path /data/wenhesun/Predictive-Eval/data/rl_csv/rl_csv_curves.parquet \
    --output_dir /data/wenhesun/Predictive-Eval/data/rl_csv/plots \
    --csv_score_col csv_pred_score \
    --min_points 3
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def sanitize_filename(s: str) -> str:
    """Remove characters that are problematic in file names."""
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s.strip("_")


def fit_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit rl_score = a + b * csv_score

    x: (n,) csv scores
    y: (n,) rl scores

    Returns: (r2, pearson_r, slope, intercept)
    """
    assert x.shape == y.shape
    n = x.shape[0]
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # reshape for sklearn
    X = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Pearson correlation
    if np.std(x) == 0 or np.std(y) == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(x, y)[0, 1])

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    return float(r2), float(pearson_r), slope, intercept


def plot_run(
    df_run: pd.DataFrame,
    experiment_id: str,
    task: str,
    csv_score_col: str,
    output_dir: Path,
):
    """
    Plot step vs rl_score and step vs csv_score for a single (experiment_id, task).
    df_run must already be sorted by step.
    """
    steps = df_run["step"].to_numpy()
    rl_score = df_run["rl_score"].to_numpy()
    csv_score = df_run[csv_score_col].to_numpy()

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(steps, rl_score, marker="o", label="RL score (true)")
    plt.plot(steps, csv_score, marker="s", linestyle="--", label=f"CSV score ({csv_score_col})")

    plt.xlabel("Step")
    plt.ylabel("Score / CSV value")
    plt.title(f"{experiment_id} | task={task}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fname = sanitize_filename(f"{experiment_id}__{task}.png")
    out_path = output_dir / fname
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_curves_path", type=str, required=True,
                    help="Path to rl_curves.parquet (from collect_rl_curves.py)")
    ap.add_argument("--csv_curves_path", type=str, required=True,
                    help="Path to rl_csv_curves.parquet (from collect_csv_along_grpo.py)")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Directory to save plots and summary.csv")
    ap.add_argument("--csv_score_col", type=str, default="csv_pred_score",
                    help="Column name in csv_curves representing CSV ability score.")
    ap.add_argument("--min_points", type=int, default=3,
                    help="Minimum number of points required per run to fit regression.")
    ap.add_argument("--experiment_filter", type=str, default=None,
                    help="Optional substring filter on experiment_id (only process matching ones).")
    ap.add_argument("--task_filter", type=str, default=None,
                    help="Optional filter on task (only process this task).")
    args = ap.parse_args()

    rl_path = Path(args.rl_curves_path)
    csv_path = Path(args.csv_curves_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading RL curves from {rl_path}")
    df_rl = pd.read_parquet(rl_path)
    print(f"[INFO] RL curves rows: {len(df_rl)}")

    print(f"[INFO] Loading CSV curves from {csv_path}")
    df_csv = pd.read_parquet(csv_path)
    print(f"[INFO] CSV curves rows: {len(df_csv)}")

    # Basic checks
    needed_cols_rl = {"experiment_id", "task", "step", "rl_score"}
    needed_cols_csv = {"experiment_id", "task", "step", args.csv_score_col}
    if not needed_cols_rl.issubset(df_rl.columns):
        missing = needed_cols_rl - set(df_rl.columns)
        raise ValueError(f"rl_curves missing columns: {missing}")
    if not needed_cols_csv.issubset(df_csv.columns):
        missing = needed_cols_csv - set(df_csv.columns)
        raise ValueError(f"csv_curves missing columns: {missing}")

    # Optional filters
    if args.experiment_filter:
        df_rl = df_rl[df_rl["experiment_id"].astype(str).str.contains(args.experiment_filter)]
        df_csv = df_csv[df_csv["experiment_id"].astype(str).str.contains(args.experiment_filter)]
        print(f"[INFO] After experiment_filter, RL rows={len(df_rl)}, CSV rows={len(df_csv)}")

    if args.task_filter:
        df_rl = df_rl[df_rl["task"] == args.task_filter]
        df_csv = df_csv[df_csv["task"] == args.task_filter]
        print(f"[INFO] After task_filter, RL rows={len(df_rl)}, CSV rows={len(df_csv)}")

    # Merge on experiment_id, task, step
    df = pd.merge(
        df_rl,
        df_csv,
        on=["experiment_id", "task", "step"],
        how="inner",
        suffixes=("_rl", "_csv"),
    )
    print(f"[INFO] Merged rows: {len(df)}")

    # Summary records for all runs
    summary_records: List[Dict[str, Any]] = []

    # Group by experiment_id, task
    grouped = df.groupby(["experiment_id", "task"])

    for (exp_id, task), df_run in grouped:
        df_run = df_run.sort_values("step").copy()

        # Drop rows where csv_score or rl_score is NaN
        df_run = df_run.replace([np.inf, -np.inf], np.nan)
        df_run = df_run.dropna(subset=["rl_score", args.csv_score_col])
        if len(df_run) < args.min_points:
            print(f"[WARN] Skip run (exp={exp_id}, task={task}) "
                  f"because valid points < min_points ({len(df_run)} < {args.min_points})")
            continue

        x = df_run[args.csv_score_col].to_numpy(dtype=float)
        y = df_run["rl_score"].to_numpy(dtype=float)

        r2, pearson_r, slope, intercept = fit_regression(x, y)

        summary_records.append({
            "experiment_id": exp_id,
            "task": task,
            "num_points": int(len(df_run)),
            "r2": r2,
            "pearson_r": pearson_r,
            "slope": slope,
            "intercept": intercept,
        })

        print(f"[RUN] exp={exp_id}, task={task}, "
              f"n={len(df_run)}, R2={r2:.4f}, r={pearson_r:.4f}, "
              f"slope={slope:.4f}, intercept={intercept:.4f}")

        # Draw and save plot
        plot_run(df_run, exp_id, task, args.csv_score_col, out_dir)

    # Save summary
    if summary_records:
        df_sum = pd.DataFrame.from_records(summary_records)
    else:
        df_sum = pd.DataFrame(columns=["experiment_id", "task", "num_points", "r2", "pearson_r", "slope", "intercept"])

    summary_path = out_dir / "summary.csv"
    df_sum.to_csv(summary_path, index=False)
    print(f"[OK] Wrote summary for {len(df_sum)} runs to {summary_path}")


if __name__ == "__main__":
    main()