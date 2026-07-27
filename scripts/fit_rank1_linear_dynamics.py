#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit linear dynamics of Rank-1 updates along RL steps, and report R^2.

This script takes the summary Parquet produced by extract_rank1_dynamics.py:
  rank1_dynamics.parquet

Columns expected (at minimum):
  - step (int)
  - layer_name (str)
  - fro_norm_rank1 (float)
  - s1 (float)
  - fro_norm_full (float)
  - rank1_fraction (float)
  - ...

For each layer_name, we:
  - Collect points (step, amplitude), where amplitude is one of:
        fro_norm_rank1  (default)
        s1
        fro_norm_full
        rank1_fraction
  - Fit a linear regression amplitude = a * step + b
  - Compute R^2
  - Record per-layer stats.

We also compute a global "average" dynamics:
  - For each step, average amplitude across all layers
  - Fit linear regression on these average amplitudes

Outputs:
  1) A per-layer summary Parquet:
       <output_dir>/rank1_linear_dynamics_layers.parquet
     Columns:
       [layer_name, n_points, amp_column, slope, intercept, r2,
        step_min, step_max, amp_min, amp_max]

  2) A global summary JSON (optional, we also print to stdout):
       <output_dir>/rank1_linear_dynamics_summary.json

Usage example:

  python scripts/fit_rank1_linear_dynamics.py \
    --rank1_dynamics_path /data/wenhesun/Predictive-Eval/data/alpharl/rank1_qwen3_8b_grpo/rank1_dynamics.parquet \
    --output_dir /data/wenhesun/Predictive-Eval/data/alpharl/linear_qwen3_8b_grpo \
    --amp_column fro_norm_rank1 \
    --min_points 3

"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def fit_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fit y = a * x + b with sklearn LinearRegression and return slope, intercept, r2.
    x: [N], y: [N]
    """
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    return {"slope": slope, "intercept": intercept, "r2": float(r2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rank1_dynamics_path",
        type=str,
        required=True,
        help="Path to rank1_dynamics.parquet produced by extract_rank1_dynamics.py",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write per-layer and global linear dynamics results",
    )
    ap.add_argument(
        "--amp_column",
        type=str,
        default="fro_norm_rank1",
        choices=["fro_norm_rank1", "s1", "fro_norm_full", "rank1_fraction"],
        help="Which scalar column to use as amplitude vs. step (default: fro_norm_rank1).",
    )
    ap.add_argument(
        "--min_points",
        type=int,
        default=3,
        help="Minimum number of (step, amplitude) points required per layer to fit a line.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- Load dynamics parquet -----------------
    df = pd.read_parquet(args.rank1_dynamics_path)
    if args.amp_column not in df.columns:
        raise ValueError(
            f"Column '{args.amp_column}' not found in {args.rank1_dynamics_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Remove rows with NaN in step or amplitude
    df = df.dropna(subset=["step", args.amp_column])
    # Make sure step is int
    df["step"] = df["step"].astype(int)
    print(f"[INFO] Loaded rank1_dynamics with {len(df)} rows. Using amplitude column: '{args.amp_column}'")

    # ----------------- Per-layer linear fits -----------------
    layer_records: List[Dict[str, Any]] = []

    grouped = df.groupby("layer_name")
    for layer_name, g in grouped:
        # each g: rows for one layer at different steps
        g_sorted = g.sort_values("step")
        steps = g_sorted["step"].to_numpy()
        amp = g_sorted[args.amp_column].to_numpy()

        if len(steps) < args.min_points:
            # skip if not enough points
            continue

        res = fit_linear(steps.astype(np.float64), amp.astype(np.float64))

        rec = {
            "layer_name": layer_name,
            "n_points": int(len(steps)),
            "amp_column": args.amp_column,
            "slope": res["slope"],
            "intercept": res["intercept"],
            "r2": res["r2"],
            "step_min": int(steps.min()),
            "step_max": int(steps.max()),
            "amp_min": float(amp.min()),
            "amp_max": float(amp.max()),
        }
        layer_records.append(rec)

    df_layers = pd.DataFrame.from_records(layer_records)
    if len(df_layers) == 0:
        raise RuntimeError("No layer had enough points to fit a line. Check min_points or your dynamics parquet.")

    layers_out_path = out_dir / "rank1_linear_dynamics_layers.parquet"
    df_layers.to_parquet(layers_out_path, index=False)
    print(f"[INFO] Wrote per-layer linear dynamics to {layers_out_path}")
    print(f"[INFO] Fitted {len(df_layers)} layers.")

    # ----------------- Global statistics -----------------
    # Overall stats on per-layer R^2
    r2_vals = df_layers["r2"].to_numpy()
    mean_r2 = float(np.mean(r2_vals))
    median_r2 = float(np.median(r2_vals))
    p90_r2 = float(np.percentile(r2_vals, 90))
    p95_r2 = float(np.percentile(r2_vals, 95))
    num_high_09 = int((r2_vals >= 0.9).sum())
    num_high_095 = int((r2_vals >= 0.95).sum())

    print("\n[STATS] Per-layer linear dynamics R^2 (amplitude = f(step))")
    print(f"  amp_column        : {args.amp_column}")
    print(f"  #layers fitted    : {len(df_layers)}")
    print(f"  mean R^2          : {mean_r2:.4f}")
    print(f"  median R^2        : {median_r2:.4f}")
    print(f"  90th pct R^2      : {p90_r2:.4f}")
    print(f"  95th pct R^2      : {p95_r2:.4f}")
    print(f"  #layers R^2>=0.90 : {num_high_09}")
    print(f"  #layers R^2>=0.95 : {num_high_095}")

    # ----------------- Global averaged dynamics over layers -----------------
    # For each step, compute mean amplitude across all layers
    # (only including layers that appear at that step)
    df_mean = (
        df.groupby("step")[args.amp_column]
        .mean()
        .reset_index()
        .rename(columns={args.amp_column: f"{args.amp_column}_mean"})
    )
    steps_mean = df_mean["step"].to_numpy().astype(np.float64)
    amp_mean = df_mean[f"{args.amp_column}_mean"].to_numpy().astype(np.float64)

    if len(steps_mean) >= args.min_points:
        res_global = fit_linear(steps_mean, amp_mean)
        print("\n[STATS] Global mean amplitude vs step (averaged across layers)")
        print(f"  #points           : {len(steps_mean)}")
        print(f"  slope             : {res_global['slope']:.6f}")
        print(f"  intercept         : {res_global['intercept']:.6f}")
        print(f"  R^2               : {res_global['r2']:.4f}")
    else:
        res_global = {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
        print("\n[WARN] Not enough points to fit global mean dynamics; skip.")

    # ----------------- Save global summary JSON -----------------
    summary = {
        "rank1_dynamics_path": str(Path(args.rank1_dynamics_path).resolve()),
        "output_dir": str(out_dir.resolve()),
        "amp_column": args.amp_column,
        "min_points": args.min_points,
        "num_layers_fitted": len(df_layers),
        "per_layer_r2_mean": mean_r2,
        "per_layer_r2_median": median_r2,
        "per_layer_r2_p90": p90_r2,
        "per_layer_r2_p95": p95_r2,
        "per_layer_num_r2_ge_0_90": num_high_09,
        "per_layer_num_r2_ge_0_95": num_high_095,
        "global_mean_slope": res_global["slope"],
        "global_mean_intercept": res_global["intercept"],
        "global_mean_r2": res_global["r2"],
    }

    summary_path = out_dir / "rank1_linear_dynamics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Wrote summary JSON to {summary_path}")


if __name__ == "__main__":
    main()