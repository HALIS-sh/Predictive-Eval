#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_behavior_predictor.py

用 (model, task) 级别的行为特征，拟合 benchmark 精度/质量分数。

输入：
  --behavior_path  data/behavior/behavior_model_task_feats.parquet
      每行: [model_name, task, ...各种行为特征...]

  --bench_path     data/eval/benchmarks_fine.csv
      每行: [model_name, task, score]
      score 可以是 acc，也可以是预先算好的 Δacc / 其它 scalar 质量指标

输出：
  1) behavior_score.parquet
     每行: [model_name, task, true_score, pred_score] + 行为特征（方便分析）
  2) behavior_fits.json
     保存全局回归的系数、截距以及 MAE/RMSE/R^2

用法示例：

  python scripts/train_behavior_predictor.py \
    --behavior_path data/behavior/behavior_model_task_feats.parquet \
    --bench_path data/eval/benchmarks_fine.csv \
    --out_parquet data/behavior/behavior_score.parquet \
    --out_fit_json data/behavior/behavior_fits.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--behavior_path",
        type=str,
        required=True,
        help="Parquet file with per-(model, task) behavior features.",
    )
    ap.add_argument(
        "--bench_path",
        type=str,
        required=True,
        help="CSV with columns [model_name, task, score].",
    )
    ap.add_argument(
        "--out_parquet",
        type=str,
        required=True,
        help="Output parquet with behavior-based predicted scores.",
    )
    ap.add_argument(
        "--out_fit_json",
        type=str,
        required=True,
        help="Output JSON with regression weights/metrics.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    behavior_path = Path(args.behavior_path)
    bench_path = Path(args.bench_path)
    out_parquet = Path(args.out_parquet)
    out_fit_json = Path(args.out_fit_json)

    # ---------- 1. 读入数据 ----------
    print(f"[INFO] Loading behavior features from {behavior_path}")
    beh_df = pd.read_parquet(behavior_path)

    print(f"[INFO] Loading benchmarks from {bench_path}")
    bench_df = pd.read_csv(bench_path)

    # 规范字段类型
    for col in ["model_name", "task"]:
        beh_df[col] = beh_df[col].astype(str)
        bench_df[col] = bench_df[col].astype(str)

    # ---------- 2. join 行为特征和分数 ----------
    merged = beh_df.merge(bench_df, on=["model_name", "task"], how="inner")
    if merged.empty:
        raise RuntimeError(
            "No overlapping (model_name, task) between behavior features "
            "and benchmark CSV. Check 'task' 命名是否一致（例如 aime24_math 等）。"
        )

    print(f"[INFO] Merged rows: {len(merged)}")

    # ---------- 3. 选择特征列 ----------
    # 行为表中除了 model_name/task 的其它列都可以当特征；
    # 如果你后面加了更多列，也会自动包含进来。
    exclude_cols = {"model_name", "task"}
    behavior_feature_cols: List[str] = [
        c for c in beh_df.columns if c not in exclude_cols
    ]

    print("[INFO] Behavior feature columns:")
    for c in behavior_feature_cols:
        print("    -", c)

    X = merged[behavior_feature_cols].astype(float)
    y = merged["score"].astype(float)

    # 去掉全 NaN 的特征列（比如 verify_late_ratio 在所有行都是 NaN 的情况）
    non_all_nan = ~X.isna().all(axis=0)
    X = X.loc[:, non_all_nan]
    behavior_feature_cols = list(X.columns)

    # 去掉含 NaN 的样本
    valid_mask = ~X.isna().any(axis=1) & ~y.isna()
    X_valid = X[valid_mask].values
    y_valid = y[valid_mask].values

    print(f"[INFO] Valid samples for regression: {len(y_valid)}")

    if len(y_valid) < 3:
        raise RuntimeError("Not enough valid samples (<3) to train a global regression.")

    # ---------- 4. 训练线性回归（行为 → score） ----------
    reg = LinearRegression()
    reg.fit(X_valid, y_valid)
    y_pred_valid = reg.predict(X_valid)

    mae = float(mean_absolute_error(y_valid, y_pred_valid))
    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred_valid)))
    r2 = float(r2_score(y_valid, y_pred_valid))

    print(f"[INFO] Global behavior regressor: n={len(y_valid)}, "
          f"MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # ---------- 5. 在所有样本上打预测 ----------
    y_pred_all = np.full(len(merged), np.nan, dtype=float)
    valid_idx = np.where(valid_mask.values)[0]
    y_pred_all[valid_idx] = reg.predict(X_valid)

    merged_out = merged.copy()
    merged_out["pred_score"] = y_pred_all
    merged_out = merged_out[["model_name", "task", "score", "pred_score"] + behavior_feature_cols]
    merged_out = merged_out.rename(columns={"score": "true_score"})

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged_out.to_parquet(out_parquet, index=False)
    print(f"[OK] Saved behavior-based scores to {out_parquet}")

    # ---------- 6. 保存回归参数 ----------
    fit_info: Dict[str, Any] = {
        "features": behavior_feature_cols,
        "coef": reg.coef_.tolist(),
        "intercept": float(reg.intercept_),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
    }

    out_fit_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fit_json, "w", encoding="utf-8") as f:
        json.dump(fit_info, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved behavior regression params to {out_fit_json}")
    print("[DONE] train_behavior_predictor.py finished.")


if __name__ == "__main__":
    main()