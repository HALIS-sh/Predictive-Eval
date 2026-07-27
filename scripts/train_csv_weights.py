#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train simple Capability Salience "weights" and map token-level loss to per-task capability scores.

概念上，这个脚本做三件事：
  1) 读取多个模型在 probe 上的推断结果 (run_inference.py 生成的 *.probe.parquet)
  2) 利用 learn_task_vocab.py 学到的 per-task vocab，把 token-level NLL 聚合成一批特征
  3) 读取真实 benchmark 分数，对每个任务训练一个简单的回归模型：
       features -> benchmark_score
     拟合出来的映射，就相当于是最简版的 CSV 能力轴。

输入：
  - 多个模型的 probe 推断结果 parquet 文件
  - 一个 task vocab JSON（learn_task_vocab.py 的输出）
  - 一个 benchmark CSV：记录每个 (model, task) 的真实分数

输出：
  1) capability 表：capability.parquet
     每行：(model_name, task) + 一堆特征 + 回归得到的 predicted_score + 真实 score(如有)
  2) 回归参数：csv_fits.json
     保存每个 task 的特征名、回归权重、intercept 以及训练上的 MAE/RMSE/R^2

Usage 示例：
  python scripts/train_csv_weights.py \
    --infer_paths \
      /data/.../Qwen2.5-1.5B-Instruct.probe.parquet \
      /data/.../AnotherModel.probe.parquet \
    --model_names Qwen2.5-1.5B-Instruct AnotherModel \
    --vocab_path /data/.../Qwen2.5-1.5B-Instruct.vocab.json \
    --bench_path /data/.../benchmarks.csv \
    --capability_out /data/.../csv/capability.parquet \
    --fit_out /data/.../csv/csv_fits.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------- CLI -------------------- #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infer_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of parquet paths from run_inference.py, one per model.",
    )
    ap.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names (same length as infer_paths).",
    )
    ap.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Task vocab JSON from learn_task_vocab.py",
    )
    ap.add_argument(
        "--bench_path",
        type=str,
        required=True,
        help="Benchmark CSV file: columns = [model_name, task, score]",
    )
    ap.add_argument(
        "--capability_out",
        type=str,
        required=True,
        help="Output parquet path for per-model per-task capability features.",
    )
    ap.add_argument(
        "--fit_out",
        type=str,
        required=True,
        help="Output JSON path for per-task regression fits.",
    )
    ap.add_argument(
        "--use_answer_only",
        action="store_true",
        help="If set,子特征会优先基于答案部分 (is_answer_mask=1) 计算（必要时无答案则回退）。",
    )
    ap.add_argument(
        "--min_tokens_for_feature",
        type=int,
        default=5,
        help="计算某个子特征（如 vocab 子集）时，至少需要多少个 token 才认为有效，否则该特征为 NaN。",
    )
    return ap.parse_args()


# -------------------- 特征提取核心逻辑 -------------------- #

def safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if not pd.isna(x)]
    if len(xs) == 0:
        return float("nan")
    return float(sum(xs) / len(xs))

def fit_sigmoid_capability(
    X: np.ndarray,
    y: np.ndarray,
    num_steps: int = 4000,
    lr: float = 1e-2,
    l2: float = 1e-4,
) -> Dict[str, Any]:
    """
    拟合 paper 形式的两步模型：
      C = w^T x
      y_hat = gamma + (1-gamma) / (1 + exp(-alpha * (C - beta)))

    输入:
      X: (n_samples, n_features)
      y: (n_samples,) 真实 accuracy (一般在 [0,1])

    输出:
      dict 包含:
        - w, alpha, beta, gamma
        - x_mean, x_std (用于后续预测时做同样的标准化)
        - train_pred, train_loss 最终训练效果
    """
    n, d = X.shape

    # 标准化特征：有助于优化稳定
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0
    Xn = (X - x_mean) / x_std

    # 初始化参数
    w = np.zeros(d, dtype=np.float64)
    # alpha 控制斜率，beta 是拐点位置，gamma 是最低精度
    alpha = 1.0
    beta = 0.0
    # gamma 初始取 y 的最小值，比较合理
    gamma = float(np.clip(y.min() - 0.05, 0.0, 1.0))

    for step in range(num_steps):
        # 能力分数 C
        C = Xn @ w  # (n,)

        # 为了数值稳定，clip 一下 z
        z = alpha * (C - beta)
        z = np.clip(z, -20.0, 20.0)

        sig = 1.0 / (1.0 + np.exp(-z))  # (n,)
        y_hat = gamma + (1.0 - gamma) * sig

        # 均方误差
        diff = y_hat - y  # (n,)
        loss = float((diff ** 2).mean() + l2 * (w @ w))

        # 如果需要调试可以每隔一段打印 loss
        # if step % 500 == 0:
        #     print(f"  [DEBUG] step={step}, loss={loss:.6f}")

        # 反向传播：dL/dparam
        # e = dL/dy_hat
        e = 2.0 * diff / n  # (n,)

        dy_dgamma = 1.0 - sig              # (n,)
        dy_dsig = (1.0 - gamma)           # 标量
        dsig_dz = sig * (1.0 - sig)       # (n,)
        dz_dalpha = C - beta              # (n,)
        dz_dbeta = -alpha                 # 标量
        dz_dC = alpha                     # 标量

        # dL/dgamma
        grad_gamma = float((e * dy_dgamma).sum())

        # dL/dalpha
        grad_alpha = float((e * dy_dsig * dsig_dz * dz_dalpha).sum())

        # dL/dbeta
        grad_beta = float((e * dy_dsig * dsig_dz * dz_dbeta).sum())

        # dL/dC
        grad_C = e * dy_dsig * dsig_dz * dz_dC  # (n,)

        # dL/dw = Xn^T grad_C + 2*l2*w
        grad_w = Xn.T @ grad_C + 2.0 * l2 * w   # (d,)

        # 参数更新（简单 SGD）
        w -= lr * grad_w
        alpha -= lr * grad_alpha
        beta  -= lr * grad_beta
        gamma -= lr * grad_gamma

    # 结束后再算一遍最终预测
    C = Xn @ w
    z = np.clip(alpha * (C - beta), -20.0, 20.0)
    sig = 1.0 / (1.0 + np.exp(-z))
    y_hat = gamma + (1.0 - gamma) * sig
    final_loss = float(((y_hat - y) ** 2).mean())

    return {
        "w": w.tolist(),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "x_mean": x_mean.flatten().tolist(),
        "x_std": x_std.flatten().tolist(),
        "train_pred": y_hat.tolist(),
        "train_loss": final_loss,
    }


def sigmoid_predict(
    X: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    使用上面 fit 出来的参数在任意 X 上做预测。
    """
    w = np.array(params["w"], dtype=np.float64)
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    gamma = float(params["gamma"])
    x_mean = np.array(params["x_mean"], dtype=np.float64).reshape(1, -1)
    x_std = np.array(params["x_std"], dtype=np.float64).reshape(1, -1)
    x_std[x_std == 0] = 1.0

    Xn = (X - x_mean) / x_std
    C = Xn @ w
    z = np.clip(alpha * (C - beta), -20.0, 20.0)
    sig = 1.0 / (1.0 + np.exp(-z))
    y_hat = gamma + (1.0 - gamma) * sig
    return y_hat





# def compute_task_features_for_model(
#     df_infer: pd.DataFrame,
#     model_name: str,
#     vocab: Dict[str, List[Dict[str, Any]]],
#     use_answer_only: bool,
#     min_tokens_for_feature: int,
# ) -> List[Dict[str, Any]]:
#     """
#     对单个模型的推断结果，按 task 聚合出一批特征：
#       - mean_nll: 样本级 mean_nll 的平均
#       - mean_nll_answer_only: 样本级 mean_nll_answer_only 的平均
#       - mean_nll_vocab_all: vocab token 上的平均 NLL
#       - mean_nll_vocab_answer: vocab token 且在答案部分的平均 NLL
#       - mean_nll_math_mask / code_mask / logic_mask / reason_mask (全局 & 答案部分)
#     返回: 每个 task 一行 dict
#     """
#     results = []

#     # 先拿到所有出现过的 task
#     tasks = sorted(df_infer["task"].dropna().unique().tolist())

#     # 需要 explode 的列（token级）
#     cols_to_explode = [
#         "tokens",
#         "nll",
#         "is_answer_mask",
#         "math_mask",
#         "code_mask",
#         "logic_mask",
#         "tool_mask",
#         "reason_mask",
#     ]

#     for task in tasks:
#         df_task = df_infer[df_infer["task"] == task].copy()
#         if df_task.empty:
#             continue

#         # 1) 样本级 mean_nll/mean_nll_answer_only 简单平均
#         mean_nll = safe_mean(df_task["mean_nll"].tolist())
#         mean_nll_ans = safe_mean(df_task["mean_nll_answer_only"].tolist())

#         # 2) token 级展开
#         # 确保每列是 list，然后 explode
#         for c in cols_to_explode:
#             df_task[c] = df_task[c].apply(list)

#         tok_df = df_task[cols_to_explode].explode(cols_to_explode, ignore_index=True)

#         # 类型/清理
#         tok_df["tokens"] = tok_df["tokens"].astype(str)
#         tok_df["nll"] = pd.to_numeric(tok_df["nll"], errors="coerce")
#         tok_df["is_answer_mask"] = tok_df["is_answer_mask"].astype(int)
#         tok_df["math_mask"] = tok_df["math_mask"].astype(int)
#         tok_df["code_mask"] = tok_df["code_mask"].astype(int)
#         tok_df["logic_mask"] = tok_df["logic_mask"].astype(int)
#         tok_df["reason_mask"] = tok_df["reason_mask"].astype(int)
#         # tool_mask 暂时不单独用，如果需要可以加
#         tok_df = tok_df.dropna(subset=["nll"])

#         # answer mask
#         if use_answer_only:
#             sub_all = tok_df  # 备份
#             tok_df_ans = tok_df[tok_df["is_answer_mask"] == 1]
#             if tok_df_ans.empty:
#                 # 若答案部分没有 token，则回退到全体
#                 tok_df_ans = sub_all
#         else:
#             tok_df_ans = tok_df

#         # 3) vocab-based 特征（只在该 task 有 vocab 配置时计算）
#         vocab_tokens = set()
#         if task in vocab:
#             vocab_tokens = {rec["token"] for rec in vocab[task]}
#         # vocab 全体 (不管是否答案部分)
#         if vocab_tokens:
#             mask_vocab_all = tok_df["tokens"].isin(vocab_tokens)
#             tok_vocab_all = tok_df[mask_vocab_all]
#             if len(tok_vocab_all) >= min_tokens_for_feature:
#                 mean_nll_vocab_all = float(tok_vocab_all["nll"].mean())
#             else:
#                 mean_nll_vocab_all = float("nan")

#             # vocab + answer-only
#             mask_vocab_ans = tok_df_ans["tokens"].isin(vocab_tokens)
#             tok_vocab_ans = tok_df_ans[mask_vocab_ans]
#             if len(tok_vocab_ans) >= min_tokens_for_feature:
#                 mean_nll_vocab_ans = float(tok_vocab_ans["nll"].mean())
#             else:
#                 mean_nll_vocab_ans = float("nan")
#         else:
#             mean_nll_vocab_all = float("nan")
#             mean_nll_vocab_ans = float("nan")

#         # 4) mask-based 特征（math/code/logic/reason）
#         def masked_loss(df_tok: pd.DataFrame, mask_col: str, answer_only: bool) -> float:
#             df_ = df_tok
#             if answer_only:
#                 df_ = df_[df_["is_answer_mask"] == 1]
#             df_ = df_[df_[mask_col] == 1]
#             if len(df_) < min_tokens_for_feature:
#                 return float("nan")
#             return float(df_["nll"].mean())

#         feat = {
#             "model_name": model_name,
#             "task": task,
#             "mean_nll": float(mean_nll),
#             "mean_nll_answer_only": float(mean_nll_ans),
#             "mean_nll_vocab_all": float(mean_nll_vocab_all),
#             "mean_nll_vocab_answer": float(mean_nll_vocab_ans),
#             # mask-based
#             "mean_nll_math_all": float(masked_loss(tok_df, "math_mask", answer_only=False)),
#             "mean_nll_math_answer": float(masked_loss(tok_df, "math_mask", answer_only=True)),
#             "mean_nll_code_all": float(masked_loss(tok_df, "code_mask", answer_only=False)),
#             "mean_nll_code_answer": float(masked_loss(tok_df, "code_mask", answer_only=True)),
#             "mean_nll_logic_all": float(masked_loss(tok_df, "logic_mask", answer_only=False)),
#             "mean_nll_logic_answer": float(masked_loss(tok_df, "logic_mask", answer_only=True)),
#             "mean_nll_reason_all": float(masked_loss(tok_df, "reason_mask", answer_only=False)),
#             "mean_nll_reason_answer": float(masked_loss(tok_df, "reason_mask", answer_only=True)),
#         }
#         results.append(feat)

#     return results

def compute_task_features_for_model(
    df_infer: pd.DataFrame,
    model_name: str,
    vocab: Dict[str, List[Dict[str, Any]]],
    use_answer_only: bool,
    min_tokens_for_feature: int,
) -> List[Dict[str, Any]]:
    """
    对单个模型的推断结果，按 (task, dataset) 细粒度聚合出一批特征。

    - df_infer 里:
        task   = 粗粒度能力轴 (如 "math")
        dataset= 数据集名 (如 "gsm8k", "aime24", "minerva" ...)
    - 我们导出到 CSV 的 task 名 = f"{dataset}_{task}"
      例如: "gsm8k_math", "aime24_math" ...

    vocab 仍然按粗粒度 task 查，比如 vocab["math"]。
    """
    results = []

    cols_to_explode = [
        "tokens",
        "nll",
        "is_answer_mask",
        "math_mask",
        "code_mask",
        "logic_mask",
        "tool_mask",
        "reason_mask",
    ]

    # 按 (task, dataset) 分组
    grouped = df_infer.groupby(["task", "dataset"], dropna=False)

    for (task, dataset), df_task in grouped:
        if df_task.empty:
            continue

        # 用于和 benchmark 对齐的细粒度 task 名
        # 例如 dataset="gsm8k", task="math" -> "gsm8k_math"
        dataset_str = str(dataset) if pd.notna(dataset) else "unknown"
        task_for_csv = f"{dataset_str}_{task}" if task else dataset_str

        # 1) 样本级 mean_nll/mean_nll_answer_only
        mean_nll = safe_mean(df_task["mean_nll"].tolist())
        mean_nll_ans = safe_mean(df_task["mean_nll_answer_only"].tolist())

        # 2) token 级展开
        for c in cols_to_explode:
            df_task[c] = df_task[c].apply(list)

        tok_df = df_task[cols_to_explode].explode(cols_to_explode, ignore_index=True)

        tok_df["tokens"] = tok_df["tokens"].astype(str)
        tok_df["nll"] = pd.to_numeric(tok_df["nll"], errors="coerce")
        tok_df["is_answer_mask"] = tok_df["is_answer_mask"].astype(int)
        tok_df["math_mask"] = tok_df["math_mask"].astype(int)
        tok_df["code_mask"] = tok_df["code_mask"].astype(int)
        tok_df["logic_mask"] = tok_df["logic_mask"].astype(int)
        tok_df["reason_mask"] = tok_df["reason_mask"].astype(int)
        tok_df = tok_df.dropna(subset=["nll"])

        # answer only 选择
        if use_answer_only:
            sub_all = tok_df
            tok_df_ans = tok_df[tok_df["is_answer_mask"] == 1]
            if tok_df_ans.empty:
                tok_df_ans = sub_all
        else:
            tok_df_ans = tok_df

        # 3) vocab-based 特征：仍然按粗粒度 task 查 vocab
        vocab_tokens = set()
        if task in vocab:
            vocab_tokens = {rec["token"] for rec in vocab[task]}

        if vocab_tokens:
            mask_vocab_all = tok_df["tokens"].isin(vocab_tokens)
            tok_vocab_all = tok_df[mask_vocab_all]
            if len(tok_vocab_all) >= min_tokens_for_feature:
                mean_nll_vocab_all = float(tok_vocab_all["nll"].mean())
            else:
                mean_nll_vocab_all = float("nan")

            mask_vocab_ans = tok_df_ans["tokens"].isin(vocab_tokens)
            tok_vocab_ans = tok_df_ans[mask_vocab_ans]
            if len(tok_vocab_ans) >= min_tokens_for_feature:
                mean_nll_vocab_ans = float(tok_vocab_ans["nll"].mean())
            else:
                mean_nll_vocab_ans = float("nan")
        else:
            mean_nll_vocab_all = float("nan")
            mean_nll_vocab_ans = float("nan")

        # 4) mask-based 特征
        def masked_loss(df_tok: pd.DataFrame, mask_col: str, answer_only: bool) -> float:
            df_ = df_tok
            if answer_only:
                df_ = df_[df_["is_answer_mask"] == 1]
            df_ = df_[df_[mask_col] == 1]
            if len(df_) < min_tokens_for_feature:
                return float("nan")
            return float(df_["nll"].mean())

        feat = {
            "model_name": model_name,
            # 用于和 benchmark join 的细粒度 task
            "task": task_for_csv,
            # 方便以后分析用的原始字段
            "base_task": task,
            "dataset": dataset_str,

            "mean_nll": float(mean_nll),
            "mean_nll_answer_only": float(mean_nll_ans),
            "mean_nll_vocab_all": float(mean_nll_vocab_all),
            "mean_nll_vocab_answer": float(mean_nll_vocab_ans),

            "mean_nll_math_all": float(masked_loss(tok_df, "math_mask", answer_only=False)),
            "mean_nll_math_answer": float(masked_loss(tok_df, "math_mask", answer_only=True)),
            "mean_nll_code_all": float(masked_loss(tok_df, "code_mask", answer_only=False)),
            "mean_nll_code_answer": float(masked_loss(tok_df, "code_mask", answer_only=True)),
            "mean_nll_logic_all": float(masked_loss(tok_df, "logic_mask", answer_only=False)),
            "mean_nll_logic_answer": float(masked_loss(tok_df, "logic_mask", answer_only=True)),
            "mean_nll_reason_all": float(masked_loss(tok_df, "reason_mask", answer_only=False)),
            "mean_nll_reason_answer": float(masked_loss(tok_df, "reason_mask", answer_only=True)),
        }
        results.append(feat)

    return results


# -------------------- 训练 per-task 回归 -------------------- #

# def train_per_task_regression(
#     feat_df: pd.DataFrame,
#     bench_df: pd.DataFrame,
#     fit_out: Path,
# ) -> Dict[str, Any]:
#     """
#     对每个 task 做一个简单线性回归：
#         X = 一组特征（可多维）
#         y = benchmark score
#     并将拟合结果保存到 JSON，同时返回全表带 predicted_score 的副本。
#     """
#     # 合并能力特征与 benchmark 分数：
#     # feat_df: [model_name, task, feat1, feat2, ...]
#     # bench_df: [model_name, task, score]
#     merged = feat_df.merge(bench_df, on=["model_name", "task"], how="inner")
#     if merged.empty:
#         raise RuntimeError("No overlapping (model_name, task) between features and benchmark.")

#     # 选择要用的特征列（可以根据需要扩展/修改）
#     feature_cols = [
#         "mean_nll",
#         "mean_nll_answer_only",
#         "mean_nll_vocab_all",
#         "mean_nll_vocab_answer",
#         "mean_nll_math_all",
#         "mean_nll_math_answer",
#         "mean_nll_code_all",
#         "mean_nll_code_answer",
#         "mean_nll_logic_all",
#         "mean_nll_logic_answer",
#         "mean_nll_reason_all",
#         "mean_nll_reason_answer",
#     ]

#     fit_info: Dict[str, Any] = {}

#     preds_all = []

#     for task, sub in merged.groupby("task"):
#         # 过滤掉全 NaN 的行
#         X = sub[feature_cols].values.astype(float)
#         y = sub["score"].values.astype(float)

#         # 去掉存在 NaN 的样本（非常重要）
#         valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
#         X_valid = X[valid_mask]
#         y_valid = y[valid_mask]
#         if len(X_valid) < 2:
#             print(f"[WARN] Task {task}: not enough valid samples ({len(X_valid)}) to fit regression, skip.")
#             continue

#         reg = LinearRegression()
#         reg.fit(X_valid, y_valid)
#         y_pred = reg.predict(X_valid)

#         mae = float(mean_absolute_error(y_valid, y_pred))
#         rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
#         r2 = float(r2_score(y_valid, y_pred))

#         print(f"[INFO] Task={task}: n={len(X_valid)}, MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

#         fit_info[task] = {
#             "features": feature_cols,
#             "coef": reg.coef_.tolist(),
#             "intercept": float(reg.intercept_),
#             "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
#         }

#         # 在该 task 的所有样本上打 predicted_score（包括那些有 NaN 特征的，直接设为 NaN）
#         y_pred_all = np.full(len(sub), np.nan, dtype=float)
#         valid_idx = np.where(valid_mask)[0]
#         y_pred_all[valid_idx] = reg.predict(X_valid)

#         tmp = sub[["model_name", "task"]].copy()
#         tmp["true_score"] = y
#         tmp["pred_score"] = y_pred_all
#         preds_all.append(tmp)

#     # 汇总所有 task 的预测结果
#     if preds_all:
#         preds_df = pd.concat(preds_all, ignore_index=True)
#     else:
#         preds_df = pd.DataFrame(columns=["model_name", "task", "true_score", "pred_score"])

#     # 保存 fit 信息
#     fit_out.parent.mkdir(parents=True, exist_ok=True)
#     with open(fit_out, "w", encoding="utf-8") as f:
#         json.dump(fit_info, f, ensure_ascii=False, indent=2)
#     print(f"[OK] Saved per-task CSV fit params to {fit_out}")

#     return {"preds_df": preds_df, "fit_info": fit_info}


def train_per_task_regression(
    feat_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    fit_out: Path,
) -> Dict[str, Any]:
    """
    对每个细粒度 task 做一次 "完整 CSV" 拟合：

      1) C = w^T x   （能力分数）
      2) A_hat = gamma + (1-gamma) / (1 + exp(-alpha * (C - beta)))

    通过梯度下降联合优化 {w, alpha, beta, gamma}，
    并计算训练上的 MAE / RMSE / R^2 作为拟合质量指标。
    """
    # 合并能力特征与 benchmark 分数：
    merged = feat_df.merge(bench_df, on=["model_name", "task"], how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping (model_name, task) between features and benchmark.")

    # 选择要用的特征列
    feature_cols = [
        "mean_nll",
        "mean_nll_answer_only",
        "mean_nll_vocab_all",
        "mean_nll_vocab_answer",
        "mean_nll_math_all",
        "mean_nll_math_answer",
        "mean_nll_code_all",
        "mean_nll_code_answer",
        "mean_nll_logic_all",
        "mean_nll_logic_answer",
        "mean_nll_reason_all",
        "mean_nll_reason_answer",
    ]


    # feature_cols = [
    #     "mean_nll",
    #     "mean_nll_answer_only",
    #     "mean_nll_vocab_all",
    #     "mean_nll_vocab_answer",
    #     "mean_nll_math_all",
    #     "mean_nll_math_answer"
    # ]

    fit_info: Dict[str, Any] = {}
    preds_all = []

    for task, sub in merged.groupby("task"):
        # 先选出这一 task 的特征 DataFrame
        feat_sub = sub[feature_cols].copy()

        # 1) 删掉在该 task 上「整列都是 NaN」的特征
        #    （一定要在 DataFrame 上做，不能在 numpy 上做）
        non_all_nan_mask = ~feat_sub.isna().all(axis=0)
        used_feature_cols = feat_sub.columns[non_all_nan_mask].tolist()
        feat_sub = feat_sub[used_feature_cols]

        if len(used_feature_cols) == 0:
            print(f"[WARN] Task {task}: all feature columns are NaN, skip.")
            continue

        # 2) 转成 numpy
        X = feat_sub.values.astype(float)
        y = sub["score"].values.astype(float)

        # 去掉存在 NaN 的样本
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        if len(X_valid) < 2:
            print(f"[WARN] Task {task}: not enough valid samples ({len(X_valid)}) to fit regression, skip.")
            continue

        # 用 sigmoid + capability 模型拟合
        params = fit_sigmoid_capability(
            X_valid,
            y_valid,
            num_steps=4000,
            lr=1e-2,
            l2=1e-4,
        )

        # 在 valid 样本上预测
        y_pred_valid = sigmoid_predict(X_valid, params)

        mae = float(mean_absolute_error(y_valid, y_pred_valid))
        rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred_valid)))
        r2 = float(r2_score(y_valid, y_pred_valid))

        print(f"[INFO] Task={task}: n={len(X_valid)}, MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        # 保存拟合参数和指标（包含 w, alpha, beta, gamma）
        fit_info[task] = {
            "features": feature_cols,
            "capability_params": {
                "w": params["w"],
                "alpha": params["alpha"],
                "beta": params["beta"],
                "gamma": params["gamma"],
                "x_mean": params["x_mean"],
                "x_std": params["x_std"],
            },
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "train_loss": params["train_loss"],
            },
        }

        # 在该 task 的所有样本上打 predicted_score
        y_pred_all = np.full(len(sub), np.nan, dtype=float)
        valid_idx = np.where(valid_mask)[0]
        y_pred_all[valid_idx] = sigmoid_predict(X_valid, params)

        tmp = sub[["model_name", "task"]].copy()
        tmp["true_score"] = y
        tmp["pred_score"] = y_pred_all
        preds_all.append(tmp)

    # 汇总所有 task 的预测结果
    if preds_all:
        preds_df = pd.concat(preds_all, ignore_index=True)
    else:
        preds_df = pd.DataFrame(columns=["model_name", "task", "true_score", "pred_score"])

    # 保存 fit 信息
    fit_out.parent.mkdir(parents=True, exist_ok=True)
    with open(fit_out, "w", encoding="utf-8") as f:
        json.dump(fit_info, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved per-task CSV (sigmoid) fit params to {fit_out}")

    return {"preds_df": preds_df, "fit_info": fit_info}


# -------------------- 主函数 -------------------- #

def main():
    args = parse_args()

    infer_paths = args.infer_paths
    model_names = args.model_names
    if len(infer_paths) != len(model_names):
        raise ValueError("infer_paths and model_names must have the same length.")

    vocab_path = Path(args.vocab_path)
    bench_path = Path(args.bench_path)
    capability_out = Path(args.capability_out)
    fit_out = Path(args.fit_out)

    # ---- 读取 vocab ----
    print(f"[INFO] Loading task vocab from {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # ---- 读取 benchmark ----
    print(f"[INFO] Loading benchmark from {bench_path}")
    bench_df = pd.read_csv(bench_path)
    expected_cols = {"model_name", "task", "score"}
    if not expected_cols.issubset(set(bench_df.columns)):
        raise ValueError(f"Benchmark CSV must contain columns: {expected_cols}, got {bench_df.columns}")
    # 清理 task / model_name
    bench_df["model_name"] = bench_df["model_name"].astype(str)
    bench_df["task"] = bench_df["task"].astype(str)

    # ---- 读取每个模型的推断 parquet 并提取能力特征 ----
    all_feat_rows: List[Dict[str, Any]] = []

    for path, model_name in zip(infer_paths, model_names):
        p = Path(path)
        print(f"[INFO] Loading inference parquet for model '{model_name}': {p}")
        df = pd.read_parquet(p)

        # 提取任务级能力特征
        rows = compute_task_features_for_model(
            df_infer=df,
            model_name=model_name,
            vocab=vocab,
            use_answer_only=args.use_answer_only,
            min_tokens_for_feature=args.min_tokens_for_feature,
        )
        print(f"[INFO] Model '{model_name}': extracted features for {len(rows)} tasks.")
        all_feat_rows.extend(rows)

    feat_df = pd.DataFrame(all_feat_rows)
    if feat_df.empty:
        raise RuntimeError("No features extracted; check infer_paths and vocab/tasks alignment.")

    # ---- 训练 per-task 回归并得到 predicted_score ----
    res = train_per_task_regression(
        feat_df=feat_df,
        bench_df=bench_df,
        fit_out=fit_out,
    )
    preds_df = res["preds_df"]

    # ---- 将预测结果合并回 capability 表 ----
    cap_df = feat_df.merge(
        preds_df,
        on=["model_name", "task"],
        how="left",
    )

    capability_out.parent.mkdir(parents=True, exist_ok=True)
    cap_df.to_parquet(capability_out, index=False)
    print(f"[OK] Saved capability features & scores to {capability_out}")
    print("[DONE] train_csv_weights.py finished.")


if __name__ == "__main__":
    main()