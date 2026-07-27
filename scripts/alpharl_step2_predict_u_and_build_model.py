#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Use Alpha-RL's PLS-based prediction to get a "predicted end"
rank-1 model, and evaluate it.

This script does:

  1) From SVD outputs (one per global_step), call
       first_vector.process_all_steps(...)
     to extract "first_u_vectors.pt" for each step.

  2) Read RL curve summary (typically alpharl_rank1_eval.parquet from
     alpharl_step1_svd_and_rank1.py), build y(step) = rl_score(step).

  3) Call pred.predict_all_keys_pls(...) to fit linear dynamics (PLS)
     on early steps [pls_start_step, pls_end_step], and predict an
     "u_pred" corresponding to target_y (target score). Save to
       pred/predicted_u.pt

  4) Call predupd.reconstruct_and_save_rank1_with_pred_u(...) to
     reconstruct a rank-1 model using predicted u, base model, and
     SVD components. This yields a new HF model directory
       ".../svd_components/<some_step>/rank1_predu" (or a subdir
     as defined inside Alpha-RL's implementation).

  5) Call Alpha-RL's data_eval.py to evaluate this predicted model,
     and compare with the true final RL checkpoint.

  6) Save a Parquet summary with columns:
       [experiment_id, base_model_name, algo, task,
        target_y, final_step, final_rl_score,
        predicted_score, predicted_model_path]

Usage example:

  python scripts/alpharl_step2_predict_u_and_build_model.py \
    --alpharl_root /data/wenhesun/Alpha-RL/Alpha-RL \
    --base_model_path /data/wenhesun/model/Qwen/Qwen3-8B \
    --ckpt_root /data/wenhesun/hf_ckpts/qwen3_8b_grpo \
    --output_root /data/wenhesun/alpharl_out/grpo_math_qwen3_8b \
    --summary_parquet /data/wenhesun/alpharl_out/grpo_math_qwen3_8b/alpharl_rank1_eval.parquet \
    --pls_start_step 100 \
    --pls_end_step 3100 \
    --target_score auto_final \
    --metric_key acc \
    --eval_extra_args "--dataset gsm8k --max_examples 500" \
    --experiment_id grpo_math_qwen3_8b \
    --base_model_name Qwen3-8B \
    --algo GRPO \
    --task math
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# -----------------------
# Helpers
# -----------------------

def ensure_in_syspath(path: str):
    """Make sure `path` is in sys.path (for importing Alpha-RL modules)."""
    p = os.path.abspath(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def run_eval(
    eval_script: str,
    model_path: str,
    metric_json_path: str,
    eval_extra_args: str = "",
    force: bool = False,
):
    """
    Call `python data_eval.py ...` as a subprocess.

    Assumes data_eval.py supports:
        --model_path <hf_dir>
        --output_path <metrics.json>
    """
    metric_json = Path(metric_json_path)
    if metric_json.exists() and not force:
        print(f"[INFO] Metrics already exist at {metric_json}, skip eval.")
        return

    cmd = ["python", eval_script, "--model_path", model_path, "--output_path", str(metric_json)]
    if eval_extra_args:
        cmd.extend(shlex.split(eval_extra_args))

    print(f"[INFO] Running eval: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_metric_from_json(json_path: str, metric_key: str) -> float:
    """
    From <metrics.json>, extract nested metric `metric_key` such as "acc"
    or "metrics.acc".
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {json_path}: {e}")
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
        print(f"[WARN] metric '{metric_key}' not float-like in {json_path}: {e}")
        return float("nan")


def find_rank1_predu_dir(svd_root: Path) -> Optional[Path]:
    """
    Try to find a unique directory named 'rank1_predu' under svd_root.
    If multiple found, pick the one under the largest step.

    Assumes pattern: svd_root/global_step_xxx/rank1_predu/ is where
    predupd.save wrote the predicted model.
    """
    candidates: List[Path] = list(svd_root.glob("global_step_*/rank1_predu"))
    if not candidates:
        print(f"[WARN] No rank1_predu dir found under {svd_root}")
        return None

    # If only one, just return
    if len(candidates) == 1:
        return candidates[0]

    # If multiple, pick the one with the largest step number
    def step_num(p: Path) -> int:
        name = p.parent.name  # global_step_xxx
        try:
            return int(name.split("_")[-1])
        except Exception:
            return -1

    best = max(candidates, key=step_num)
    return best


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()

    # Alpha-RL repo root
    ap.add_argument(
        "--alpharl_root",
        type=str,
        required=True,
        help="Path to Alpha-RL repo root (folder containing 'eval', 'data', 'utils').",
    )

    # Base model & RL trajectory
    ap.add_argument("--base_model_path", type=str, required=True,
                    help="HF path of base model (θ_0).")
    ap.add_argument("--ckpt_root", type=str, required=True,
                    help="Root dir of RL checkpoints with global_step_xxx subdirs.")

    # Output root (should be same root used in step1)
    ap.add_argument("--output_root", type=str, required=True,
                    help="Root dir that already contains SVD outputs (svd_components).")

    # RL curve from step1
    ap.add_argument("--summary_parquet", type=str, required=True,
                    help="Parquet from step1 (alpharl_rank1_eval.parquet).")

    # PLS fitting window
    ap.add_argument("--pls_start_step", type=int, required=True,
                    help="First step used for PLS fit (inclusive).")
    ap.add_argument("--pls_end_step", type=int, required=True,
                    help="Last step used for PLS fit (inclusive).")

    # Target score
    ap.add_argument(
        "--target_score",
        type=str,
        default="auto_final",
        help="Target score y* for PLS prediction. "
             "If 'auto_final', use the RL score at the largest step in summary. "
             "If 'max', use max rl_score in summary. "
             "Otherwise parse as float.",
    )

    # PLS hyper-params (pass-through to Alpha-RL.pred)
    ap.add_argument("--min_samples", type=int, default=3,
                    help="min_samples argument for predict_all_keys_pls (if present).")
    ap.add_argument("--n_components", type=int, default=1,
                    help="PLS n_components.")

    # Eval & metric
    ap.add_argument("--metric_key", type=str, default="acc",
                    help="Metric key in eval json, e.g. 'acc' or 'metrics.acc'.")
    ap.add_argument("--eval_extra_args", type=str, default="",
                    help="Extra CLI args forwarded to eval/data_eval.py.")
    ap.add_argument("--force_eval", action="store_true",
                    help="Re-run eval even if metrics json already exists.")

    # Metadata
    ap.add_argument("--experiment_id", type=str, default=None)
    ap.add_argument("--base_model_name", type=str, default=None)
    ap.add_argument("--algo", type=str, default=None)
    ap.add_argument("--task", type=str, default=None)

    # Output summary
    ap.add_argument("--summary_parquet_out", type=str, default=None,
                    help="Where to save summary Parquet for predicted model; "
                         "default: <output_root>/alpharl_predicted_end.parquet")

    args = ap.parse_args()

    alpharl_root = os.path.abspath(args.alpharl_root)
    alpharl_eval_dir = os.path.join(alpharl_root, "eval")

    if not os.path.isdir(alpharl_eval_dir):
        raise ValueError(f"alpharl_eval_dir not found: {alpharl_eval_dir}")

    # Make Alpha-RL modules importable
    ensure_in_syspath(alpharl_eval_dir)

    # Import Alpha-RL helpers
    from first_vector import process_all_steps
    from pred import predict_all_keys_pls
    from predupd import reconstruct_and_save_rank1_with_pred_u

    # Paths
    output_root = Path(args.output_root).absolute()
    svd_root = output_root / "svd_components"
    pred_root = output_root / "pred"
    pred_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_parquet(args.summary_parquet)
    if "step" not in summary_df.columns or "rl_score" not in summary_df.columns:
        raise ValueError(
            f"summary_parquet {args.summary_parquet} must contain columns ['step','rl_score'] "
            "from alpharl_step1_svd_and_rank1.py"
        )

    # -------------------------
    # 1) 构建 y(step) = rl_score
    # -------------------------
    # keep only rows with non-null rl_score, within [pls_start_step, pls_end_step]
    mask = (
        summary_df["rl_score"].notna()
        & (summary_df["step"] >= args.pls_start_step)
        & (summary_df["step"] <= args.pls_end_step)
    )
    df_pls = summary_df[mask].copy()
    if df_pls.empty:
        raise ValueError(
            f"No RL scores found in [{args.pls_start_step}, {args.pls_end_step}] "
            f"in {args.summary_parquet}"
        )

    # build y dict: {step: rl_score}
    y: Dict[int, float] = {int(row["step"]): float(row["rl_score"]) for _, row in df_pls.iterrows()}
    print(f"[INFO] PLS fit will use steps: {sorted(y.keys())}")
    print(f"[INFO] RL scores used for PLS: {y}")

    # -------------------------
    # 2) 选定 target_y
    # -------------------------
    if args.target_score == "auto_final":
        # choose rl_score at largest step in summary
        df_ok = summary_df[summary_df["rl_score"].notna()]
        final_row = df_ok.loc[df_ok["step"].idxmax()]
        target_y = float(final_row["rl_score"])
        final_step = int(final_row["step"])
        print(f"[INFO] target_score=auto_final -> step={final_step}, target_y={target_y:.4f}")
    elif args.target_score == "max":
        df_ok = summary_df[summary_df["rl_score"].notna()]
        max_row = df_ok.loc[df_ok["rl_score"].idxmax()]
        target_y = float(max_row["rl_score"])
        final_step = int(max_row["step"])
        print(f"[INFO] target_score=max -> step={final_step}, target_y={target_y:.4f}")
    else:
        target_y = float(args.target_score)
        # pick largest step in summary as "final"
        df_ok = summary_df[summary_df["rl_score"].notna()]
        final_step = int(df_ok["step"].max())
        print(f"[INFO] target_score={target_y:.4f} (user specified), final_step={final_step}")

    # 真实终点成绩（用于对比）
    true_final_row = summary_df[summary_df["step"] == final_step].iloc[0]
    true_final_score = float(true_final_row["rl_score"])

    # -------------------------
    # 3) 从 SVD 里抽取 first_u_vectors
    # -------------------------
    print("\n[PHASE 1] Extracting first_u_vectors from SVD components ...")
    # Alpha-RL 里的 process_all_steps 一般签名类似:
    #   process_all_steps(base_path, start, end)
    process_all_steps(
        base_path=str(svd_root),
        start=args.pls_start_step,
        end=final_step,
    )
    print("[PHASE 1] Done first_u extraction.")

    # -------------------------
    # 4) PLS 拟合 + 预测 u_pred
    # -------------------------
    print("\n[PHASE 2] Running PLS via Alpha-RL.pred.predict_all_keys_pls ...")

    predicted_u_file = pred_root / f"predicted_u_step{final_step}_target{target_y:.4f}.pt"

    # Alpha-RL 的 predict_all_keys_pls 在原仓库大致签名为：
    #   predict_all_keys_pls(base_path, y, target_y, filename, start_step, end_step,
    #                        save_file, min_samples=3, n_components=1, ...)
    predict_all_keys_pls(
        base_path=str(svd_root),
        y=y,
        target_y=target_y,
        filename="first_u_vectors.pt",
        start_step=args.pls_start_step,
        end_step=args.pls_end_step,
        save_file=str(predicted_u_file),
        min_samples=args.min_samples,
        n_components=args.n_components,
    )

    print(f"[PHASE 2] Saved predicted_u to {predicted_u_file}")

    # -------------------------
    # 5) 用 u_pred 重建 rank-1 模型
    # -------------------------
    print("\n[PHASE 3] Reconstruct rank-1 predicted-end model via Alpha-RL.predupd ...")

    reconstruct_and_save_rank1_with_pred_u(
        model1_path=args.base_model_path,
        model2_path=args.ckpt_root,
        svd_components_base_path=str(svd_root),
        predicted_u_file=str(predicted_u_file),
        output_subdir="rank1_predu",
    )

    # 找到 rank1_predu 目录
    predicted_model_dir = find_rank1_predu_dir(svd_root)
    if predicted_model_dir is None:
        raise RuntimeError(
            f"Failed to find rank1_predu directory under {svd_root}. "
            "Please check Alpha-RL.predupd implementation."
        )

    print(f"[PHASE 3] Predicted rank-1 model at: {predicted_model_dir}")

    # -------------------------
    # 6) 评测 predicted model
    # -------------------------
    print("\n[PHASE 4] Evaluating predicted-end model via data_eval.py ...")

    eval_script = os.path.join(alpharl_eval_dir, "data_eval.py")
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"data_eval.py not found at {eval_script}")

    metrics_dir = output_root / "metrics_predicted"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_metrics_json = metrics_dir / f"metrics_predicted_step{final_step}.json"

    run_eval(
        eval_script=eval_script,
        model_path=str(predicted_model_dir),
        metric_json_path=str(pred_metrics_json),
        eval_extra_args=args.eval_extra_args,
        force=args.force_eval,
    )

    predicted_score = extract_metric_from_json(str(pred_metrics_json), args.metric_key)
    print(f"[RESULT] Predicted-end model | {args.metric_key}={predicted_score}")

    # -------------------------
    # 7) 汇总并写 Parquet
    # -------------------------
    rec: Dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "base_model_name": args.base_model_name,
        "algo": args.algo,
        "task": args.task,
        "metric_key": args.metric_key,
        "target_y": target_y,
        "final_step": int(final_step),
        "true_final_score": true_final_score,
        "predicted_score": predicted_score,
        "predicted_model_path": str(predicted_model_dir),
        "pls_start_step": int(args.pls_start_step),
        "pls_end_step": int(args.pls_end_step),
        "min_samples": int(args.min_samples),
        "n_components": int(args.n_components),
    }

    df_out = pd.DataFrame([rec])

    if args.summary_parquet_out is None:
        summary_out = output_root / "alpharl_predicted_end.parquet"
    else:
        summary_out = Path(args.summary_parquet_out)

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(summary_out, index=False)
    print(f"\n[OK] Wrote Alpha-RL predicted-end summary to {summary_out}")


if __name__ == "__main__":
    main()