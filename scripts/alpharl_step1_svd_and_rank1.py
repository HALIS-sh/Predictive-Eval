#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Use Alpha-RL's SVD + Rank-1 reconstruction and evaluate.

This script does:

  1) Call `svd.save_svd_components` to compute SVD between base model
     and each RL checkpoint (global_step_xxx).
  2) Call `upd_rank.reconstruct_and_save_rank1` to reconstruct Rank-1
     approximations for each checkpoint, saved as HF models.
  3) For each step, run Alpha-RL's `data_eval.py` on:
       - the real RL checkpoint model
       - the Rank-1 reconstructed model
     and parse the chosen metric from the output JSON.
  4) Save a Parquet file summarizing the curves:
       [experiment_id, base_model_name, algo, task, step,
        rl_score, rank1_score, base_score, frac_recovered]

Usage example:

  python scripts/alpharl_step1_svd_and_rank1.py \
    --alpharl_root /data/wenhesun/Alpha-RL/Alpha-RL \
    --base_model_path /data/wenhesun/model/Qwen/Qwen3-8B \
    --ckpt_root /data/wenhesun/hf_ckpts/qwen3_8b_grpo \
    --output_root /data/wenhesun/alpharl_out/grpo_math_qwen3_8b \
    --start_step 100 \
    --end_step 3100 \
    --step_stride 500 \
    --eval_extra_args "--dataset gsm8k --max_examples 500" \
    --metric_key "acc" \
    --experiment_id grpo_math_qwen3_8b \
    --base_model_name Qwen3-8B \
    --algo GRPO \
    --task math
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pathlib import Path

def find_hf_model_dir_for_step(ckpt_root: str, step: int) -> Optional[str]:
    """
    在 ckpt_root 下为某个 step 找到真正的 HF 模型目录：
      ckpt_root/global_step_{step}[/huggingface_merged | /huggingface | /]
    里要能找到 pytorch_model*.bin 或 model*.safetensors 才算成功。
    """
    base = Path(ckpt_root) / f"global_step_{step}"

    if not base.is_dir():
        print(f"[WARN] step dir not found: {base}")
        return None

    candidates = [
        base / "huggingface_merged",
        base / "huggingface",
        base,
    ]

    for c in candidates:
        if not c.is_dir():
            continue
        has_bin = (c / "pytorch_model.bin").exists() or list(c.glob("pytorch_model-*.bin"))
        has_sft = (c / "model.safetensors").exists() or list(c.glob("model-*.safetensors"))
        if has_bin or has_sft:
            print(f"[INFO] step {step}: use HF model dir {c}")
            return str(c)

    print(f"[WARN] No HF weights found for step {step} under {base}")
    return None

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
    alpharl_root: str = None,
):
    """
    调用 Alpha-RL 的 data_eval.py 做评测。

    这里使用:
        python -m eval.data_eval \
            --model_name_or_path <model_path> \
            --output_dir <metric_json 所在目录> \
            [eval_extra_args...]

    注意：data_eval.py 本身没有 --output_path 这种参数，
    它会在 output_dir 下自己写结果文件。我们只是用 metric_json_path
    占一个“期望的结果路径”，方便后面 extract_metric_from_json；
    如果这个文件不存在，会返回 NaN，不会 crash。
    """
    from pathlib import Path
    import shlex
    import subprocess
    import os

    metric_json = Path(metric_json_path)

    # 如果已经有结果且不强制重算，直接跳过
    if metric_json.exists() and not force:
        print(f"[INFO] Metrics already exist at {metric_json}, skip eval.")
        return

    # data_eval 使用的是 output_dir，而不是 output_path
    output_dir = metric_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 在 Alpha-RL repo 下面以模块方式运行，保证 import utils 正常
    if alpharl_root is not None:
        cwd = alpharl_root
        cmd = [
            "python", "-m", "eval.data_eval",
            "--model_name_or_path", model_path,
            "--output_dir", str(output_dir),
        ]
    else:
        # 退路：直接跑脚本（一般用不到）
        cwd = None
        cmd = [
            "python", eval_script,
            "--model_name_or_path", model_path,
            "--output_dir", str(output_dir),
        ]

    # 追加用户传入的额外参数（必须是 data_eval 支持的）
    if eval_extra_args:
        cmd.extend(shlex.split(eval_extra_args))

    print(f"[INFO] Running eval: {' '.join(cmd)}  (cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=cwd)

# def run_eval(
#     eval_script: str,
#     model_path: str,
#     metric_json_path: str,
#     eval_extra_args: str = "",
#     force: bool = False,
# ):
#     """
#     Call `python data_eval.py ...` as a subprocess.

#     We assume Alpha-RL's `data_eval.py` supports at least:
#         --model_path <hf_dir>
#         --output_path <metrics.json>

#     Any extra CLI arguments should be passed via `eval_extra_args`
#     (e.g. "--dataset gsm8k --max_examples 500").
#     """
#     metric_json = Path(metric_json_path)
#     if metric_json.exists() and not force:
#         print(f"[INFO] Metrics already exist at {metric_json}, skip eval.")
#         return

#     cmd = ["python", eval_script, "--model_path", model_path, "--output_path", str(metric_json)]
#     if eval_extra_args:
#         cmd.extend(shlex.split(eval_extra_args))

#     print(f"[INFO] Running eval: {' '.join(cmd)}")
#     subprocess.run(cmd, check=True)


def extract_metric_from_json(json_path: str, metric_key: str) -> float:
    """
    From <metrics.json>, extract a nested metric given `metric_key` like "acc"
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


def auto_discover_steps(ckpt_root: str) -> List[int]:
    """
    If user doesn't specify step range, scan ckpt_root for global_step_* directories.
    """
    steps: List[int] = []
    for p in Path(ckpt_root).glob("global_step_*"):
        m = re.search(r"global_step_(\d+)", p.name)
        if m:
            steps.append(int(m.group(1)))
    steps = sorted(set(steps))
    if not steps:
        raise ValueError(f"No global_step_* found under {ckpt_root}")
    print(f"[INFO] auto-discovered steps: {steps}")
    return steps


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
        help="Path to Alpha-RL repo root (the folder containing 'eval', 'data', 'utils').",
    )

    # Base model & RL trajectory
    ap.add_argument("--base_model_path", type=str, required=True,
                    help="HF path of base model (θ_0).")
    ap.add_argument("--ckpt_root", type=str, required=True,
                    help="Root dir of RL checkpoints with global_step_xxx subdirs.")

    # Output dirs
    ap.add_argument("--output_root", type=str, required=True,
                    help="Root dir to save SVD, rank-1 models and metrics.")

    # Step range
    ap.add_argument("--start_step", type=int, default=None,
                    help="First step (inclusive). If None, auto-detect from ckpt_root.")
    ap.add_argument("--end_step", type=int, default=None,
                    help="Last step (inclusive). If None, auto-detect.")
    ap.add_argument("--step_stride", type=int, default=1,
                    help="Stride between steps (e.g., 1000 -> 1000,2000,...)")

    # Eval config
    ap.add_argument("--metric_key", type=str, default="acc",
                    help="Metric key in metrics.json (e.g., 'acc' or 'metrics.acc').")
    ap.add_argument("--eval_extra_args", type=str, default="",
                    help="Extra CLI args forwarded to data_eval.py.")
    ap.add_argument("--force_eval", action="store_true",
                    help="Re-run eval even if metrics.json already exists.")

    # Metadata for downstream analysis
    ap.add_argument("--experiment_id", type=str, default=None)
    ap.add_argument("--base_model_name", type=str, default=None)
    ap.add_argument("--algo", type=str, default=None)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--base_score", type=float, default=None,
                    help="Optional base model score; if set, will compute frac_recovered "
                         "= (rank1_score - base_score) / (rl_score - base_score).")

    # Output summary
    ap.add_argument("--summary_parquet", type=str, default=None,
                    help="Path to write summary Parquet; default: <output_root>/alpharl_rank1_eval.parquet")

    ap.add_argument("--skip_svd", action="store_true",
                help="Skip Phase 1 (SVD) even if no files.")
    ap.add_argument("--skip_rank1", action="store_true",
                help="Skip Phase 2 (Rank-1 reconstruction).")

    args = ap.parse_args()

    alpharl_root = os.path.abspath(args.alpharl_root)
    alpharl_eval_dir = os.path.join(alpharl_root, "eval")

    if not os.path.isdir(alpharl_eval_dir):
        raise ValueError(f"alpharl_eval_dir not found: {alpharl_eval_dir}")

    # Make Alpha-RL importable
    ensure_in_syspath(alpharl_eval_dir)

    # Import Alpha-RL modules
    from svd import save_svd_components
    from upd_rank import reconstruct_and_save_rank1

    # Paths
    output_root = Path(args.output_root).absolute()
    svd_out_dir = output_root / "svd_components"
    rank1_metrics_dir = output_root / "metrics_rank1"
    rl_metrics_dir = output_root / "metrics_rl"

    svd_out_dir.mkdir(parents=True, exist_ok=True)
    rank1_metrics_dir.mkdir(parents=True, exist_ok=True)
    rl_metrics_dir.mkdir(parents=True, exist_ok=True)

    # Step range
    if args.start_step is None or args.end_step is None:
        all_steps = auto_discover_steps(args.ckpt_root)
        start_step = all_steps[0] if args.start_step is None else args.start_step
        end_step = all_steps[-1] if args.end_step is None else args.end_step
    else:
        start_step = args.start_step
        end_step = args.end_step

    step_stride = max(1, args.step_stride)
    steps = list(range(start_step, end_step + 1, step_stride))
    print(f"[INFO] Will process steps: {steps}")

    # ---------------------
    # 1) SVD components
    # ---------------------
    if not args.skip_svd:
        print("\n[PHASE 1] Computing SVD components via Alpha-RL.svd ...")

        for step in steps:
            print(f"\n--- SVD for step {step} ---")
            model2_dir = find_hf_model_dir_for_step(args.ckpt_root, step)
            if model2_dir is None:
                print(f"[SKIP] step {step} due to missing HF model dir")
                continue

            step_out = svd_out_dir / f"global_step_{step}"
            step_out.mkdir(parents=True, exist_ok=True)

            try:
                # 注意这里 start_step=end_step=step，只做一个点
                save_svd_components(
                    model1_path=args.base_model_path,
                    model2_path=model2_dir,
                    base_output_path=str(step_out),
                    start_step=step,
                    end_step=step,
                )
            except Exception as e:
                print(f"[ERROR] SVD failed for step {step}: {e}")
                import traceback; traceback.print_exc()

        print("[PHASE 1] Done SVD.")
    else:
        print("\n[PHASE 1] Skipped (skip_svd=True)")

    # ---------------------
    # 2) Rank-1 reconstruction
    # ---------------------
    if not args.skip_rank1:
        print("\n[PHASE 2] Reconstructing Rank-1 models via Alpha-RL.upd_rank ...")

        for step in steps:
            print(f"\n--- Rank-1 reconstruction for step {step} ---")
            model2_dir = find_hf_model_dir_for_step(args.ckpt_root, step)
            if model2_dir is None:
                print(f"[SKIP] step {step} due to missing HF model dir")
                continue

            step_svd_out = svd_out_dir / f"global_step_{step}"

            # 尝试两种可能的 SVD 目录结构
            if (step_svd_out / "svd_components.pt").exists():
                svd_base_path = step_svd_out
            elif (step_svd_out / f"global_step_{step}" / "svd_components.pt").exists():
                svd_base_path = step_svd_out / f"global_step_{step}"
            else:
                print(f"[WARN] No svd_components.pt for step {step}, skip rank-1.")
                continue

            try:
                reconstruct_and_save_rank1(
                    model1_path=args.base_model_path,
                    model2_path=model2_dir,
                    svd_components_base_path=str(svd_base_path),
                    start_step=step,
                    end_step=step,
                )
            except Exception as e:
                print(f"[ERROR] Rank-1 reconstruction failed for step {step}: {e}")
                import traceback; traceback.print_exc()

        print("[PHASE 2] Done Rank-1 reconstruction.")
    else:
        print("\n[PHASE 2] Skipped (skip_rank1=True)")

    # data_eval.py 脚本路径
    eval_script = os.path.join(alpharl_eval_dir, "data_eval.py")
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"data_eval.py not found at {eval_script}")

    # ---------------------
    # 3) Evaluate RL vs Rank-1
    # ---------------------
    print("\n[PHASE 3] Evaluating real RL checkpoints and Rank-1 models ...")

    records: List[Dict[str, Any]] = []

    for step in steps:
        print(f"\n[STEP] step={step}")

        # 真实 RL ckpt：global_step_xxx
        # rl_model_path = os.path.join(args.ckpt_root, f"global_step_{step}")
        # if not os.path.isdir(rl_model_path):
        #     print(f"[WARN] RL model dir not found for step={step}: {rl_model_path}")
        #     rl_model_path = None

        # # Rank-1 模型：svd_components/global_step_xxx/rank_1
        # rank1_model_path = os.path.join(str(svd_out_dir), f"global_step_{step}", "rank_1")
        # if not os.path.isdir(rank1_model_path):
        #     print(f"[WARN] Rank-1 model dir not found for step={step}: {rank1_model_path}")
        #     rank1_model_path = None

        # 真实 RL：HF ckpt 目录
        rl_model_path = find_hf_model_dir_for_step(args.ckpt_root, step)

        # Rank-1：svd_components/global_step_xxx/rank_1（两种可能结构）
        rank1_model_path = None
        for p in [
            svd_out_dir / f"global_step_{step}" / "rank_1",
            svd_out_dir / f"global_step_{step}" / f"global_step_{step}" / "rank_1",
        ]:
            if p.is_dir():
                rank1_model_path = str(p)
                break

        rl_score = None
        rank1_score = None

        # ---- Eval RL model ----
        if rl_model_path is not None:
            rl_metric_json = rl_metrics_dir / f"metrics_rl_step_{step}.json"
            run_eval(
                eval_script=eval_script,
                model_path=rl_model_path,
                metric_json_path=str(rl_metric_json),
                eval_extra_args=args.eval_extra_args,
                force=args.force_eval,
                alpharl_root=alpharl_root,
            )
            rl_score = extract_metric_from_json(str(rl_metric_json), args.metric_key)
            print(f"[RESULT] RL step={step} | {args.metric_key}={rl_score}")

        # ---- Eval Rank-1 model ----
        if rank1_model_path is not None:
            rank1_metric_json = rank1_metrics_dir / f"metrics_rank1_step_{step}.json"
            run_eval(
                eval_script=eval_script,
                model_path=rank1_model_path,
                metric_json_path=str(rank1_metric_json),
                eval_extra_args=args.eval_extra_args,
                force=args.force_eval,
                alpharl_root=alpharl_root,
            )
            rank1_score = extract_metric_from_json(str(rank1_metric_json), args.metric_key)
            print(f"[RESULT] Rank-1 step={step} | {args.metric_key}={rank1_score}")

        # ---- 记录 ----
        rec: Dict[str, Any] = {
            "experiment_id": args.experiment_id,
            "base_model_name": args.base_model_name,
            "algo": args.algo,
            "task": args.task,
            "step": int(step),
            "metric_key": args.metric_key,
            "rl_model_path": rl_model_path,
            "rank1_model_path": rank1_model_path,
            "rl_score": rl_score,
            "rank1_score": rank1_score,
            "base_score": args.base_score,
        }

        frac_recovered = None
        if args.base_score is not None and rl_score is not None and rank1_score is not None:
            denom = rl_score - args.base_score
            if abs(denom) > 1e-8:
                frac_recovered = (rank1_score - args.base_score) / denom
        rec["frac_recovered"] = frac_recovered

        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # 默认 summary 路径
    if args.summary_parquet is None:
        summary_parquet = output_root / "alpharl_rank1_eval.parquet"
    else:
        summary_parquet = Path(args.summary_parquet)

    summary_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(summary_parquet, index=False)
    print(f"\n[OK] Wrote Alpha-RL rank1 eval summary ({len(df)} rows) to {summary_parquet}")


if __name__ == "__main__":
    main()






# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Step 1: Use Alpha-RL's SVD + Rank-1 reconstruction and evaluate.

# This script does:

#   1) Prepares a temporary Hugging Face model view for each verl checkpoint step.
#      (Uses symlinks to combine config files and model weights).
#   2) Call `svd.save_svd_components` to compute SVD.
#   3) Call `upd_rank.reconstruct_and_save_rank1` to reconstruct Rank-1 models.
#   4) For each step, run Alpha-RL's `data_eval.py` on:
#        - the real RL checkpoint model (via the prepared view)
#        - the Rank-1 reconstructed model
#   5) Save a Parquet file summarizing the curves.
# """

# import argparse
# import json
# import os
# import re
# import shlex
# import subprocess
# import sys
# import shutil
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# import pandas as pd


# # -----------------------
# # Helpers
# # -----------------------

# def ensure_in_syspath(path: str):
#     """Make sure `path` is in sys.path (for importing Alpha-RL modules)."""
#     p = os.path.abspath(path)
#     if p not in sys.path:
#         sys.path.insert(0, p)


# def run_eval(
#     eval_script: str,
#     model_path: str,
#     metric_json_path: str,
#     eval_extra_args: str = "",
#     force: bool = False,
# ):
#     """
#     Call `python data_eval.py ...` as a subprocess.
#     """
#     metric_json = Path(metric_json_path)
#     if metric_json.exists() and not force:
#         print(f"[INFO] Metrics already exist at {metric_json}, skip eval.")
#         return

#     cmd = ["python", eval_script, "--model_path", model_path, "--output_path", str(metric_json)]
#     if eval_extra_args:
#         cmd.extend(shlex.split(eval_extra_args))

#     print(f"[INFO] Running eval: {' '.join(cmd)}")
#     subprocess.run(cmd, check=True)


# def extract_metric_from_json(json_path: str, metric_key: str) -> float:
#     """
#     From <metrics.json>, extract a nested metric given `metric_key`.
#     """
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"[WARN] Failed to read {json_path}: {e}")
#         return float("nan")

#     cur: Any = data
#     for part in metric_key.split("."):
#         if isinstance(cur, dict) and part in cur:
#             cur = cur[part]
#         else:
#             print(f"[WARN] metric_key '{metric_key}' not found in {json_path}")
#             return float("nan")
#     try:
#         return float(cur)
#     except Exception as e:
#         print(f"[WARN] metric '{metric_key}' not float-like in {json_path}: {e}")
#         return float("nan")


# def auto_discover_steps(ckpt_root: str) -> List[int]:
#     """
#     If user doesn't specify step range, scan ckpt_root for global_step_* directories.
#     """
#     steps: List[int] = []
#     for p in Path(ckpt_root).glob("global_step_*"):
#         m = re.search(r"global_step_(\d+)", p.name)
#         if m:
#             steps.append(int(m.group(1)))
#     steps = sorted(set(steps))
#     if not steps:
#         raise ValueError(f"No global_step_* found under {ckpt_root}")
#     print(f"[INFO] auto-discovered steps: {steps}")
#     return steps


# def prepare_hf_view(ckpt_root: Path, step: int, output_root: Path) -> Optional[Path]:
#     """
#     Creates a temporary directory structure that looks like a Hugging Face model.
#     It symlinks the config files from 'actor/huggingface' and the weights from 'actor/'.
    
#     Returns the path to the specific step directory (e.g., .../temp_hf_models/global_step_100).
#     Also creates a recursive symlink inside to handle cases where scripts append 'global_step_X'.
#     """
#     step_dir = ckpt_root / f"global_step_{step}"
#     actor_dir = step_dir / "actor"
#     hf_config_dir = actor_dir / "huggingface"
    
#     if not hf_config_dir.exists():
#         print(f"[WARN] Config dir not found for step {step}: {hf_config_dir}")
#         return None

#     # Create temp root: output_root/temp_hf_models/global_step_{step}
#     temp_root = output_root / "temp_hf_models"
#     target_dir = temp_root / f"global_step_{step}"
    
#     if target_dir.exists():
#         shutil.rmtree(target_dir)
#     target_dir.mkdir(parents=True, exist_ok=True)

#     # 1. Symlink all config files (json, tokenizer, etc.) from actor/huggingface
#     for item in hf_config_dir.iterdir():
#         if item.is_file():
#             os.symlink(item, target_dir / item.name)

#     # 2. Symlink weights
#     if (hf_config_dir / "pytorch_model.bin").exists():
#         os.symlink(hf_config_dir / "pytorch_model.bin", target_dir / "pytorch_model.bin")
#     elif (hf_config_dir / "model.safetensors").exists():
#         os.symlink(hf_config_dir / "model.safetensors", target_dir / "model.safetensors")
#     else:
#         # Look in actor_dir (parent of hf_config_dir) for model_world_size_*_rank_0.pt
#         candidates = list(actor_dir.glob("model_world_size_*_rank_0.pt"))
#         if candidates:
#             src_bin = candidates[0]
#             print(f"[INFO] Linking {src_bin.name} to pytorch_model.bin for step {step}")
#             os.symlink(src_bin, target_dir / "pytorch_model.bin")
#         else:
#             print(f"[WARN] No model weights found in {actor_dir} for step {step}")
#             return None

#     # 3. Create recursive symlink: target_dir/global_step_{step} -> target_dir
#     # This ensures that if svd.py appends "global_step_{step}", it resolves to the same directory.
#     recursive_link = target_dir / f"global_step_{step}"
#     if not recursive_link.exists():
#         os.symlink(target_dir, recursive_link)

#     return target_dir


# # -----------------------
# # Main
# # -----------------------

# def main():
#     ap = argparse.ArgumentParser()
#     # Alpha-RL repo root
#     ap.add_argument(
#         "--alpharl_root",
#         type=str,
#         required=True,
#         help="Path to Alpha-RL repo root (the folder containing 'eval', 'data', 'utils').",
#     )

#     # Base model & RL trajectory
#     ap.add_argument("--base_model_path", type=str, required=True,
#                     help="HF path of base model (θ_0).")
#     ap.add_argument("--ckpt_root", type=str, required=True,
#                     help="Root dir of RL checkpoints with global_step_xxx subdirs.")

#     # Output dirs
#     ap.add_argument("--output_root", type=str, required=True,
#                     help="Root dir to save SVD, rank-1 models and metrics.")

#     # Step range
#     ap.add_argument("--start_step", type=int, default=None,
#                     help="First step (inclusive). If None, auto-detect from ckpt_root.")
#     ap.add_argument("--end_step", type=int, default=None,
#                     help="Last step (inclusive). If None, auto-detect.")
#     ap.add_argument("--step_stride", type=int, default=1,
#                     help="Stride between steps (e.g., 1000 -> 1000,2000,...)")

#     # Eval config
#     ap.add_argument("--metric_key", type=str, default="acc",
#                     help="Metric key in metrics.json (e.g., 'acc' or 'metrics.acc').")
#     ap.add_argument("--eval_extra_args", type=str, default="",
#                     help="Extra CLI args forwarded to data_eval.py.")
#     ap.add_argument("--force_eval", action="store_true",
#                     help="Re-run eval even if metrics.json already exists.")

#     # Metadata for downstream analysis
#     ap.add_argument("--experiment_id", type=str, default=None)
#     ap.add_argument("--base_model_name", type=str, default=None)
#     ap.add_argument("--algo", type=str, default=None)
#     ap.add_argument("--task", type=str, default=None)
#     ap.add_argument("--base_score", type=float, default=None,
#                     help="Optional base model score; if set, will compute frac_recovered "
#                          "= (rank1_score - base_score) / (rl_score - base_score).")

#     # Output summary
#     ap.add_argument("--summary_parquet", type=str, default=None,
#                     help="Path to write summary Parquet; default: <output_root>/alpharl_rank1_eval.parquet")

#     args = ap.parse_args()

#     alpharl_root = os.path.abspath(args.alpharl_root)
#     alpharl_eval_dir = os.path.join(alpharl_root, "eval")

#     if not os.path.isdir(alpharl_eval_dir):
#         raise ValueError(f"alpharl_eval_dir not found: {alpharl_eval_dir}")

#     # Make Alpha-RL importable
#     ensure_in_syspath(alpharl_eval_dir)

#     # Import Alpha-RL modules
#     from svd import save_svd_components
#     from upd_rank import reconstruct_and_save_rank1

#     # Paths
#     output_root = Path(args.output_root).absolute()
#     svd_out_dir = output_root / "svd_components"
#     rank1_metrics_dir = output_root / "metrics_rank1"
#     rl_metrics_dir = output_root / "metrics_rl"

#     svd_out_dir.mkdir(parents=True, exist_ok=True)
#     rank1_metrics_dir.mkdir(parents=True, exist_ok=True)
#     rl_metrics_dir.mkdir(parents=True, exist_ok=True)

#     # Step range
#     if args.start_step is None or args.end_step is None:
#         all_steps = auto_discover_steps(args.ckpt_root)
#         start_step = all_steps[0] if args.start_step is None else args.start_step
#         end_step = all_steps[-1] if args.end_step is None else args.end_step
#     else:
#         start_step = args.start_step
#         end_step = args.end_step

#     step_stride = max(1, args.step_stride)
#     steps = list(range(start_step, end_step + 1, step_stride))
#     print(f"[INFO] Will process steps: {steps}")

#     # ---------------------
#     # 1) SVD components
#     # ---------------------
#     print("\n[PHASE 1] Computing SVD components via Alpha-RL.svd ...")
    
#     for step in steps:
#         print(f"\n--- Processing SVD for step {step} ---")
        
#         # Prepare HF model view (symlinks)
#         # This returns .../temp_hf_models/global_step_{step}
#         temp_hf_model_dir = prepare_hf_view(Path(args.ckpt_root), step, output_root)
        
#         if temp_hf_model_dir is None:
#             print(f"[SKIP] Skipping step {step} due to model preparation failure.")
#             continue

#         step_svd_out = svd_out_dir / f"global_step_{step}"
#         step_svd_out.mkdir(parents=True, exist_ok=True)

#         try:
#             # We pass the specific directory as model2_path.
#             # Because we created a recursive symlink inside, it works whether svd.py
#             # uses the path directly OR appends "global_step_{step}".
#             save_svd_components(
#                 model1_path=args.base_model_path,
#                 model2_path=str(temp_hf_model_dir),
#                 base_output_path=str(step_svd_out),
#                 start_step=step,
#                 end_step=step,
#             )
#         except Exception as e:
#             print(f"[ERROR] Failed SVD for step {step}: {e}")
#             import traceback
#             traceback.print_exc()

#     print("[PHASE 1] Done SVD.")

#     # ---------------------
#     # 2) Rank-1 reconstruction
#     # ---------------------
#     print("\n[PHASE 2] Reconstructing Rank-1 models via Alpha-RL.upd_rank ...")
    
#     for step in steps:
#         print(f"\n--- Reconstructing Rank-1 for step {step} ---")
        
#         temp_hf_model_dir = prepare_hf_view(Path(args.ckpt_root), step, output_root)
#         if temp_hf_model_dir is None:
#             continue

#         step_svd_out = svd_out_dir / f"global_step_{step}"
        
#         # Determine where SVD components were actually saved
#         # They could be in step_svd_out OR step_svd_out/global_step_{step}
#         actual_svd_dir = step_svd_out
#         if (step_svd_out / f"global_step_{step}" / "svd_components.pt").exists():
#             actual_svd_dir = step_svd_out / f"global_step_{step}"
#         elif not (step_svd_out / "svd_components.pt").exists():
#              print(f"[WARN] SVD components missing for step {step}, skipping reconstruction.")
#              continue

#         # Create recursive symlink in actual_svd_dir as well, just in case reconstruct appends
#         recursive_svd_link = actual_svd_dir / f"global_step_{step}"
#         if not recursive_svd_link.exists():
#              os.symlink(actual_svd_dir, recursive_svd_link)

#         try:
#             reconstruct_and_save_rank1(
#                 model1_path=args.base_model_path,
#                 model2_path=str(temp_hf_model_dir),
#                 svd_components_base_path=str(actual_svd_dir),
#                 start_step=step,
#                 end_step=step,
#             )
#         except Exception as e:
#             print(f"[ERROR] Failed Rank-1 reconstruction for step {step}: {e}")
#             import traceback
#             traceback.print_exc()

#     print("[PHASE 2] Done Rank-1 reconstruction.")

#     # data_eval.py 脚本路径
#     eval_script = os.path.join(alpharl_eval_dir, "data_eval.py")
#     if not os.path.isfile(eval_script):
#         raise FileNotFoundError(f"data_eval.py not found at {eval_script}")

#     # ---------------------
#     # 3) Evaluate RL vs Rank-1
#     # ---------------------
#     print("\n[PHASE 3] Evaluating real RL checkpoints and Rank-1 models ...")

#     records: List[Dict[str, Any]] = []

#     for step in steps:
#         print(f"\n[STEP] step={step}")

#         # 真实 RL ckpt (HF format view)
#         temp_hf_model_dir = prepare_hf_view(Path(args.ckpt_root), step, output_root)
#         rl_model_path = temp_hf_model_dir

#         # Rank-1 模型：svd_components/global_step_xxx/rank_1
#         # Check both possible locations
#         rank1_model_path = None
#         possible_paths = [
#             svd_out_dir / f"global_step_{step}" / "rank_1",
#             svd_out_dir / f"global_step_{step}" / f"global_step_{step}" / "rank_1"
#         ]
#         for p in possible_paths:
#             if p.is_dir():
#                 rank1_model_path = str(p)
#                 break
        
#         if rank1_model_path is None:
#             print(f"[WARN] Rank-1 model dir not found for step={step}")

#         rl_score = None
#         rank1_score = None

#         # ---- Eval RL model ----
#         if rl_model_path is not None:
#             rl_metric_json = rl_metrics_dir / f"metrics_rl_step_{step}.json"
#             run_eval(
#                 eval_script=eval_script,
#                 model_path=str(rl_model_path),
#                 metric_json_path=str(rl_metric_json),
#                 eval_extra_args=args.eval_extra_args,
#                 force=args.force_eval,
#             )
#             rl_score = extract_metric_from_json(str(rl_metric_json), args.metric_key)
#             print(f"[RESULT] RL step={step} | {args.metric_key}={rl_score}")

#         # ---- Eval Rank-1 model ----
#         if rank1_model_path is not None:
#             rank1_metric_json = rank1_metrics_dir / f"metrics_rank1_step_{step}.json"
#             run_eval(
#                 eval_script=eval_script,
#                 model_path=rank1_model_path,
#                 metric_json_path=str(rank1_metric_json),
#                 eval_extra_args=args.eval_extra_args,
#                 force=args.force_eval,
#             )
#             rank1_score = extract_metric_from_json(str(rank1_metric_json), args.metric_key)
#             print(f"[RESULT] Rank-1 step={step} | {args.metric_key}={rank1_score}")

#         # ---- 记录 ----
#         rec: Dict[str, Any] = {
#             "experiment_id": args.experiment_id,
#             "base_model_name": args.base_model_name,
#             "algo": args.algo,
#             "task": args.task,
#             "step": int(step),
#             "metric_key": args.metric_key,
#             "rl_model_path": str(rl_model_path) if rl_model_path else None,
#             "rank1_model_path": rank1_model_path,
#             "rl_score": rl_score,
#             "rank1_score": rank1_score,
#             "base_score": args.base_score,
#         }

#         frac_recovered = None
#         if args.base_score is not None and rl_score is not None and rank1_score is not None:
#             denom = rl_score - args.base_score
#             if abs(denom) > 1e-8:
#                 frac_recovered = (rank1_score - args.base_score) / denom
#         rec["frac_recovered"] = frac_recovered

#         records.append(rec)

#     df = pd.DataFrame.from_records(records)

#     # 默认 summary 路径
#     if args.summary_parquet is None:
#         summary_parquet = output_root / "alpharl_rank1_eval.parquet"
#     else:
#         summary_parquet = Path(args.summary_parquet)

#     summary_parquet.parent.mkdir(parents=True, exist_ok=True)
#     df.to_parquet(summary_parquet, index=False)
#     print(f"\n[OK] Wrote Alpha-RL rank1 eval summary ({len(df)} rows) to {summary_parquet}")


# if __name__ == "__main__":
#     main()