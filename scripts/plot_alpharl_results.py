import os
import json
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 你关心的 task 名（按需再加）
TASK_NAMES = {"aime24", "aime25", "gpqa", "gsm8k", "math", "minerva"}


def load_examples(path: Path):
    """尽量鲁棒地把一个 json/jsonl 文件读成 list[dict]."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 尝试整个文件是一个 JSON（list 或 dict）
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    # 否则当成 jsonl
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Bad JSON line in {path}: {e}")
    return examples


def compute_acc_from_examples(examples):
    """从样本列表中，利用 is_correct / answers_correctness 计算 accuracy."""
    flags = []
    for ex in examples:
        if isinstance(ex, dict):
            if "is_correct" in ex:
                flags.append(bool(ex["is_correct"]))
            elif "answers_correctness" in ex and ex["answers_correctness"]:
                flags.append(bool(ex["answers_correctness"][0]))
    if not flags:
        return None
    return float(sum(flags) / len(flags))


def collect_metrics(root: Path):
    """
    递归遍历 metrics_rl / metrics_rank1 下所有 json/jsonl，
    对每个 (kind, step, task, file) 计算 acc.
    """
    rows = []

    for kind in ["metrics_rl", "metrics_rank1"]:
        base = root / kind
        if not base.exists():
            print(f"[WARN] {kind} root not found: {base}")
            continue

        print(f"[INFO] Scanning {kind} under {base} ...")

        # 找所有 json/jsonl
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if not (str(path).endswith(".json") or str(path).endswith(".jsonl")):
                continue

            # 从路径里抽 step
            m_step = re.search(r"global_step_(\d+)", str(path))
            if not m_step:
                continue
            step = int(m_step.group(1))

            # 从路径名中抽 task：最后一个属于 TASK_NAMES 的目录名
            task = None
            for part in reversed(path.parts):
                if part in TASK_NAMES:
                    task = part
                    break
            if task is None:
                # 非我们关心的数据集，跳过
                continue

            # 读文件并计算 acc
            examples = load_examples(path)
            if not examples:
                print(f"[WARN] No examples in {path}")
                continue

            acc = compute_acc_from_examples(examples)
            if acc is None:
                print(f"[WARN] No correctness info in {path}")
                continue

            rows.append(
                {
                    "kind": "rl" if "metrics_rl" in path.parts else "rank1",
                    "step": step,
                    "task": task,
                    "acc": acc,
                    "json_path": str(path),
                    "n_examples": len(examples),
                }
            )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="alpharl_step1 的 --output_root 目录，例如 /data/.../alpharl_out/grpo_math_qwen3_8b",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"[INFO] Using ROOT = {root}")

    out_fig_dir = root / "plots"
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_metrics(root)
    if not rows:
        print("[ERROR] No metrics collected.")
        return

    df = pd.DataFrame(rows)
    print("\n==== Collected metrics (head) ====")
    print(df.head())
    print("\nTasks:", sorted(df["task"].unique()))
    print("Steps:", sorted(df["step"].unique()))

    # 保存 summary CSV
    out_csv = root / "alpharl_metrics_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved summary to {out_csv}")

    # 画图：每个 task 一张 RL vs Rank-1 曲线
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task].copy()
        if sub.empty:
            continue

        wide = sub.pivot_table(
            index="step",
            columns="kind",
            values="acc",
            aggfunc="mean",
        ).sort_index()

        plt.figure(figsize=(6, 4))
        if "rl" in wide.columns:
            plt.plot(wide.index, wide["rl"], marker="o", label="RL checkpoint")
        if "rank1" in wide.columns:
            plt.plot(wide.index, wide["rank1"], marker="s", linestyle="--", label="Rank-1")

        plt.xlabel("global_step")
        plt.ylabel("Accuracy")
        plt.title(f"Qwen3-8B GRPO on {task}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig_path = out_fig_dir / f"grpo_qwen3_8b_{task}_acc.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[OK] Saved plot for task={task} -> {fig_path}")


if __name__ == "__main__":
    main()