#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 data/eval 目录下自动收集所有 acc_*.csv，
解析出 (model_name, dataset, accuracy)，
并转换成细粒度 benchmark 表：

    model_name, task, score

其中 task = "<标准化后的dataset>_math"（目前只做 math，可以改成别的后缀）。

输出文件：
    data/eval/benchmarks_fine.csv

如果目标文件已存在，会先读入旧数据，和新数据拼接，
按 (model_name, task) 去重后再写回（保留最新的一条）。
"""

import pandas as pd
from pathlib import Path

# 目前只做 math，如果以后有 coding / logic，可以再扩展
TASK_SUFFIX = "math"

# 和 build_probe.py / run_inference / train_csv_weights.py 保持一致的数据集别名映射
DATASET_ALIAS = {
    # 你在 acc_*.csv 里用的名字 -> probe.jsonl / feature 里用的名字
    "gsm8k": "gsm8k",
    "math": "MATH500",        # MATH500_test.json
    "minerva": "MinervaMath", # Minerva math
    "aime24": "AIME2024",     # AIME_2024
    # 如果以后 probe 里加了这些，就在这里补上：
    "aime25": "AIME2025",
    "gpqa": "GPQA",
    "odyssey": "Odyssey",
    "olympiadbench": "OlympiadBench",
}


def acc_csv_to_bench_df(acc_path: Path, task_suffix: str) -> pd.DataFrame:
    """
    把一个 acc_*.csv 转成统一的 (model_name, task, score) DataFrame.
    """
    # acc_Qwen3-8B.csv -> Qwen3-8B
    stem = acc_path.stem  # "acc_Qwen3-8B"
    if not stem.startswith("acc_"):
        raise ValueError(f"Unexpected acc csv name: {acc_path}")
    model_name = stem[len("acc_"):]  # 去掉前缀 "acc_"

    df = pd.read_csv(acc_path)

    # 去掉 OVERALL 或其它汇总行（如果没有这列，不会报错）
    if "dataset" not in df.columns:
        raise ValueError(f"{acc_path} 中缺少 'dataset' 列")

    df = df[df["dataset"] != "OVERALL"]

    # 兼容列名：优先用 accuracy
    if "accuracy" in df.columns:
        score_col = "accuracy"
    elif "acc" in df.columns:
        score_col = "acc"
    else:
        raise ValueError(f"{acc_path} 中找不到 accuracy/acc 列")

    rows = []
    for _, row in df.iterrows():
        dataset_raw = str(row["dataset"])
        score = float(row[score_col])

        # 统一映射到 probe / feature 侧用的 dataset 名
        dataset_norm = DATASET_ALIAS.get(dataset_raw, dataset_raw)
        task_name = f"{dataset_norm}_{task_suffix}"

        rows.append(
            {
                "model_name": model_name,
                "task": task_name,
                "score": score,
            }
        )

    out_df = pd.DataFrame(rows)
    return out_df


def main():
    # 假设脚本放在 scripts/ 目录下，项目根目录是上一层
    repo_root = Path(__file__).resolve().parent.parent
    eval_dir = repo_root / "data" / "eval"
    out_csv = eval_dir / "benchmarks_fine.csv"

    # 找到所有 acc_*.csv
    acc_files = sorted(eval_dir.glob("acc_*.csv"))
    if not acc_files:
        print(f"[WARN] 在 {eval_dir} 下没有找到 acc_*.csv，什么都不做。")
        return

    all_new = []
    print("[INFO] 将以下 acc_* 文件转换为细粒度 benchmark：")
    for acc_path in acc_files:
        print("  -", acc_path)
        df_one = acc_csv_to_bench_df(acc_path, TASK_SUFFIX)
        all_new.append(df_one)

    new_df = pd.concat(all_new, ignore_index=True)

    # 如果已有旧的 benchmarks_fine.csv，则一起合并并去重
    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        # (model_name, task) 去重，保留最后出现的那一行
        merged = merged.drop_duplicates(subset=["model_name", "task"], keep="last")
    else:
        merged = new_df

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    print(f"[OK] 已保存合并后的 benchmark 到: {out_csv}")
    print(merged)


if __name__ == "__main__":
    main()