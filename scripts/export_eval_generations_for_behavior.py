#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_eval_generations_for_behavior.py

用途：
  读取 data/eval/model 目录下各个模型、各个 task 的 eval jsonl（已有 generated_responses），
  统一整理成 behavior_label_and_train.py 所需的 parquet：

    [id, model_name, task, dataset, prompt, output]

路径解析规则：
  eval_root / <vendor> / <model_name> / <dataset_dir> / *.jsonl

其中 <dataset_dir> 可能是:
  - aime24 / aime25 / gpqa / gsm8k / math / minerva

我们会用一个映射把它们规范成和 benchmarks_fine.csv 一致的名字:
  - aime24  -> AIME2024
  - aime25  -> AIME2025
  - gpqa    -> GPQA
  - gsm8k   -> gsm8k
  - math    -> MATH500
  - minerva -> MinervaMath

最后导出的:
  - dataset 列 = 规范后的名字 (例如 "AIME2024", "MATH500")
  - task    列 = f"{dataset}_{task_suffix}"  (例如 "AIME2024_math", "MATH500_math")

这样就可以和 benchmarks_fine.csv 里的 task 完全对齐。
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_root",
        type=str,
        default="data/eval/model",
        help="eval 结果根目录（包含 meta-llama/Qwen/... 子目录）",
    )
    ap.add_argument(
        "--output_path",
        type=str,
        default="data/completions/eval_generations_all.parquet",
        help="导出的 parquet 路径",
    )
    ap.add_argument(
        "--task_suffix",
        type=str,
        default="math",
        help='task 后缀，比如 "math"，生成的 task 形如 "<dataset>_math"',
    )
    return ap.parse_args()


# ---------- 规范化 dataset 名的映射表 ----------

DATASET_CANONICAL = {
    "aime24": "AIME2024",
    "aime25": "AIME2025",
    "gpqa": "GPQA",
    "gsm8k": "gsm8k",          # 与 benchmarks_fine.csv 中一致
    "math": "MATH500",         # 原来 behavior 里是 math_math，要改成 MATH500_math
    "minerva": "MinervaMath",  # 对应 MinervaMath_math
}


# ----------- 一些小工具 ----------- #

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def pick_first(fields: List[str], obj: Dict[str, Any]) -> Optional[Any]:
    for k in fields:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def extract_output(line: Dict[str, Any]) -> Optional[str]:
    # 1) generated_responses: list[str] or list[{"text": "..."}]
    gr = line.get("generated_responses", None)
    if isinstance(gr, list) and len(gr) > 0:
        if isinstance(gr[0], str):
            return gr[0]
        elif isinstance(gr[0], dict) and "text" in gr[0]:
            return str(gr[0]["text"])

    # 2) 单字段备选
    cand = pick_first(
        ["output", "answer", "model_output", "generated_text", "completion"],
        line,
    )
    if cand is None:
        return None
    if not isinstance(cand, str):
        cand = str(cand)
    return cand


def extract_prompt(line: Dict[str, Any]) -> Optional[str]:
    cand = pick_first(
        ["prompt", "question", "input", "query", "instruction"],
        line,
    )
    if cand is None:
        return None
    if not isinstance(cand, str):
        cand = str(cand)
    return cand


def extract_id(line: Dict[str, Any], dataset: str, idx: int) -> str:
    cand = pick_first(
        ["id", "problem_id", "uid", "index", "idx"],
        line,
    )
    if cand is None:
        return f"{dataset}-{idx}"
    return str(cand)


# ----------- 主逻辑 ----------- #

def main():
    args = parse_args()
    eval_root = Path(args.eval_root)
    out_path = Path(args.output_path)
    task_suffix = args.task_suffix

    if not eval_root.exists():
        raise FileNotFoundError(f"Eval root not found: {eval_root}")

    rows: List[Dict[str, Any]] = []

    # 遍历 vendor / model / dataset_dir / *.jsonl
    pattern = "**/*.jsonl"
    all_files = sorted(eval_root.glob(pattern))

    if not all_files:
        print(f"[WARN] No jsonl files found under {eval_root}")
        return

    print(f"[INFO] Found {len(all_files)} jsonl files under {eval_root}")

    for jf in all_files:
        # rel: vendor/model_name/dataset_dir/filename
        rel = jf.relative_to(eval_root)
        parts = rel.parts
        if len(parts) < 3:
            print(f"[WARN] Skip {jf}, path depth < 3")
            continue

        vendor = parts[0]
        model_name = parts[1]
        raw_dataset = parts[2]  # aime24 / math / minerva / etc.

        # 规范化 dataset 名，以便和 benchmarks_fine.csv 对齐
        dataset = DATASET_CANONICAL.get(raw_dataset, raw_dataset)

        print(f"[INFO] Reading {jf} (vendor={vendor}, model={model_name}, dataset={dataset}, raw_dataset={raw_dataset})")

        for i, line in enumerate(read_jsonl(jf)):
            sample_id = extract_id(line, dataset, i)
            prompt = extract_prompt(line) or ""
            output = extract_output(line) or ""

            # task 使用规范化后的 dataset
            task_name = f"{dataset}_{task_suffix}"

            rows.append(
                {
                    "id": sample_id,
                    "model_name": model_name,
                    "task": task_name,
                    "dataset": dataset,   # 保存规范名，方便后续分析
                    "prompt": prompt,
                    "output": output,
                }
            )

    if not rows:
        print("[WARN] No rows parsed, nothing to write.")
        return

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"[OK] Wrote {len(df)} rows to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()