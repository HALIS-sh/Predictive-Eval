#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect accuracies from Alpha-RL style eval outputs.

Assumed directory layout, e.g.:

data/eval/model/Qwen/Qwen3-8B/
  ├── gsm8k/
  │     └── test_qwen-base_t0.0_1st0_r1_k1_s0_e500.jsonl
  ├── math/
  │     └── ...
  ├── aime24/
  │     └── ...
  └── ...

Each json/jsonl sample is expected to contain at least:
  - is_correct : bool (preferred)
    OR
  - answers_correctness : list[bool] / list[int] / bool / int

Output: a CSV with per-dataset accuracy, e.g.:

dataset,n,correct,accuracy
gsm8k,500,430,0.86
math,500,210,0.42
overall,1500,1000,0.6667
"""

import argparse
import csv
import json
import io
from pathlib import Path
from typing import List, Dict, Any


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root eval dir, e.g. data/eval/model/Qwen/Qwen3-8B",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Where to save summary CSV",
    )
    return ap.parse_args()


def load_any_json(path: Path) -> List[Dict[str, Any]]:
    """Robust loader: support JSON(list) or JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    # try: whole file is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            # 单个 dict 的情况也兼容一下
            return [obj]
    except Exception:
        pass

    # fallback: treat as jsonl
    items: List[Dict[str, Any]] = []
    for line in io.StringIO(text):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            print(f"[WARN] bad json line in {path}: {line[:80]}...")
    return items


def get_is_correct(ex: Dict[str, Any]) -> bool | None:
    """Try to extract a boolean correctness from a sample."""
    if "is_correct" in ex:
        val = ex["is_correct"]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)

    if "answers_correctness" in ex:
        val = ex["answers_correctness"]
        # 可能是 list / bool / int
        if isinstance(val, list) and len(val) > 0:
            # 只要第一条即可；或者要求全部为真，这里用“全部为真”
            return all(bool(x) for x in val)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)

    return None


def main():
    args = parse_args()
    root = Path(args.root_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_for_csv = []
    total_n = 0
    total_correct = 0

    print(f"[INFO] Scanning eval root: {root}")

    # 遍历每个数据集子目录
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset_name = ds_dir.name

        json_files = list(ds_dir.glob("*.json")) + list(ds_dir.glob("*.jsonl"))
        if not json_files:
            print(f"[WARN] No json/jsonl files found in {ds_dir}, skip")
            continue

        # 把目录下所有文件的样本合并
        all_examples: List[Dict[str, Any]] = []
        for f in sorted(json_files):
            exs = load_any_json(f)
            print(f"[INFO] Loaded {len(exs)} examples from {f}")
            all_examples.extend(exs)

        if not all_examples:
            print(f"[WARN] No examples in {ds_dir}, skip")
            continue

        n = 0
        c = 0
        missing = 0
        for ex in all_examples:
            flag = get_is_correct(ex)
            if flag is None:
                missing += 1
                continue
            n += 1
            if flag:
                c += 1

        if n == 0:
            print(f"[WARN] Dataset {dataset_name} has 0 examples with correctness info, skip")
            continue

        acc = c / n
        total_n += n
        total_correct += c

        print(
            f"[INFO] Dataset {dataset_name}: "
            f"n={n}, correct={c}, acc={acc:.4f}, missing_labels={missing}"
        )

        rows_for_csv.append(
            {
                "dataset": dataset_name,
                "n": n,
                "correct": c,
                "accuracy": round(acc, 6),
                "missing_labels": missing,
            }
        )

    # overall
    if total_n > 0:
        overall_acc = total_correct / total_n
        rows_for_csv.append(
            {
                "dataset": "OVERALL",
                "n": total_n,
                "correct": total_correct,
                "accuracy": round(overall_acc, 6),
                "missing_labels": 0,
            }
        )
        print(
            f"[INFO] OVERALL: n={total_n}, correct={total_correct}, "
            f"acc={overall_acc:.4f}"
        )

    # 写 CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "n", "correct", "accuracy", "missing_labels"],
        )
        writer.writeheader()
        for r in rows_for_csv:
            writer.writerow(r)

    print(f"[OK] Saved accuracy summary to {out_csv}")


if __name__ == "__main__":
    main()