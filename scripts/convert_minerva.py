#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 Minerva 原始数据转成 jsonl，每行包含：
  problem, solution, answer, type, idx, index 等字段。

输入示例（json 或 jsonl）：
[
  {
    "problem": "...",
    "solution": "...",
    "type": "Introduction to Astronomy (8.282J Spring 2006)",
    "idx": 0,
    "answer": "1.6",
    "index": 0
  },
  ...
]

输出示例（jsonl，每行一条）：
{"problem": "...", "solution": "...", "answer": "1.6", "type": "...", "idx": 0, "index": 0}
"""

import argparse
import json
import io
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_path", type=str, required=True,
        help="Minerva 原始 json/jsonl 路径（例如 minerva_math_test.json）"
    )
    ap.add_argument(
        "--output_path", type=str, required=True,
        help="输出 jsonl 路径"
    )
    return ap.parse_args()


def load_any(path: Path):
    """支持 json(list) 和 jsonl 两种格式。"""
    text = path.read_text(encoding="utf-8").strip()
    # 先尝试整体 json
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 否则按 jsonl 读
    samples = []
    for line in io.StringIO(text):
        line = line.strip()
        if not line:
            continue
        samples.append(json.loads(line))
    return samples


def main():
    args = parse_args()
    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {in_path}")
    data = load_any(in_path)
    print(f"[INFO] Loaded {len(data)} examples")

    with out_path.open("w", encoding="utf-8") as f_out:
        for ex in data:
            problem = ex.get("problem", "")
            solution = ex.get("solution", "") or ""
            answer = ex.get("answer", "")
            if answer is None:
                answer = ""
            else:
                answer = str(answer)

            new_ex = {
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "type": ex.get("type"),
                "idx": ex.get("idx"),
                "index": ex.get("index"),
            }

            f_out.write(json.dumps(new_ex, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote converted data to {out_path}")


if __name__ == "__main__":
    main()