#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert MATH500 raw data to jsonl with fields:
  problem, solution, answer, subject, level, unique_id

输入示例（list[dict] / json）：
[
  {
    "problem": "...",
    "solution": "...",
    "answer": "\\sqrt{51}",
    "subject": "Precalculus",
    "level": 1,
    "unique_id": "test/precalculus/1303.json"
  },
  ...
]

输出示例（jsonl，每行一条）：
{"problem": "...", "solution": "...", "answer": "58", "subject": "Number Theory", "level": 4, "unique_id": "test/number_theory/488.json"}
"""

import argparse
import json
import io
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_path", type=str, required=True,
        help="MATH500 原始 json / jsonl 路径，例如 MATH500_test.json"
    )
    ap.add_argument(
        "--output_path", type=str, required=True,
        help="输出的 jsonl 路径"
    )
    return ap.parse_args()


def load_any(path: Path):
    """既支持 json(list) 也支持 jsonl。"""
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
        for i, ex in enumerate(data):
            problem = ex.get("problem", "")
            solution = ex.get("solution", "")
            answer = ex.get("answer", "")
            subject = ex.get("subject")
            level = ex.get("level")
            unique_id = ex.get("unique_id")

            # answer 统一成字符串
            if not isinstance(answer, str):
                answer = str(answer)

            new_ex = {
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "subject": subject,
                "level": level,
                "unique_id": unique_id,
            }

            # 如果原来有 idx / id，也可以顺便保留
            for k in ("idx", "id"):
                if k in ex:
                    new_ex[k] = ex[k]

            f_out.write(json.dumps(new_ex, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote converted data to {out_path}")


if __name__ == "__main__":
    main()