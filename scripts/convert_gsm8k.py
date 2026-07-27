#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert GSM8K format:
  {"question": "...", "answer": ".... #### 18"}

to:
  {"problem": "...", "solution": ".... #### 18", "answer": "18"}

支持两种输入：
- JSONL：每行一个样本
- JSON：整体是一个 list[dict]

用法示例：
  python convert_gsm8k.py \
      --input_path gsm8k/test.jsonl \
      --output_path gsm8k/test_converted.jsonl
"""

import argparse
import json
import io
import re
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_path", type=str, required=True,
        help="原始 GSM8K 文件路径（jsonl 或 json）"
    )
    ap.add_argument(
        "--output_path", type=str, required=True,
        help="转换后输出的 jsonl 路径"
    )
    return ap.parse_args()


def load_any(path: Path):
    """既能读 jsonl 也能读 json list。"""
    text = path.read_text(encoding="utf-8").strip()
    # 先尝试整体当 json
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 否则按 jsonl 逐行读
    samples = []
    for line in io.StringIO(text):
        line = line.strip()
        if not line:
            continue
        samples.append(json.loads(line))
    return samples


def extract_final_answer(solution: str) -> str:
    """
    从 GSM8K 的 solution 里抽取最后一个 '#### xxx' 的 xxx 部分。
    如果找不到 '####'，就返回整段去首尾空白的字符串。
    """
    if not isinstance(solution, str):
        return str(solution)

    matches = list(re.finditer(r"####\s*([^\n]+)", solution))
    if not matches:
        return solution.strip()
    return matches[-1].group(1).strip()


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
            q = ex.get("question", "")
            sol = ex.get("answer", "")
            final_ans = extract_final_answer(sol)

            new_ex = {
                "problem": q,
                "solution": sol,
                "answer": final_ans,
            }
            # 保留原来的 idx 等信息（如果有）
            for k in ("idx", "id"):
                if k in ex:
                    new_ex[k] = ex[k]

            f_out.write(json.dumps(new_ex, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote converted data to {out_path}")


if __name__ == "__main__":
    main()