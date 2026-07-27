#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert OlympiadBench raw data to jsonl with fields:
  problem, solution, answer, subject, subfield, difficulty, id, index, ...

输入示例（json array）：
[
  {
    "id": 1606,
    "question": "...",
    "solution": [
      "Sergey can determine Xenia's number in 2 but not fewer moves.\n...",
      "...",
    ],
    "final_answer": ["2"],
    "context": null,
    "modality": "Text-only",
    "difficulty": "Competition",
    "is_multiple_answer": false,
    "unit": null,
    "answer_type": "Numerical",
    "question_type": "Open-ended",
    "subfield": "Combinatorics",
    "subject": "Math",
    "language": "English",
    "index": 0
  },
  ...
]

输出示例（jsonl，每行一条）：
{"problem": "...", "solution": "Sergey can determine ...", "answer": "2", "subject": "Math", "subfield": "Combinatorics", "difficulty": "Competition", "id": 1606, "index": 0}
"""

import argparse
import json
import io
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_path", type=str, required=True,
        help="OlympiadBench 原始 json / jsonl 路径，例如 OlympiadBench_OE_TO_maths_en_COMP.json"
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


def normalize_solution(sol):
    """solution 可能是 list[str] 或 str，这里统一成一个字符串。"""
    if sol is None:
        return ""
    if isinstance(sol, list):
        # 多段解答拼在一起，用空行分隔
        return "\n\n".join(str(s) for s in sol)
    return str(sol)


def extract_answer(ex):
    """
    优先使用 final_answer（list），否则用 answer 字段。
    都没有就返回 ""。
    """
    fa = ex.get("final_answer")
    if isinstance(fa, list) and len(fa) > 0:
        return str(fa[0])
    if "answer" in ex and ex["answer"] is not None:
        return str(ex["answer"])
    return ""


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
            problem = ex.get("question") or ex.get("problem", "")
            solution = normalize_solution(ex.get("solution"))
            answer = extract_answer(ex)

            new_ex = {
                "problem": problem,
                "solution": solution,
                "answer": answer,
                # 常用元信息
                "subject": ex.get("subject"),
                "subfield": ex.get("subfield"),
                "difficulty": ex.get("difficulty"),
                "id": ex.get("id"),
                "index": ex.get("index"),
                "language": ex.get("language"),
                "answer_type": ex.get("answer_type"),
                "question_type": ex.get("question_type"),
            }

            # 有 context/image 之类的也可以保留
            for extra_key in ["context", "image_1", "image_2", "image_3", "image_4", "image_5", "unit"]:
                if extra_key in ex:
                    new_ex[extra_key] = ex[extra_key]

            f_out.write(json.dumps(new_ex, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote converted data to {out_path}")


if __name__ == "__main__":
    main()