#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_llm_label_glm.py

用 GLM 对模型生成的解答进行行为打标（decomp / verify / backtrack / reflect），
输出每条样本、每种行为的出现次数和字符 span。

用法示例：

python scripts/behavior_llm_label_glm.py \
  --in_parquet data/completions/eval_generations_all.parquet \
  --out_parquet data/behavior/behavior_llm_labels.parquet \
  --model_filter "Qwen3-8B-GRPO-math" \
  --task_prefix "AIME2024_math,AIME2025_math,MATH500_math,MinervaMath_math,gsm8k_math" \
  --max_samples_per_task 200 \
  --glm_model gemini-3-pro-preview

环境变量：
  GLM_API_KEY   必须设置为你的 GLM API key
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

import pandas as pd
import requests
from tqdm import tqdm

# 复用你已经测试过的 glm_client
from glm_client import glm_chat as glm_chat_base


# ----------------------------
# 对 glm_client 再包一层：加重试 & 更长超时
# ----------------------------
def glm_chat_with_retry(
    messages: List[Dict[str, str]],
    model: str = "gemini-3-pro-preview",
    temperature: float = 0.0,
    timeout: int = 300,
    max_retries: int = 3,
    retry_sleep: float = 3.0,
) -> str:
    """
    调用 GLM（通过 glm_client.glm_chat），带重试。

    返回：
        成功时：LLM 返回的文本 content
        全部重试失败时：抛出最后一个异常（上层会捕获并打 0）
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return glm_chat_base(
                messages,
                model=model,
                temperature=temperature,
                timeout=timeout,
            )
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            print(
                f"[WARN] ReadTimeout (attempt {attempt}/{max_retries}) for model={model}: {e}",
                file=sys.stderr,
            )
        except requests.exceptions.RequestException as e:
            last_err = e
            print(
                f"[WARN] RequestException (attempt {attempt}/{max_retries}) for model={model}: {e}",
                file=sys.stderr,
            )
        except Exception as e:
            last_err = e
            print(
                f"[WARN] GLM general exception (attempt {attempt}/{max_retries}) for model={model}: {e}",
                file=sys.stderr,
            )

        if attempt < max_retries:
            time.sleep(retry_sleep * attempt)

    # 全部失败，抛出最后一个异常，由上层统一处理
    raise last_err if last_err is not None else RuntimeError("Unknown GLM error")


# ----------------------------
# 行为打标 prompt 模板
# ----------------------------

BEHAVIOR_DEFS = {
    "decomp": {
        "name": "DECOMPOSITION",
        "zh": "分步：将问题明确拆解成多个子步骤、子任务或子结论。",
    },
    "verify": {
        "name": "VERIFICATION",
        "zh": "校验：在得到候选答案后，显式地代入原式、检验条件、验证结果是否正确。",
    },
    "backtrack": {
        "name": "BACKTRACKING",
        "zh": "回退：发现之前的推理或计算有问题，并回到较早步骤重新开始或大幅修改思路。",
    },
    "reflect": {
        "name": "REFLECTION",
        "zh": "反思：在解答结尾对前文步骤进行总结、复盘、或讨论答案合理性。",
    },
}


def build_behavior_prompt(solution: str, behavior_type: str) -> List[Dict[str, str]]:
    """
    构造给 GLM 的 messages，要求输出 JSON：
    {
      "behavior": "<decomp|verify|backtrack|reflect>",
      "count": <int>,
      "spans": [{"start_char": int, "end_char": int}, ...]
    }
    """
    if behavior_type not in BEHAVIOR_DEFS:
        raise ValueError(f"Unknown behavior_type={behavior_type}")

    meta = BEHAVIOR_DEFS[behavior_type]

    system_msg = (
        "You are a precise annotator for reasoning behaviors in math solutions. "
        "You must ONLY output a JSON object, no extra text."
    )

    user_msg = f"""
We analyze ONE type of reasoning behavior in the following solution.

[BEHAVIOR TYPE]
- English name: {meta["name"]}
- Chinese description: {meta["zh"]}

[INSTRUCTIONS]
1. Carefully read the solution.
2. Find all spans (continuous text segments) that clearly exhibit THIS behavior.
3. For each span, record the character-level [start, end) indices in the ORIGINAL solution string.
   - start_char: 0-based index of the first character of the span
   - end_char: index just after the last character of the span
4. Count how many such spans exist.

[OUTPUT FORMAT]
Return STRICTLY a JSON object like:
{{
  "behavior": "{behavior_type}",
  "count": <integer number of spans>,
  "spans": [
    {{"start_char": <int>, "end_char": <int>}}
  ]
}}

Do NOT include any explanation or commentary outside this JSON.

[SOLUTION]
{solution}
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def safe_parse_behavior_json(raw: str, behavior_type: str) -> Dict[str, Any]:
    """
    解析 GLM 返回的 JSON；若失败则回落到默认 0 / []。
    """
    raw = raw.strip()
    # 有些模型可能会在 JSON 外面加前后说明，尝试从第一个 { 开始截取
    if not raw.startswith("{"):
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]
    if not raw.endswith("}"):
        # 粗略从最后一个 } 截断
        idx = raw.rfind("}")
        if idx != -1:
            raw = raw[: idx + 1]

    try:
        obj = json.loads(raw)
    except Exception:
        # 解析失败
        return {"behavior": behavior_type, "count": 0, "spans": []}

    # 做一点健壮性检查
    beh = obj.get("behavior", behavior_type)
    count = obj.get("count", 0)
    spans = obj.get("spans", [])

    # 规范化 spans
    norm_spans = []
    if isinstance(spans, list):
        for sp in spans:
            try:
                s = int(sp.get("start_char", 0))
                e = int(sp.get("end_char", 0))
                if 0 <= s < e:
                    norm_spans.append({"start_char": s, "end_char": e})
            except Exception:
                continue

    if not isinstance(count, int):
        try:
            count = int(count)
        except Exception:
            count = len(norm_spans)

    return {"behavior": beh, "count": count, "spans": norm_spans}


# ----------------------------
# 主流程
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_parquet",
        type=str,
        required=True,
        help="输入 completions 表路径（eval_generations_all.parquet）",
    )
    ap.add_argument(
        "--out_parquet",
        type=str,
        required=True,
        help="输出行为标注表路径（parquet）",
    )
    ap.add_argument(
        "--model_filter",
        type=str,
        default="",
        help="只保留这些 model_name，逗号分隔；为空则不过滤。",
    )
    ap.add_argument(
        "--task_prefix",
        type=str,
        default="",
        help="只保留 task 以这些前缀开头的样本，逗号分隔；为空则不过滤。",
    )
    ap.add_argument(
        "--max_samples_per_task",
        type=int,
        default=200,
        help="每个 task 最大采样条数；<=0 表示全部保留。",
    )
    ap.add_argument(
        "--glm_model",
        type=str,
        default="gemini-3-pro-preview",  # 和你 glm_client 自测保持一致
        help="GLM 模型名称。",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机采样 seed。",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"[INFO] Loading completions from {args.in_parquet}")
    df = pd.read_parquet(args.in_parquet)

    required_cols = {"id", "model_name", "task", "prompt", "output"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet missing columns: {missing}")

    # model 过滤
    if args.model_filter:
        models = [m.strip() for m in args.model_filter.split(",") if m.strip()]
        df = df[df["model_name"].isin(models)]
        print(f"[INFO] Filtered by model_name, remain {len(df)} rows.")

    # task 过滤（前缀）
    if args.task_prefix:
        prefixes = [p.strip() for p in args.task_prefix.split(",") if p.strip()]

        def keep_task(t: str) -> bool:
            return any(t.startswith(p) for p in prefixes)

        df = df[df["task"].astype(str).apply(keep_task)]
        print(f"[INFO] Filtered by task_prefix, remain {len(df)} rows.")

    if df.empty:
        print("[ERROR] No data left after filtering.")
        sys.exit(1)

    # 每个 task 限制样本数
    if args.max_samples_per_task > 0:
        parts = []
        for (task, _model), g in df.groupby(["task", "model_name"]):
            if len(g) > args.max_samples_per_task:
                g = g.sample(
                    n=args.max_samples_per_task,
                    random_state=args.seed,
                    replace=False,
                )
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)
        print(
            f"[INFO] After per-task sampling (max {args.max_samples_per_task}), total {len(df)} rows."
        )

    records: List[Dict[str, Any]] = []

    # 确保输出目录存在
    out_dir = os.path.dirname(args.out_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    behavior_types = ["decomp", "verify", "backtrack", "reflect"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM behavior labeling", ncols=100):
        sample_id = row["id"]
        model_name = row["model_name"]
        task = row["task"]
        output = row["output"]
        out_len = len(output) if isinstance(output, str) else 0

        # 本 sample 四类行为的统计
        sample_summary = {b: 0 for b in behavior_types}

        if not isinstance(output, str) or not output.strip():
            # 空输出，直接填 0
            for b in behavior_types:
                records.append(
                    {
                        "id": sample_id,
                        "model_name": model_name,
                        "task": task,
                        "behavior_type": b,
                        "count": 0,
                        "spans_json": "[]",
                        "output_len": out_len,
                        "glm_raw": "",
                    }
                )

            # 打印空输出 sample 的 summary
            print(
                f"[SAMPLE DONE] id={sample_id} | model={model_name} | task={task} | "
                f"decomp=0, verify=0, backtrack=0, reflect=0"
            )
            print("---- Model output (empty) ----")
            print("(no output)")
            print("====================================")
            sys.stdout.flush()

            # 每处理完一个 sample 就写一次 parquet（覆盖）
            out_df = pd.DataFrame.from_records(records)
            out_df.to_parquet(args.out_parquet, index=False)

            continue

        # 非空输出：对每种行为都跑一次 LLM
        for b in behavior_types:
            glm_raw = ""
            try:
                messages = build_behavior_prompt(output, b)
                glm_raw = glm_chat_with_retry(
                    messages,
                    model=args.glm_model,
                    temperature=0.0,
                    timeout=300,
                    max_retries=3,
                    retry_sleep=3.0,
                )
                # 打印 GLM 完整回复
                print(
                    f"[GLM RAW] id={sample_id} | model={model_name} | task={task} | behavior={b}:\n"
                )
                print(glm_raw)
                print("---------- END OF GLM RAW ----------")
                sys.stdout.flush()

                parsed = safe_parse_behavior_json(glm_raw, b)
            except Exception as e:
                print(
                    f"[WARN] GLM call failed for id={sample_id}, behavior={b}: {e}",
                    file=sys.stderr,
                )
                parsed = {"behavior": b, "count": 0, "spans": []}

            count_b = int(parsed.get("count", 0))
            sample_summary[b] = count_b

            spans_json = json.dumps(parsed.get("spans", []), ensure_ascii=False)

            records.append(
                {
                    "id": sample_id,
                    "model_name": model_name,
                    "task": task,
                    "behavior_type": b,
                    "count": count_b,
                    "spans_json": spans_json,
                    "output_len": out_len,
                    "glm_raw": glm_raw,
                }
            )

        # 打印本 sample 的 summary + 模型输出（这里模型输出仍然做 400 字截断）
        print(
            f"[SAMPLE DONE] id={sample_id} | model={model_name} | task={task} | "
            f"decomp={sample_summary['decomp']}, "
            f"verify={sample_summary['verify']}, "
            f"backtrack={sample_summary['backtrack']}, "
            f"reflect={sample_summary['reflect']}"
        )
        print("---- Model output (first 400 chars) ----")
        print(output[:400])
        if len(output) > 400:
            print("...(truncated)")
        print("====================================")
        sys.stdout.flush()

        # 每处理完一个 sample 就写一次 parquet（覆盖）
        out_df = pd.DataFrame.from_records(records)
        out_df.to_parquet(args.out_parquet, index=False)

    # 最后再写一次，确保完整（其实上面已经覆盖写过很多次了）
    if records:
        out_df = pd.DataFrame.from_records(records)
        out_df.to_parquet(args.out_parquet, index=False)

    print(
        f"[OK] Saved LLM behavior labels ({len(records)} rows) to {args.out_parquet}"
    )


if __name__ == "__main__":
    main()