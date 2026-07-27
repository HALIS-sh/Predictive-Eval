#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
behavior_marker_usage_diff.py

Phase 3: 对比 base vs RL 在“行为 marker token”上的使用频率。

思路：
  1）读入 Phase 2.2 输出的 marker JSON（每个行为一批 token_str）。
  2）读入 eval_generations_all.parquet，筛选出 base / RL 两种 model 的 completions。
  3）用同一个 tokenizer 对 base / RL 的 output 分词，统计：
        - total_tokens_base / total_tokens_rl
        - 对每个 (behavior, token_str)：
            c_base: 在 base 输出中出现次数
            c_rl  : 在 RL 输出中出现次数
  4）计算各自频率以及 log 频率比：
        freq_base = c_base / total_tokens_base
        freq_rl   = c_rl   / total_tokens_rl
        log_ratio = log((freq_rl + eps) / (freq_base + eps))
  5）输出一张表，用于分析“RL 是否更常用这些行为 marker token”。

用法示例（Qwen2.5-7B base & Qwen2.5-7B-Math RL）：

python scripts/behavior_marker_usage_diff.py \
  --completions_parquet data/completions/eval_generations_all.parquet \
  --marker_json        data/behavior/behavior_marker_tokens.json \
  --base_model_name    "Qwen2.5-7B" \
  --rl_model_name      "Qwen2.5-7B-Math" \
  --tokenizer_path     "/data/wenhesun/model/Qwen/Qwen2.5-7B" \
  --task_prefix        "AIME2024_math,MinervaMath_math" \
  --top_k              50 \
  --out_parquet        data/behavior/behavior_marker_usage_qwen2.5_math.parquet

输入：
  - completions_parquet: eval_generations_all.parquet
      需要至少有列：["id", "model_name", "task", "output"]
  - marker_json: Phase 2.2 的输出，结构大致为：
      {
        "decomp": [
          {"token_str": "ĠTherefore", "token_id": 123, "lor": 2.3, ...},
          ...
        ],
        "verify": [...],
        "backtrack": [...],
        "reflect": [...]
      }

输出：
  - out_parquet: 一张 token 级聚合表，每行一个 (behavior, token_str)：

      behavior_type | token_str | token_id | lor | c_base | c_rl |
      total_tokens_base | total_tokens_rl | freq_base | freq_rl | log_freq_ratio_rl_vs_base
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--completions_parquet",
        type=str,
        required=True,
        help="包含 base / RL 输出的 completions 表（eval_generations_all.parquet）",
    )
    ap.add_argument(
        "--marker_json",
        type=str,
        required=True,
        help="Phase 2.2 的行为 marker token JSON 文件路径",
    )
    ap.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="在 completions 表中作为 base 的 model_name",
    )
    ap.add_argument(
        "--rl_model_name",
        type=str,
        required=True,
        help="在 completions 表中作为 RL 的 model_name",
    )
    ap.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="HF tokenizer 路径（Qwen2.5-7B 的 checkpoint 目录即可）",
    )
    ap.add_argument(
        "--task_prefix",
        type=str,
        default="",
        help="可选：只保留 task 以这些前缀开头的样本，逗号分隔；为空则不过滤。",
    )
    ap.add_argument(
        "--behaviors",
        type=str,
        default="decomp,verify,backtrack,reflect",
        help="要分析的行为类型，逗号分隔（需与 marker_json 中的 key 对齐）",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="每个行为只保留前 top_k 个 marker token（按 lor 降序）；<=0 表示全部使用。",
    )
    ap.add_argument(
        "--out_parquet",
        type=str,
        required=True,
        help="输出 parquet 文件路径",
    )
    return ap.parse_args()


def load_marker_tokens(marker_json_path: str,
                       behaviors: List[str],
                       top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    从 JSON 里加载每个行为的 marker token 列表，并按 top_k 截断。

    返回：
      {
        "decomp": [
          {"token_str": str, "token_id": int or None, "lor": float or None, ...},
          ...
        ],
        ...
      }
    """
    with open(marker_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    marker_by_behavior: Dict[str, List[Dict[str, Any]]] = {}
    for b in behaviors:
        if b not in data:
            print(f"[WARN] behavior '{b}' not found in marker_json, skip it.", file=sys.stderr)
            continue
        lst = data[b]

        # 尝试根据 lor 排序（如果存在），否则按原顺序
        def _lor_key(x: Dict[str, Any]) -> float:
            v = x.get("lor", 0.0)
            try:
                return float(v)
            except Exception:
                return 0.0

        lst_sorted = sorted(lst, key=_lor_key, reverse=True)
        if top_k > 0 and len(lst_sorted) > top_k:
            lst_sorted = lst_sorted[:top_k]

        marker_by_behavior[b] = lst_sorted
        print(f"[INFO] behavior={b}: loaded {len(lst_sorted)} marker tokens (top_k={top_k})")

    return marker_by_behavior


def main():
    args = parse_args()

    # ------------------------------
    # 1. 加载 marker tokens
    # ------------------------------
    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    marker_by_behavior = load_marker_tokens(args.marker_json, behaviors, args.top_k)

    if not marker_by_behavior:
        print("[ERROR] No behaviors loaded from marker_json. Exit.", file=sys.stderr)
        sys.exit(1)

    # 为快速查找，构建：behavior -> {token_str -> marker_info}
    marker_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for b, lst in marker_by_behavior.items():
        d = {}
        for item in lst:
            ts = item.get("token_str")
            if ts is None:
                continue
            d[str(ts)] = item
        marker_maps[b] = d
        print(f"[INFO] behavior={b}: marker set size = {len(d)}")

    # ------------------------------
    # 2. 加载 completions
    # ------------------------------
    print(f"[INFO] Loading completions from {args.completions_parquet}")
    df = pd.read_parquet(args.completions_parquet)

    required_cols = {"id", "model_name", "task", "output"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"completions_parquet missing columns: {missing}")

    # 只保留 base / RL 两种模型
    df = df[df["model_name"].isin([args.base_model_name, args.rl_model_name])]
    print(f"[INFO] After model_name filter (base+RL), rows = {len(df)}")

    if args.task_prefix:
        prefixes = [p.strip() for p in args.task_prefix.split(",") if p.strip()]

        def keep_task(t: str) -> bool:
            return any(str(t).startswith(p) for p in prefixes)

        df = df[df["task"].astype(str).apply(keep_task)]
        print(f"[INFO] After task_prefix filter, rows = {len(df)}")

    if df.empty:
        print("[ERROR] No data left after filtering. Exit.")
        sys.exit(1)

    # ------------------------------
    # 3. 加载 tokenizer（统一用 base 对应的 tokenizer）
    # ------------------------------
    print(f"[INFO] Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # ------------------------------
    # 4. 统计 token 频率
    # ------------------------------
    # 计数器：
    #   stats[(behavior, token_str)] = {
    #       "token_id": ...,
    #       "lor": ...,
    #       "c_base": ...,
    #       "c_rl": ...
    #   }
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

    total_tokens_base = 0
    total_tokens_rl = 0

    # 初始化：把所有 marker 先写进 stats，计数为 0
    for b, m_map in marker_maps.items():
        for token_str, info in m_map.items():
            key = (b, token_str)
            if key not in stats:
                stats[key] = {
                    "behavior_type": b,
                    "token_str": token_str,
                    "token_id": info.get("token_id", None),
                    "lor": info.get("lor", None),
                    "c_base": 0,
                    "c_rl": 0,
                }

    # 遍历所有 completions，tokenize & 计数
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Count marker usage", ncols=100):
        model_name = row["model_name"]
        output = row["output"]
        if not isinstance(output, str) or not output.strip():
            continue

        # tokenize：不加 special tokens，避免 [BOS]/[EOS] 干扰
        enc = tokenizer(
            output,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        if model_name == args.base_model_name:
            total_tokens_base += len(tokens)
            which = "c_base"
        elif model_name == args.rl_model_name:
            total_tokens_rl += len(tokens)
            which = "c_rl"
        else:
            # 理论上不会走到这里
            continue

        # 对每个行为分别检查 marker hit
        for t in tokens:
            t_str = str(t)
            for b, m_map in marker_maps.items():
                if t_str in m_map:
                    key = (b, t_str)
                    stats[key][which] += 1

    print(f"[INFO] total_tokens_base = {total_tokens_base}")
    print(f"[INFO] total_tokens_rl   = {total_tokens_rl}")

    if total_tokens_base == 0 or total_tokens_rl == 0:
        print("[WARN] total_tokens_base or total_tokens_rl is 0, results may be meaningless.", file=sys.stderr)

    # ------------------------------
    # 5. 计算频率与 log 频率比
    # ------------------------------
    records: List[Dict[str, Any]] = []
    eps = 1e-8

    for key, info in stats.items():
        b, token_str = key
        c_base = int(info.get("c_base", 0))
        c_rl = int(info.get("c_rl", 0))

        if total_tokens_base > 0:
            freq_base = c_base / float(total_tokens_base)
        else:
            freq_base = 0.0

        if total_tokens_rl > 0:
            freq_rl = c_rl / float(total_tokens_rl)
        else:
            freq_rl = 0.0

        # log 频率比（RL vs base）
        log_ratio = math.log((freq_rl + eps) / (freq_base + eps))

        records.append(
            {
                "behavior_type": b,
                "token_str": token_str,
                "token_id": info.get("token_id", None),
                "lor": info.get("lor", None),
                "c_base": c_base,
                "c_rl": c_rl,
                "total_tokens_base": total_tokens_base,
                "total_tokens_rl": total_tokens_rl,
                "freq_base": freq_base,
                "freq_rl": freq_rl,
                "log_freq_ratio_rl_vs_base": log_ratio,
            }
        )

    out_df = pd.DataFrame.from_records(records)

    # ------------------------------
    # 6. 保存结果
    # ------------------------------
    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)
    print(f"[OK] Saved marker usage diff to {args.out_parquet} (rows={len(out_df)})")

    # 顺便打印一下每个行为 top 几个 RL 更偏好的 token 预览
    for b in behaviors:
        sub = out_df[out_df["behavior_type"] == b].copy()
        if sub.empty:
            print(f"[INFO] behavior={b}: no records.")
            continue
        sub = sub.sort_values("log_freq_ratio_rl_vs_base", ascending=False)
        print(f"\n[PREVIEW] behavior={b}: top 10 RL > base marker tokens")
        print(sub[["token_str", "c_base", "c_rl", "freq_base", "freq_rl", "log_freq_ratio_rl_vs_base"]].head(10))


if __name__ == "__main__":
    main()