#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
behavior_token_enrichment.py

在 token 级行为标签上，做行为相关 token 的富集分析，
为每种 behavior（decomp/verify/backtrack/reflect）找出“marker token”。

输入：
  --in_parquet    e.g. data/behavior/behavior_token_labels.parquet
      需要列：
        token_str,
        is_decomp, is_verify, is_backtrack, is_reflect

输出：
  --out_json      e.g. data/behavior/behavior_token_markers.json
      JSON 结构：
      {
        "decomp": [
          {"token_str": "...", "lor": float, "c_in": int, "c_out": int},
          ...
        ],
        "verify": [...],
        "backtrack": [...],
        "reflect": [...]
      }
"""

import argparse
import json
import math
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd


BEHAVIOR_TYPES = ["decomp", "verify", "backtrack", "reflect"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_parquet",
        type=str,
        required=True,
        help="token 级行为标签表（behavior_token_labels.parquet）",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="输出 marker token 的 JSON 文件路径",
    )
    ap.add_argument(
        "--min_freq",
        type=int,
        default=10,
        help="只考虑总频次 c_in + c_out >= min_freq 的 token_str",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="每种行为输出前 top_k 个 token 作为 marker",
    )
    return ap.parse_args()


def compute_markers_for_behavior(
    df: pd.DataFrame, behavior: str, min_freq: int, top_k: int
) -> List[Dict[str, Any]]:
    """
    对单个行为（e.g. 'decomp'），基于 is_decomp 列做 log-odds 富集分析。
    """
    col = f"is_{behavior}"
    if col not in df.columns:
        raise ValueError(f"列 {col} 不存在于输入表中。")

    # 行为内 / 行为外 token
    df_in = df[df[col] == 1]
    df_out = df[df[col] == 0]

    if df_in.empty or df_out.empty:
        print(f"[WARN] behavior={behavior} 没有足够的正/负样本，跳过。")
        return []

    # 按 token_str 计数
    c_in = df_in.groupby("token_str").size()
    c_out = df_out.groupby("token_str").size()

    C_in = c_in.sum()
    C_out = c_out.sum()

    # 合并所有 token_str
    all_tokens = set(c_in.index).union(set(c_out.index))

    records = []
    for tok in all_tokens:
        ci = int(c_in.get(tok, 0))
        co = int(c_out.get(tok, 0))
        if ci + co < min_freq:
            continue

        # 带 0.5 smoothing 的 log-odds ratio
        p_in = (ci + 0.5) / (C_in + 0.5)
        p_out = (co + 0.5) / (C_out + 0.5)
        lor = math.log(p_in / p_out)

        records.append(
            {
                "token_str": tok,
                "lor": float(lor),
                "c_in": ci,
                "c_out": co,
            }
        )

    # 按 lor 从大到小排序，取 top_k
    records.sort(key=lambda x: x["lor"], reverse=True)
    if top_k > 0 and len(records) > top_k:
        records = records[:top_k]

    return records


def main():
    args = parse_args()
    print(f"[INFO] Loading token-level labels from {args.in_parquet}")
    df = pd.read_parquet(args.in_parquet)

    required_cols = {"token_str"}
    for b in BEHAVIOR_TYPES:
        required_cols.add(f"is_{b}")

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入表缺少列: {missing}")

    # 确保 token_str 是字符串
    df["token_str"] = df["token_str"].astype(str)

    result: Dict[str, Any] = {}

    for behavior in BEHAVIOR_TYPES:
        print(f"[INFO] Computing markers for behavior={behavior} ...")
        markers = compute_markers_for_behavior(
            df, behavior, min_freq=args.min_freq, top_k=args.top_k
        )
        print(
            f"[INFO] behavior={behavior}, got {len(markers)} marker tokens "
            f"(min_freq={args.min_freq}, top_k={args.top_k})"
        )
        result[behavior] = markers

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved behavior marker tokens to {args.out_json}")


if __name__ == "__main__":
    main()