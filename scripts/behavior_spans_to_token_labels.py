#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
behavior_spans_to_token_labels.py

把字符级行为 span（来自 behavior_llm_labels.parquet）对齐到 tokenizer token 上，
输出 token 级别的行为标签表。

输入：
  --completions_parquet   e.g. data/completions/eval_generations_all.parquet
      需要列：id, model_name, task, output

  --labels_parquet        e.g. data/behavior/behavior_llm_labels.parquet
      需要列：id, model_name, task, behavior_type, spans_json, output_len

输出：
  --out_parquet           e.g. data/behavior/behavior_token_labels.parquet
      每行一个 token：
        id, model_name, task,
        token_idx, token_id, token_str,
        char_start, char_end,
        is_decomp, is_verify, is_backtrack, is_reflect
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


# 你需要根据实际模型名，填充这个映射
# key: 在 completions/labels 里的 model_name
# value: HF 上的 tokenizer 名称（或本地路径）
TOKENIZER_NAME_MAP: Dict[str, str] = {
    # 示例：
    "Qwen2.5-7B": "/data/wenhesun/model/Qwen/Qwen2.5-7B",
    "Qwen2.5-Math-7B": "/data/wenhesun/model/Qwen/Qwen2.5-Math-7B",
    "Qwen3-8B": "/data/wenhesun/model/Qwen/Qwen3-8B",
    "Llama3.1-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
}


BEHAVIOR_TYPES = ["decomp", "verify", "backtrack", "reflect"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--completions_parquet",
        type=str,
        required=True,
        help="包含模型输出的 parquet，通常是 eval_generations_all.parquet",
    )
    ap.add_argument(
        "--labels_parquet",
        type=str,
        required=True,
        help="字符级行为标注表 behavior_llm_labels.parquet",
    )
    ap.add_argument(
        "--out_parquet",
        type=str,
        required=True,
        help="输出 token 级行为标签表的 parquet 路径",
    )
    return ap.parse_args()


def load_tokenizer(model_name: str, cache: Dict[str, Any]) -> Any:
    """根据 model_name 加载/复用 HF tokenizer."""
    if model_name in cache:
        return cache[model_name]

    if model_name not in TOKENIZER_NAME_MAP:
        raise KeyError(
            f"TOKENIZER_NAME_MAP 中没有为 model_name='{model_name}' 配置 tokenizer 名称，"
            "请在脚本顶部的 TOKENIZER_NAME_MAP 中添加映射。"
        )
    tok_name = TOKENIZER_NAME_MAP[model_name]
    print(f"[INFO] Loading tokenizer for model_name={model_name} -> {tok_name}")
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    cache[model_name] = tok
    return tok


def spans_overlap(token_span: Tuple[int, int], span: Tuple[int, int]) -> bool:
    """判断 token 的 [s1, e1) 与 标注 span 的 [s2, e2) 是否有交集."""
    s1, e1 = token_span
    s2, e2 = span
    if e1 <= s2:
        return False
    if e2 <= s1:
        return False
    return True


def main():
    args = parse_args()

    print(f"[INFO] Loading completions from {args.completions_parquet}")
    df_comp = pd.read_parquet(args.completions_parquet)

    print(f"[INFO] Loading labels from {args.labels_parquet}")
    df_labels = pd.read_parquet(args.labels_parquet)

    # 统一 id 类型为字符串，避免 int/str 对不上
    df_comp["id"] = df_comp["id"].astype(str)
    df_labels["id"] = df_labels["id"].astype(str)

    required_comp_cols = {"id", "model_name", "task", "output"}
    missing = required_comp_cols - set(df_comp.columns)
    if missing:
        raise ValueError(f"completions parquet 缺少列: {missing}")

    required_label_cols = {
        "id",
        "model_name",
        "task",
        "behavior_type",
        "spans_json",
    }
    missing2 = required_label_cols - set(df_labels.columns)
    if missing2:
        raise ValueError(f"labels parquet 缺少列: {missing2}")

    # 为了加速查找，把 completions 按 (id, model_name, task) 建索引
    comp_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for _, row in df_comp.iterrows():
        key = (str(row["id"]), row["model_name"], row["task"])
        # 如果一个 (id,model,task) 有多条输出，这里只保留第一条（如有多样本请自行调整逻辑）
        if key not in comp_index:
            comp_index[key] = {
                "output": row["output"],
            }

    # 只保留在 completions 里也存在的 labels
    def has_comp(row):
        key = (row["id"], row["model_name"], row["task"])
        return key in comp_index

    df_labels = df_labels[df_labels.apply(has_comp, axis=1)]
    print(f"[INFO] After aligning with completions, labels rows = {len(df_labels)}")

    if df_labels.empty:
        print("[ERROR] No overlapping (id, model_name, task) between labels and completions.")
        sys.exit(1)

    # tokenizer 缓存
    tok_cache: Dict[str, Any] = {}

    token_records: List[Dict[str, Any]] = []

    # groupby 每个 (id, model_name, task)，一次 tokenize，然后打四种行为的标签
    grouped = df_labels.groupby(["id", "model_name", "task"])

    for (sid, model_name, task), g in tqdm(
        grouped, total=len(grouped), desc="Align spans to tokens", ncols=100
    ):
        key = (sid, model_name, task)
        output = comp_index[key]["output"]
        if not isinstance(output, str):
            continue

        # 加载 tokenizer
        try:
            tok = load_tokenizer(model_name, tok_cache)
        except Exception as e:
            print(f"[WARN] Skip sample {key} because tokenizer load failed: {e}", file=sys.stderr)
            continue

        # tokenize 输出文本，拿 offset_mapping
        enc = tok(
            output,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        n_tok = len(input_ids)
        if n_tok != len(offsets):
            print(
                f"[WARN] offsets length mismatch for sample {key}: "
                f"{n_tok} tokens vs {len(offsets)} offsets",
                file=sys.stderr,
            )
            continue

        # 初始化每个 token 的行为标记
        flags = {
            "decomp": [0] * n_tok,
            "verify": [0] * n_tok,
            "backtrack": [0] * n_tok,
            "reflect": [0] * n_tok,
        }

        # 对当前 sample 下，所有行为的 span 打标
        for _, row in g.iterrows():
            behavior = row["behavior_type"]
            if behavior not in BEHAVIOR_TYPES:
                continue
            spans = []
            try:
                spans = json.loads(row["spans_json"])
            except Exception:
                spans = []

            for sp in spans:
                try:
                    s = int(sp["start_char"])
                    e = int(sp["end_char"])
                except Exception:
                    continue
                if not (0 <= s < e):
                    continue
                span_range = (s, e)

                # 给所有与 span 有交集的 token 打上标记
                for i, off in enumerate(offsets):
                    ts, te = off
                    # 有些 special token 的 offset 可能是 (0,0)，可以视作不参与
                    if ts == te:
                        continue
                    if spans_overlap((ts, te), span_range):
                        flags[behavior][i] = 1

        # 把每个 token 写成一行记录
        tokens_str = tok.convert_ids_to_tokens(input_ids)

        for i, (tid, off, tstr) in enumerate(zip(input_ids, offsets, tokens_str)):
            cs, ce = off
            token_records.append(
                {
                    "id": sid,
                    "model_name": model_name,
                    "task": task,
                    "token_idx": i,
                    "token_id": int(tid),
                    "token_str": tstr,
                    "char_start": int(cs),
                    "char_end": int(ce),
                    "is_decomp": int(flags["decomp"][i]),
                    "is_verify": int(flags["verify"][i]),
                    "is_backtrack": int(flags["backtrack"][i]),
                    "is_reflect": int(flags["reflect"][i]),
                }
            )

    out_df = pd.DataFrame.from_records(token_records)
    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)
    print(f"[OK] Saved token-level behavior labels to {args.out_parquet} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
