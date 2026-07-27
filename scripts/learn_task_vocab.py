#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn task-specific token vocabularies / salience from probe inference results.

Input:
  - A parquet file produced by run_inference.py, with columns:
      id, task, dataset, split, text, prompt_len,
      input_ids, tokens, nll,
      is_answer_mask,
      math_mask, code_mask, logic_mask, tool_mask, reason_mask,
      mean_nll, mean_nll_answer_only,
      answer_type  (none / final_only / cot_or_long)

Output:
  - A JSON file containing, for each task, the most "salient" tokens:
      {
        "math": [
          {
            "token": "Ġsin",
            "task_count": 120,
            "global_count": 130,
            "task_share": 0.923,
            "task_mean_nll": 1.23,
            "global_mean_nll": 1.56,
            "nll_diff": 0.33,
            "salience_score": 0.305
          },
          ...
        ],
        "coding": [...],
        "logic": [...]
      }

  You can later:
    - Inspect these tokens as "task vocabularies"
    - Use them as priors for CSV/capability salience training
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_path", type=str, required=True,
        help="Parquet file from run_inference.py"
    )
    ap.add_argument(
        "--output_path", type=str, required=True,
        help="Where to save learned vocab JSON"
    )
    ap.add_argument(
        "--tasks", type=str, default="",
        help="Comma-separated list of tasks to include (default: all in file), e.g. 'math,coding,logic'"
    )
    ap.add_argument(
        "--min_freq", type=int, default=20,
        help="Minimum global token frequency to consider"
    )
    ap.add_argument(
        "--top_k_per_task", type=int, default=200,
        help="How many top tokens (per task) to keep in the vocabulary"
    )
    ap.add_argument(
        "--use_answer_only", action="store_true",
        help="If set, only use tokens in the answer part (is_answer_mask=1) when computing stats"
    )
    ap.add_argument(
        "--only_cot_answers", action="store_true",
        help="If set, only use samples whose answer_type == 'cot_or_long'"
    )
    ap.add_argument(
        "--math_vocab_json", type=str, default="",
        help="Optional: path to *.math_vocab.json produced by run_inference.py"
    )
    ap.add_argument(
        "--restrict_math_to_refined_vocab", action="store_true",
        help="If set, for task=='math', only keep tokens in refined math vocab"
    )
    return ap.parse_args()


def masked_mean(x: pd.Series, mask: pd.Series) -> float:
    sel = x[mask.astype(bool)]
    return float(sel.mean()) if len(sel) > 0 else float("nan")


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading parquet from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"[INFO] Loaded {len(df)} rows")

    # ---- optional task filtering ----
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        before = len(df)
        df = df[df["task"].isin(wanted)]
        print(f"[INFO] Filtered tasks {wanted}: {before} -> {len(df)} rows")

    # ---- only use CoT / long answers if requested ----
    if args.only_cot_answers:
        if "answer_type" not in df.columns:
            print("[WARN] --only_cot_answers specified but 'answer_type' "
                  "column not found; this option will be ignored")
        else:
            before = len(df)
            df = df[df["answer_type"] == "cot_or_long"]
            print(f"[INFO] Kept only CoT/long-answer samples: {before} -> {len(df)} rows")

    # ---- optional refined math vocab ----
    math_vocab_set = None
    if args.math_vocab_json:
        mv_path = Path(args.math_vocab_json)
        if mv_path.is_file():
            print(f"[INFO] Loading math vocab JSON from {mv_path}")
            with open(mv_path, "r", encoding="utf-8") as f:
                mv_obj = json.load(f)
            # 优先使用 refined_vocab，如果没有就退回 candidate_counts
            refined = mv_obj.get("refined_vocab")
            if refined is not None:
                math_vocab_set = set(refined)
            else:
                cand = mv_obj.get("candidate_counts", [])
                math_vocab_set = {t for t, _ in cand}
            print(f"[INFO] Loaded {len(math_vocab_set)} refined math vocab tokens")
        else:
            print(f"[WARN] math_vocab_json={mv_path} not found; ignore refined vocab")

    # ---- explode to token-level table ----
    cols_to_explode = [
        "input_ids", "tokens", "nll",
        "is_answer_mask",
        "math_mask", "code_mask", "logic_mask", "tool_mask", "reason_mask",
    ]

    for c in cols_to_explode:
        df[c] = df[c].apply(list)

    print("[INFO] Exploding to token-level rows...")
    tok_df = df[["task", "dataset"] + cols_to_explode].explode(
        cols_to_explode, ignore_index=True
    )

    # cast types
    tok_df["tokens"] = tok_df["tokens"].astype(str)
    tok_df["nll"] = pd.to_numeric(tok_df["nll"], errors="coerce")
    tok_df["is_answer_mask"] = tok_df["is_answer_mask"].astype(int)

    # drop tokens with NaN nll (first position of each sequence)
    before = len(tok_df)
    tok_df = tok_df.dropna(subset=["nll"])
    print(f"[INFO] Dropped NaN-nll tokens: {before} -> {len(tok_df)} token rows")

    # if only answer tokens are desired
    if args.use_answer_only:
        before = len(tok_df)
        tok_df = tok_df[tok_df["is_answer_mask"] == 1]
        print(f"[INFO] Kept only answer tokens: {before} -> {len(tok_df)} token rows")

    # 对 math 任务可选地限制到 refined math vocab
    if args.restrict_math_to_refined_vocab and math_vocab_set is not None:
        before = len(tok_df)
        mask_keep = (tok_df["task"] != "math") | (tok_df["tokens"].isin(math_vocab_set))
        tok_df = tok_df[mask_keep]
        print(f"[INFO] Restrict math tokens to refined vocab: {before} -> {len(tok_df)} token rows")
    elif args.restrict_math_to_refined_vocab and math_vocab_set is None:
        print("[WARN] --restrict_math_to_refined_vocab set but no math_vocab_set loaded; "
              "this option has no effect")

    # ---- global token stats (across all tasks) ----
    print("[INFO] Computing global token statistics...")
    global_stats = (
        tok_df
        .groupby("tokens")
        .agg(
            global_count=("nll", "size"),
            global_mean_nll=("nll", "mean"),
        )
        .reset_index()
    )

    # filter by global frequency
    before = len(global_stats)
    global_stats = global_stats[global_stats["global_count"] >= args.min_freq]
    print(f"[INFO] Global tokens with freq >= {args.min_freq}: {before} -> {len(global_stats)}")

    # ---- per-task token stats ----
    print("[INFO] Computing per-task token statistics...")
    task_stats = (
        tok_df
        .groupby(["task", "tokens"])
        .agg(
            task_count=("nll", "size"),
            task_mean_nll=("nll", "mean"),
        )
        .reset_index()
    )

    # join with global stats
    merged = task_stats.merge(global_stats, on="tokens", how="inner")
    # compute task_share & nll_diff & salience_score
    merged["task_share"] = merged["task_count"] / merged["global_count"]
    merged["nll_diff"] = merged["global_mean_nll"] - merged["task_mean_nll"]
    merged["salience_score"] = merged["task_share"] * merged["nll_diff"]

    # ---- build per-task vocab dict ----
    vocab = {}
    for task, sub in merged.groupby("task"):
        sub_sorted = sub.sort_values(
            by=["salience_score", "task_share", "task_count"],
            ascending=[False, False, False],
        ).head(args.top_k_per_task)

        records = []
        for _, row in sub_sorted.iterrows():
            rec = {
                "token": row["tokens"],
                "task_count": int(row["task_count"]),
                "global_count": int(row["global_count"]),
                "task_share": float(round(row["task_share"], 6)),
                "task_mean_nll": float(round(row["task_mean_nll"], 6)),
                "global_mean_nll": float(round(row["global_mean_nll"], 6)),
                "nll_diff": float(round(row["nll_diff"], 6)),
                "salience_score": float(round(row["salience_score"], 6)),
            }
            records.append(rec)

        vocab[task] = records
        print(f"[INFO] Task '{task}': kept top {len(records)} tokens")

    # ---- save JSON ----
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved task vocab to {output_path}")


if __name__ == "__main__":
    main()