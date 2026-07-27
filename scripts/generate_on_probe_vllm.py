#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_on_probe_vllm.py

用 vLLM 在 probe 上生成 CoT，用于后续行为特征抽取。

输入：
  - --model_path:  HF / 本地权重路径（和 Alpha-RL 的 data_eval 一样）
  - --model_name:  用在输出表里的模型名（列 model_name）
  - --probe_path:  build_probe.py 生成的 probe.jsonl
  - --output_path: 生成结果保存的 parquet 路径
  - 其它采样 / batch / vLLM 参数见 argparse

输出（parquet，每行一个 probe 样本）：
  - id
  - model_name
  - task
  - dataset
  - prompt
  - output   （模型生成完整文本）

用法示例：

  python scripts/generate_on_probe_vllm.py \
    --model_path /data/.../Qwen/Qwen3-8B \
    --model_name Qwen3-8B \
    --probe_path data/probe/probe.jsonl \
    --output_path data/completions/model/Qwen/Qwen3-8B/probe_generations.parquet \
    --batch_size 8 --max_new_tokens 768
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from vllm import LLM, SamplingParams


# ---------- 读取 probe ----------

def load_probe(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True,
                    help="HF id 或本地权重路径（与 Alpha-RL data_eval 用法一致）")
    ap.add_argument("--model_name", type=str, required=True,
                    help="写入结果表中的模型名（例如 Qwen3-8B）")
    ap.add_argument("--probe_path", type=str, required=True,
                    help="build_probe.py 生成的 probe.jsonl")
    ap.add_argument("--output_path", type=str, required=True,
                    help="输出 parquet 路径")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--tensor_parallel_size", type=int, default=None,
                    help="vLLM tensor parallel size；默认自动按 CUDA_VISIBLE_DEVICES 个数推断")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096,
                    help="vLLM 的 max_model_len（context 长度上限）")
    return ap.parse_args()


def infer_with_vllm(args):
    # ---------- vLLM 初始化 ----------
    tp_size = args.tensor_parallel_size
    if tp_size is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            tp_size = len([x for x in cuda_visible.split(",") if x.strip() != ""])
        else:
            tp_size = 1

    print(f"[INFO] Using tensor_parallel_size = {tp_size}")

    print(f"[INFO] Loading model from {args.model_path} ...")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tokenizer_mode="auto",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        # 这里不加 stop，让模型完整输出推理 + 最后一行 ### <answer>
        stop=None,
    )

    # ---------- 读取 probe ----------
    print(f"[INFO] Loading probe from {args.probe_path}")
    probes = load_probe(args.probe_path)
    print(f"[INFO] Loaded {len(probes)} probe samples.")

    records = []
    bs = args.batch_size

    for start in range(0, len(probes), bs):
        batch = probes[start:start + bs]
        prompts = [ex["prompt"] for ex in batch]

        # vLLM generate
        completions = llm.generate(prompts, sampling_params)

        for ex, completion in zip(batch, completions):
            # 单样本 n=1
            out_text = completion.outputs[0].text

            rec = {
                "id": ex.get("id"),
                "model_name": args.model_name,
                "task": ex.get("task", "math"),
                "dataset": ex.get("dataset"),
                "prompt": ex.get("prompt"),
                "output": out_text,
            }
            records.append(rec)

        done = min(start + bs, len(probes))
        if done % 100 == 0 or done == len(probes):
            print(f"[INFO] Generated {done}/{len(probes)} samples ...")

    # ---------- 写 parquet ----------
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote {len(df)} generations to {out_path}")


def main():
    args = parse_args()
    infer_with_vllm(args)


if __name__ == "__main__":
    main()