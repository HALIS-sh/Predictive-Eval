#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_on_probe.py

在 build_probe.py 生成的 probe.jsonl 上，让本地 Causal LM 生成一条完整回答
（含 chain-of-thought），并把结果存成一个统一的 parquet，方便后续做
behavior 分析 / RL 特征等。

输入：
  - --model_path : HF / 本地模型路径，如  /data/.../Qwen3-8B
  - --model_name : 结果里用来区分模型的名字，随便但要和 benchmark 里一致
  - --probe_path : probe.jsonl（build_probe.py 的输出）
  - --output_path: 输出 parquet 路径

输出 parquet 列大致为：
  - id        : 与 probe 中对齐
  - model_name
  - task      : "math" / "coding" / "logic" ...
  - dataset   : "gsm8k" / "MATH500" / ...
  - split     : "test" / ...
  - prompt    : 输入给模型的 prompt（英文）
  - output    : 模型生成的完整回答（不包含 prompt，只是 continuation）

用法示例：

  python scripts/generate_on_probe.py \
    --model_path /data/wenhesun/model/Qwen/Qwen3-8B \
    --model_name Qwen3-8B \
    --probe_path data/probe/probe.jsonl \
    --output_path data/completions/Qwen3-8B.probe_generations.parquet \
    --batch_size 4 \
    --max_new_tokens 512
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------- CLI ----------------- #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path", type=str, required=True,
        help="Local path or HF id of a causal LM",
    )
    ap.add_argument(
        "--model_name", type=str, required=True,
        help="Name written into the output parquet (e.g. Qwen3-8B)",
    )
    ap.add_argument(
        "--probe_path", type=str, required=True,
        help="probe.jsonl built by build_probe.py",
    )
    ap.add_argument(
        "--output_path", type=str, required=True,
        help="Output parquet path for generations",
    )
    ap.add_argument(
        "--batch_size", type=int, default=4,
        help="Generation batch size (number of prompts per forward)",
    )
    ap.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum number of new tokens to generate for each probe",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (0 -> greedy)",
    )
    ap.add_argument(
        "--top_p", type=float, default=0.95,
        help="Top-p nucleus sampling parameter",
    )
    ap.add_argument(
        "--device", type=str, default="auto",
        help="auto / cpu / cuda",
    )
    return ap.parse_args()


# ----------------- Helpers ----------------- #

def load_probe(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ----------------- Main ----------------- #

def main():
    args = parse_args()

    # ----- device -----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device = {device}")

    # ----- model & tokenizer -----
    print(f"[INFO] Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # ----- load probe -----
    print(f"[INFO] Loading probe from {args.probe_path}")
    probes = load_probe(args.probe_path)
    print(f"[INFO] Loaded {len(probes)} probe samples.")

    bs = args.batch_size
    all_records: List[Dict[str, Any]] = []

    for start in range(0, len(probes), bs):
        batch = probes[start:start + bs]
        prompts = [ex["prompt"] for ex in batch]

        # tokenize batch
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,          # 只限制输入长度，若需要可改成参数
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        input_lengths = attention_mask.sum(dim=1).tolist()  # 每条 prompt 的 token 长度

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = gen_ids.cpu()

        for ex, ids, in_len in zip(batch, gen_ids, input_lengths):
            ids = ids.tolist()
            # 只取新增部分作为 output
            gen_part = ids[in_len:]
            output_text = tokenizer.decode(gen_part, skip_special_tokens=True)

            rec = {
                "id": ex.get("id"),
                "model_name": args.model_name,
                "task": ex.get("task"),
                "dataset": ex.get("dataset"),
                "split": ex.get("split"),
                "prompt": ex.get("prompt"),
                "output": output_text,
            }
            all_records.append(rec)

        if (start // bs) % 10 == 0:
            print(f"[INFO] Generated {min(start + bs, len(probes))}/{len(probes)} samples ...")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_records)
    df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote {len(df)} generations to {out_path}")


if __name__ == "__main__":
    main()