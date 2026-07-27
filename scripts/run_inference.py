# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Run unified inference on probe.jsonl with a local causal LM.

# For each probe sample:
#   - Build full_text = prompt  (if answer is None)
#                   or prompt + "\n### " + answer
#   - Tokenize and run the model in teacher-forcing mode
#   - Compute per-token negative log-likelihood (NLL)
#   - Tag each token with simple category flags:
#       * math_token
#       * code_token
#       * logic_token
#       * tool_token
#       * reason_token   (reasoning directives / structure)
#   - Mark answer vs context tokens

# Output: a Parquet file with one row per probe:
#   {
#     id, task, dataset, split,
#     text, prompt_len,
#     input_ids, tokens, nll,
#     is_answer_mask,
#     math_mask, code_mask, logic_mask, tool_mask, reason_mask,
#     mean_nll, mean_nll_answer_only (if answer exists)
#   }

# Usage:
#   python run_inference.py \
#       --model_path /path/to/local/model \
#       --probe_path data/probe/probe.jsonl \
#       --output_path data/infer/my_model.probe.parquet \
#       --batch_size 4 --max_length 2048
# """

# import argparse
# import json
# import math
# from pathlib import Path
# from typing import List, Dict, Any

# import torch
# from torch.nn import functional as F
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer


# # ========= Special-token vocabulary heuristics ========= #

# MATH_CHARS = set("0123456789+-*/=^%()[]{}<>.,:;|&π∞√±÷×")
# MATH_WORDS = {
#     "sum", "product", "difference", "equation", "equations", "fraction",
#     "probability", "integral", "derivative", "limit", "theorem", "lemma",
#     "corollary", "proof", "value", "variable", "expression"
# }

# CODE_KEYWORDS = {
#     # Python-ish
#     "def", "class", "for", "while", "if", "else", "elif", "return",
#     "try", "except", "finally", "with", "as", "lambda", "yield",
#     "import", "from", "global", "nonlocal", "pass", "break", "continue",
#     "True", "False", "None",
#     # Generic
#     "function", "var", "let", "const", "public", "private", "static",
# }

# CODE_SYMBOLS = set("(){}[]:.,=+-*/%<>!|&")

# TOOL_WORDS = {
#     "tool", "tools", "api", "call", "tool_call", "function_call", "browser",
#     "search", "query", "url", "http", "https", "request", "env",
#     "action", "observation", "plan", "execute", "step", "agent", "assistant",
# }

# LOGIC_WORDS = {
#     "if", "then", "else", "and", "or", "not", "xor", "therefore", "hence",
#     "thus", "because", "so", "implies", "imply", "equivalent", "iff",
#     "assume", "assumption", "suppose", "contradiction", "conclude",
# }

# REASON_WORDS = {
#     "let's", "lets", "reason", "think", "step", "steps",
#     "first", "second", "third", "next", "finally",
#     "in", "conclusion", "summary", "summarize", "overall",
# }

# REASON_PHRASE_FRAGMENTS = [
#     "let's think step by step",
#     "let us think",
#     "let's reason step by step",
#     "we can reason",
#     "step by step",
#     "the key idea",
#     "our plan is",
# ]


# def clean_token(tok: str) -> str:
#     # strip common BPE prefixes and spaces, lowercase
#     tok = tok.replace("Ġ", " ").replace("▁", " ")
#     tok = tok.strip()
#     return tok.lower()


# def categorize_token(tok: str) -> Dict[str, int]:
#     """
#     Very simple heuristic categorization.
#     Returns dict with 0/1 flags:
#       math, code, logic, tool, reason
#     """
#     s = clean_token(tok)
#     if not s:
#         return {"math": 0, "code": 0, "logic": 0, "tool": 0, "reason": 0}

#     # math
#     is_math = any(ch in MATH_CHARS for ch in s) or s in MATH_WORDS

#     # code
#     is_code = (s in CODE_KEYWORDS) or any(ch in CODE_SYMBOLS for ch in s)
#     if s.startswith("```") or s in {"python", "java", "c++", "c", "rust"}:
#         is_code = 1

#     # tool / agent
#     is_tool = any(w in s for w in TOOL_WORDS)

#     # logic
#     is_logic = s in LOGIC_WORDS or s in {"=>", "->", "⇒", "→", "∴"}

#     # reasoning directive / structure
#     is_reason = s in REASON_WORDS or any(frag in s for frag in ["step", "reason"])
#     for frag in REASON_PHRASE_FRAGMENTS:
#         if frag in s:
#             is_reason = 1
#             break

#     return {
#         "math": int(is_math),
#         "code": int(is_code),
#         "logic": int(is_logic),
#         "tool": int(is_tool),
#         "reason": int(is_reason),
#     }


# # ========= Core inference ========= #

# def load_probe(path: str) -> List[Dict[str, Any]]:
#     rows = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             rows.append(json.loads(line))
#     return rows


# def build_text(prompt: str, answer: str | None, answer_tag: str = "###") -> (str, str):
#     """
#     Return (full_text, prompt_only_text).
#     If answer is present, append '\n### <answer>'.
#     """
#     prompt = prompt.rstrip()
#     if answer is None or answer == "":
#         return prompt, prompt
#     full = f"{prompt}\n{answer_tag} {answer}"
#     return full, prompt


# def run_model_on_batch(
#     model, tokenizer, texts: List[str], device: torch.device, max_length: int
# ):
#     """
#     Returns:
#       input_ids: list[list[int]]
#       tokens:    list[list[str]]
#       nll:       list[list[float]]  (same length as tokens, first token NLL is nan)
#     """
#     enc = tokenizer(
#         texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=max_length,
#         add_special_tokens=False,
#     )
#     input_ids = enc["input_ids"].to(device)
#     attention_mask = enc["attention_mask"].to(device)

#     with torch.no_grad():
#         out = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = out.logits  # [B, T, V]

#     # shift for causal loss: predict token t using logits at t-1
#     shift_logits = logits[:, :-1, :].contiguous()
#     shift_labels = input_ids[:, 1:].contiguous()
#     shift_mask = attention_mask[:, 1:].contiguous()

#     log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
#     gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B,T-1]
#     nll_shift = -gathered  # positive
#     nll_shift = nll_shift * shift_mask + (1 - shift_mask) * 0.0

#     # now pad back to length T (put nan for first position)
#     B, T = input_ids.shape
#     nll_full = torch.full((B, T), float("nan"), device=device)
#     nll_full[:, 1:] = nll_shift

#     input_ids_list = input_ids.tolist()
#     nll_list = nll_full.tolist()
#     tokens_list: List[List[str]] = [
#         tokenizer.convert_ids_to_tokens(seq) for seq in input_ids_list
#     ]
#     return input_ids_list, tokens_list, nll_list, enc["attention_mask"].tolist()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True,
#                         help="Local path or HF id of a causal LM")
#     parser.add_argument("--probe_path", type=str, required=True,
#                         help="probe.jsonl built by build_probe.py")
#     parser.add_argument("--output_path", type=str, required=True,
#                         help="Output Parquet path")
#     parser.add_argument("--batch_size", type=int, default=2)
#     parser.add_argument("--max_length", type=int, default=2048)
#     parser.add_argument("--answer_tag", type=str, default="###",
#                         help="Tag used when concatenating answers, default '###'")
#     parser.add_argument("--device", type=str, default="auto",
#                         help="auto / cpu / cuda")
#     args = parser.parse_args()

#     # ---------- device ----------
#     if args.device == "auto":
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(args.device)

#     # ---------- model & tokenizer ----------
#     print(f"[INFO] Loading model from {args.model_path} on {device} ...")
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_path,
#         use_fast=True,
#         trust_remote_code=True,   # 关键修改：兼容 Qwen 等自定义代码
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_path,
#         torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
#         device_map=None,
#         trust_remote_code=True,   # 关键修改：兼容 Qwen 等自定义代码
#     ).to(device)
#     model.eval()

#     # ---------- load probe ----------
#     print(f"[INFO] Loading probe from {args.probe_path}")
#     probes = load_probe(args.probe_path)
#     print(f"[INFO] Loaded {len(probes)} samples")

#     # ---------- iterate in batches ----------
#     records = []

#     bs = args.batch_size
#     for i in range(0, len(probes), bs):
#         batch = probes[i: i + bs]
#         texts_full = []
#         texts_prompt = []
#         for ex in batch:
#             full, prompt_only = build_text(ex["prompt"], ex.get("answer"), args.answer_tag)
#             texts_full.append(full)
#             texts_prompt.append(prompt_only)

#         input_ids_list, tokens_list, nll_list, attn_mask_list = run_model_on_batch(
#             model, tokenizer, texts_full, device, args.max_length
#         )

#         # compute prompt lengths (in tokens) separately
#         prompt_lens = [
#             len(
#                 tokenizer(
#                     p,
#                     add_special_tokens=False,
#                     truncation=True,
#                     max_length=args.max_length,
#                 )["input_ids"]
#             )
#             for p in texts_prompt
#         ]

#         for ex, text, ids, toks, nll, attn, p_len in zip(
#             batch, texts_full, input_ids_list, tokens_list, nll_list, attn_mask_list, prompt_lens
#         ):
#             seq_len = sum(attn)
#             ids = ids[:seq_len]
#             toks = toks[:seq_len]
#             nll = nll[:seq_len]

#             # masks
#             is_answer_mask = [int(idx >= p_len) for idx in range(seq_len)]

#             math_mask, code_mask, logic_mask, tool_mask, reason_mask = [], [], [], [], []
#             for t in toks:
#                 cat = categorize_token(t)
#                 math_mask.append(cat["math"])
#                 code_mask.append(cat["code"])
#                 logic_mask.append(cat["logic"])
#                 tool_mask.append(cat["tool"])
#                 reason_mask.append(cat["reason"])

#             # mean losses
#             valid_nll = [x for x in nll if not math.isnan(x)]
#             mean_nll = float(sum(valid_nll) / max(1, len(valid_nll)))
#             ans_nll_vals = [x for x, m in zip(nll, is_answer_mask) if m == 1 and not math.isnan(x)]
#             mean_nll_ans = float(sum(ans_nll_vals) / max(1, len(ans_nll_vals))) if ans_nll_vals else float("nan")

#             rec = {
#                 "id": ex.get("id"),
#                 "task": ex.get("task"),
#                 "dataset": ex.get("dataset"),
#                 "split": ex.get("split"),
#                 "text": text,
#                 "prompt_len": int(p_len),
#                 "input_ids": ids,
#                 "tokens": toks,
#                 "nll": nll,
#                 "is_answer_mask": is_answer_mask,
#                 "math_mask": math_mask,
#                 "code_mask": code_mask,
#                 "logic_mask": logic_mask,
#                 "tool_mask": tool_mask,
#                 "reason_mask": reason_mask,
#                 "mean_nll": mean_nll,
#                 "mean_nll_answer_only": mean_nll_ans,
#             }
#             records.append(rec)

#         if (i // bs) % 10 == 0:
#             print(f"[INFO] processed {min(i+bs, len(probes))}/{len(probes)}")

#     # ---------- write parquet ----------
#     out_path = Path(args.output_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     df = pd.DataFrame.from_records(records)
#     df.to_parquet(out_path, index=False)
#     print(f"[OK] wrote {len(records)} rows to {out_path}")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run unified inference on probe.jsonl with a local causal LM.

For each probe sample:
  - Build full_text = prompt  (if answer is None)
                  or prompt + "\n### " + answer
  - Tokenize and run the model in teacher-forcing mode
  - Compute per-token negative log-likelihood (NLL)
  - Tag each token with simple category flags:
      * math_token
      * code_token
      * logic_token
      * tool_token
      * reason_token   (reasoning directives / structure)
  - Mark answer vs context tokens
  - Classify answer type: none / final_only / cot_or_long
  - Accumulate math-vocabulary statistics from math answers.

Output:
  - A Parquet file with one row per probe:
      {
        id, task, dataset, split,
        text, prompt_len,
        input_ids, tokens, nll,
        is_answer_mask,
        math_mask, code_mask, logic_mask, tool_mask, reason_mask,
        mean_nll, mean_nll_answer_only (if answer exists),
        answer_type
      }
  - Optionally, a JSON file with refined math vocabulary candidates:
      <output_path>.math_vocab.json

Usage:
  python run_inference.py \
      --model_path /path/to/local/model \
      --probe_path data/probe/probe.jsonl \
      --output_path data/infer/my_model.probe.parquet \
      --batch_size 4 --max_length 2048
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.nn import functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========= Special-token vocabulary heuristics ========= #

MATH_CHARS = set("0123456789+-*/=^%()[]{}<>.,:;|&π∞√±÷×\\")
MATH_WORDS = {
    "sum", "product", "difference", "equation", "equations", "fraction",
    "probability", "integral", "derivative", "limit", "theorem", "lemma",
    "corollary", "proof", "value", "variable", "expression",
    "triangle", "angle", "radius", "circle", "line", "segment",
}

CODE_KEYWORDS = {
    # Python-ish
    "def", "class", "for", "while", "if", "else", "elif", "return",
    "try", "except", "finally", "with", "as", "lambda", "yield",
    "import", "from", "global", "nonlocal", "pass", "break", "continue",
    "true", "false", "none",
    # Generic
    "function", "var", "let", "const", "public", "private", "static",
}

CODE_SYMBOLS = set("(){}[]:.,=+-*/%<>!|&")

TOOL_WORDS = {
    "tool", "tools", "api", "call", "tool_call", "function_call", "browser",
    "search", "query", "url", "http", "https", "request", "env",
    "action", "observation", "plan", "execute", "step", "agent", "assistant",
}

LOGIC_WORDS = {
    "if", "then", "else", "and", "or", "not", "xor", "therefore", "hence",
    "thus", "because", "so", "implies", "imply", "equivalent", "iff",
    "assume", "assumption", "suppose", "contradiction", "conclude",
}

REASON_WORDS = {
    "let's", "lets", "reason", "think", "step", "steps",
    "first", "second", "third", "next", "finally",
    "in", "conclusion", "summary", "summarize", "overall",
}

REASON_PHRASE_FRAGMENTS = [
    "let's think step by step",
    "let us think",
    "let's reason step by step",
    "we can reason",
    "step by step",
    "the key idea",
    "our plan is",
]

STOPWORDS = {
    "the", "a", "an", "of", "to", "is", "are", "in", "on", "at", "for",
    "this", "that", "it", "its", "be", "by", "as", "we", "you", "i",
}


def clean_token(tok: str) -> str:
    """Strip common BPE prefixes/spaces and lowercase."""
    tok = tok.replace("Ġ", " ").replace("▁", " ")
    tok = tok.strip()
    return tok.lower()


def categorize_token(tok: str) -> Dict[str, int]:
    """
    Very simple heuristic categorization.
    Returns dict with 0/1 flags:
      math, code, logic, tool, reason
    """
    s = clean_token(tok)
    if not s:
        return {"math": 0, "code": 0, "logic": 0, "tool": 0, "reason": 0}

    # math
    is_math = any(ch in MATH_CHARS for ch in s) or s in MATH_WORDS

    # code
    is_code = (s in CODE_KEYWORDS) or any(ch in CODE_SYMBOLS for ch in s)
    if s.startswith("```") or s in {"python", "java", "c++", "c", "rust"}:
        is_code = 1

    # tool / agent
    is_tool = any(w in s for w in TOOL_WORDS)

    # logic
    is_logic = s in LOGIC_WORDS or s in {"=>", "->", "⇒", "→", "∴"}

    # reasoning directive / structure
    is_reason = s in REASON_WORDS or any("step" in s or "reason" in s for _ in [0])
    for frag in REASON_PHRASE_FRAGMENTS:
        if frag in s:
            is_reason = 1
            break

    return {
        "math": int(is_math),
        "code": int(is_code),
        "logic": int(is_logic),
        "tool": int(is_tool),
        "reason": int(is_reason),
    }


# ========= Answer-type heuristics ========= #

def classify_answer_text(ans) -> str:
    """
    Roughly classify answer into:
      - "none": no answer
      - "final_only": short numeric/choice-style answer without reasoning
      - "cot_or_long": contains words / multiple tokens → likely has reasoning

    ans 可能是 str/int/float/list 等，这里统一做成字符串再处理。
    """
    if ans is None:
        return "none"

    # 转成字符串
    if not isinstance(ans, str):
        try:
            ans = str(ans)
        except Exception:
            return "final_only"

    s = ans.strip()
    if not s:
        return "none"

    # 单个字母或 (A) 之类
    if re.fullmatch(r"[A-Za-z]|$begin:math:text$\[A\-E\]$end:math:text$", s):
        return "final_only"

    # 短数字 / 简单算式
    if len(s) <= 20 and re.fullmatch(r"[0-9\s\+\-\.,/]*", s):
        return "final_only"

    return "cot_or_long"


# ========= Core inference ========= #

def load_probe(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_text(prompt: str, answer, answer_tag: str = "###") -> Tuple[str, str]:
    """
    Return (full_text, prompt_only_text).
    If answer is present, append '\n### <answer>'.
    不人为构造 CoT，只使用验证集中的 answer 字段（优先是完整 solution）。
    """
    prompt = prompt.rstrip()
    if answer is None or answer == "":
        return prompt, prompt

    if not isinstance(answer, str):
        try:
            answer = str(answer)
        except Exception:
            return prompt, prompt

    full = f"{prompt}\n{answer_tag} {answer}"
    return full, prompt


def run_model_on_batch(
    model, tokenizer, texts: List[str], device: torch.device, max_length: int
):
    """
    Returns:
      input_ids: list[list[int]]
      tokens:    list[list[str]]
      nll:       list[list[float]]  (same length as tokens, first token NLL is nan)
      attention_mask: list[list[int]]
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]

    # shift for causal loss: predict token t using logits at t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
    gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B,T-1]
    nll_shift = -gathered
    nll_shift = nll_shift * shift_mask + (1 - shift_mask) * 0.0

    B, T = input_ids.shape
    nll_full = torch.full((B, T), float("nan"), device=device)
    nll_full[:, 1:] = nll_shift

    input_ids_list = input_ids.tolist()
    nll_list = nll_full.tolist()
    tokens_list: List[List[str]] = [
        tokenizer.convert_ids_to_tokens(seq) for seq in input_ids_list
    ]
    return input_ids_list, tokens_list, nll_list, enc["attention_mask"].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local path or HF id of a causal LM")
    parser.add_argument("--probe_path", type=str, required=True,
                        help="probe.jsonl built by build_probe.py")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output Parquet path")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--answer_tag", type=str, default="###",
                        help="Tag used when concatenating answers, default '###'")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto / cpu / cuda")
    parser.add_argument("--math_vocab_min_freq", type=int, default=5,
                        help="Min frequency for a token to be kept in math vocab candidates")
    parser.add_argument("--math_vocab_topk", type=int, default=200,
                        help="Top-K most frequent math vocab tokens to keep")
    parser.add_argument("--disable_math_vocab_dump", action="store_true",
                        help="If set, do not dump math vocab JSON alongside parquet")
    args = parser.parse_args()

    # ---------- device ----------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---------- model & tokenizer ----------
    print(f"[INFO] Loading model from {args.model_path} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # ---------- load probe ----------
    print(f"[INFO] Loading probe from {args.probe_path}")
    probes = load_probe(args.probe_path)
    print(f"[INFO] Loaded {len(probes)} samples")

    # ---------- iterate in batches ----------
    records = []
    math_vocab_counter: Counter[str] = Counter()

    bs = args.batch_size
    for i in range(0, len(probes), bs):
        batch = probes[i: i + bs]
        texts_full = []
        texts_prompt = []
        answer_types = []

        for ex in batch:
            ans = ex.get("answer")
            ans_type = classify_answer_text(ans)
            answer_types.append(ans_type)

            full, prompt_only = build_text(ex["prompt"], ans, args.answer_tag)
            texts_full.append(full)
            texts_prompt.append(prompt_only)

        input_ids_list, tokens_list, nll_list, attn_mask_list = run_model_on_batch(
            model, tokenizer, texts_full, device, args.max_length
        )

        # compute prompt lengths (in tokens) separately
        prompt_lens = [
            len(
                tokenizer(
                    p,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_length,
                )["input_ids"]
            )
            for p in texts_prompt
        ]

        for ex, text, ids, toks, nll, attn, p_len, ans_type in zip(
            batch, texts_full, input_ids_list, tokens_list, nll_list,
            attn_mask_list, prompt_lens, answer_types
        ):
            seq_len = sum(attn)
            ids = ids[:seq_len]
            toks = toks[:seq_len]
            nll = nll[:seq_len]

            # masks
            is_answer_mask = [int(idx >= p_len) for idx in range(seq_len)]

            math_mask, code_mask, logic_mask, tool_mask, reason_mask = [], [], [], [], []
            for t in toks:
                cat = categorize_token(t)
                math_mask.append(cat["math"])
                code_mask.append(cat["code"])
                logic_mask.append(cat["logic"])
                tool_mask.append(cat["tool"])
                reason_mask.append(cat["reason"])

            # ----- accumulate math vocab from solutions -----
            if ex.get("task") == "math":
                for t, is_ans in zip(toks, is_answer_mask):
                    if not is_ans:
                        continue
                    s = clean_token(t)
                    if not s:
                        continue
                    if s in STOPWORDS:
                        continue
                    if s.startswith("<") and s.endswith(">"):
                        continue

                    cond1 = any(ch in MATH_CHARS for ch in s) or ("\\" in s)
                    cond2 = s in MATH_WORDS
                    cond3 = s.isalpha() and len(s) >= 3

                    if cond1 or cond2 or cond3:
                        math_vocab_counter[s] += 1

            # mean losses
            valid_nll = [x for x in nll if not math.isnan(x)]
            mean_nll = float(sum(valid_nll) / max(1, len(valid_nll)))
            ans_nll_vals = [x for x, m in zip(nll, is_answer_mask) if m == 1 and not math.isnan(x)]
            mean_nll_ans = float(sum(ans_nll_vals) / max(1, len(ans_nll_vals))) if ans_nll_vals else float("nan")

            rec = {
                "id": ex.get("id"),
                "task": ex.get("task"),
                "dataset": ex.get("dataset"),
                "split": ex.get("split"),
                "text": text,
                "prompt_len": int(p_len),
                "input_ids": ids,
                "tokens": toks,
                "nll": nll,
                "is_answer_mask": is_answer_mask,
                "math_mask": math_mask,
                "code_mask": code_mask,
                "logic_mask": logic_mask,
                "tool_mask": tool_mask,
                "reason_mask": reason_mask,
                "mean_nll": mean_nll,
                "mean_nll_answer_only": mean_nll_ans,
                "answer_type": ans_type,
            }
            records.append(rec)

        if (i // bs) % 10 == 0:
            print(f"[INFO] processed {min(i+bs, len(probes))}/{len(probes)}")

    # ---------- write parquet ----------
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_path, index=False)
    print(f"[OK] wrote {len(records)} rows to {out_path}")

    # ---------- dump math vocab candidates ----------
    if not args.disable_math_vocab_dump and len(math_vocab_counter) > 0:
        min_freq = args.math_vocab_min_freq
        topk = args.math_vocab_topk

        filtered = [
            (tok, freq)
            for tok, freq in math_vocab_counter.most_common()
            if freq >= min_freq
        ][:topk]

        refined_vocab = sorted(set(MATH_WORDS).union({t for t, _ in filtered}))
        vocab_path = out_path.with_suffix(out_path.suffix + ".math_vocab.json")
        vocab_obj = {
            "base_math_words": sorted(MATH_WORDS),
            "candidate_counts": filtered,
            "refined_vocab": refined_vocab,
            "min_freq": min_freq,
            "topk": topk,
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_obj, f, ensure_ascii=False, indent=2)
        print(f"[OK] dumped math vocab stats to {vocab_path}")


if __name__ == "__main__":
    main()