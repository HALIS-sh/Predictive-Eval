#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a unified probe set for math / coding / logic with standardized schema:
{id, task, dataset, split, prompt, answer, meta: {...}}

- Prompts are written in ENGLISH.
- Required output format:
  * Math/Logic: The VERY LAST line must be exactly: `### <final_answer>`
  * Coding: First output ONLY one Python code block, then one final line: `### DONE`

Usage:
  python build_probe.py --config configs/data.yaml
"""

import argparse, json, random, re, sys, os, io
from pathlib import Path
from typing import List, Dict, Any
import yaml
from datasets import load_dataset

random.seed(42)

# =========================
# Global answer tag (can be overridden by config: answer_tag)
ANS_TAG = "###"
# =========================

# --------------------------
# Helpers
# --------------------------
def make_templates(ans_tag: str):
    """Always return EN templates, with an explicit answer tag requirement."""
    return {
        "math_prompt":
            "Reason step by step. Keep the reasoning concise.\n"
            f"Output format: the VERY LAST line must be `{ans_tag} <final_answer>` with no extra text.\n\n"
            "Problem:\n{q}",
        "code_prompt":
            "Write Python code to solve the following task. "
            "Output ONLY one Markdown code block (```python ... ```). "
            f"After the code block, output exactly one line: `{ans_tag} DONE`.\n\n"
            "Task:\n{q}",
        "logic_prompt":
            "Do logical reasoning and give a concise final answer.\n"
            f"Output format: the VERY LAST line must be `{ans_tag} <final_answer>` with no extra text.\n\n"
            "Question:\n{q}",
    }

def normalize_prompt(s: str) -> str:
    """Preserve newlines; collapse excessive spaces and blank lines."""
    # collapse multiple spaces but keep newlines
    s = re.sub(r"[ \t]+", " ", s)
    # at most two consecutive newlines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def norm_whitespace_for_dedup(s: str) -> str:
    """For dedup only: collapse to single-line signature."""
    return re.sub(r"\s+", " ", s).strip().lower()

def bucketer(length: int, edges: List[int]) -> int:
    for i, e in enumerate(edges):
        if length <= e:
            return i
    return len(edges)

def stratified_sample(rows: List[Dict[str, Any]], k: int, edges: List[int], key="prompt"):
    if k <= 0 or len(rows) <= k:
        return rows
    buckets = {}
    for r in rows:
        b = bucketer(len(r[key]), edges)
        buckets.setdefault(b, []).append(r)
    per = max(1, k // max(1, len(buckets)))
    out = []
    for _, arr in buckets.items():
        random.shuffle(arr)
        out.extend(arr[:per])
    if len(out) < k:
        remain = [r for arr in buckets.values() for r in arr if r not in out]
        random.shuffle(remain)
        out.extend(remain[: k - len(out)])
    return out[:k]

def dedup_keep_order(items: List[Dict[str, Any]], key="prompt"):
    seen, out = set(), []
    for r in items:
        sig = norm_whitespace_for_dedup(r[key])
        if sig not in seen:
            out.append(r)
            seen.add(sig)
    return out

# --------------------------
# HF loaders (EN prompts with answer tag)
# --------------------------
def load_math_gsm8k(split="test") -> List[Dict[str, Any]]:
    T = make_templates(ANS_TAG)
    ds = load_dataset("gsm8k", "main")
    if split not in ds:
        split = "test" if "test" in ds else list(ds.keys())[0]
    out = []
    for ex in ds[split]:
        q = ex["question"]
        prompt = T["math_prompt"].format(q=q)
        ans = None
        m = re.search(r"####\s*([^\n]+)", ex.get("answer", ""))
        if m:
            ans = m.group(1).strip()
        out.append({
            "task": "math",
            "dataset": "gsm8k",
            "split": split,
            "prompt": normalize_prompt(prompt),
            "answer": ans,
            "meta": {"id": ex.get("id")}
        })
    return out

def load_coding_mbpp(split="test") -> List[Dict[str, Any]]:
    T = make_templates(ANS_TAG)
    try:
        ds = load_dataset("mbpp", "sanitized")
    except Exception:
        ds = load_dataset("mbpp")
    if split not in ds:
        split = "test" if "test" in ds else "train"
    out = []
    for i, ex in enumerate(ds[split]):
        q = ex.get("text") or ex.get("prompt") or ex.get("instructions") or ""
        code = ex.get("code")
        out.append({
            "task": "coding",
            "dataset": "mbpp",
            "split": split,
            "prompt": normalize_prompt(T["code_prompt"].format(q=q)),
            "answer": code,
            "meta": {"source_id": ex.get("task_id", i), "has_tests": bool(ex.get("test_list") or ex.get("test"))}
        })
    return out

def load_coding_humaneval(split="test") -> List[Dict[str, Any]]:
    T = make_templates(ANS_TAG)
    name_candidates = ["openai_humaneval", "nuprl/HumanEval", "openai/humaneval"]
    ds = None; err = None
    for n in name_candidates:
        try:
            ds = load_dataset(n)
            break
        except Exception as e:
            err = e
            continue
    if ds is None:
        raise RuntimeError(f"HumanEval not found on HF: last error {err}")
    if split not in ds:
        split = list(ds.keys())[0]
    out = []
    for ex in ds[split]:
        prompt = ex.get("prompt") or ex.get("task_id")
        out.append({
            "task": "coding",
            "dataset": "human_eval",
            "split": split,
            "prompt": normalize_prompt(T["code_prompt"].format(q=prompt)),
            "answer": ex.get("canonical_solution"),
            "meta": {"task_id": ex.get("task_id")}
        })
    return out

# --------------------------
# Local JSON helpers & loaders (MATH/AIME/OlympiadBench etc.)
# --------------------------
def _read_json_any(path: str):
    """Robust reader: JSON array or JSONL; return list[dict]."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        data = json.loads(txt)
        return list(data) if isinstance(data, (list, tuple)) else [data]
    except Exception:
        out = []
        for line in io.StringIO(txt):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

def load_local_math500(cfg):
    T = make_templates(ANS_TAG)
    rows = []
    for ex in _read_json_any(cfg["path"]):
        q = ex.get("problem") or ""
        ans = ex.get("answer")
        rows.append({
            "task": "math",
            "dataset": "MATH500",
            "split": "test",
            "prompt": normalize_prompt(T["math_prompt"].format(q=q)),
            "answer": ans,
            "meta": {"subject": ex.get("subject"), "level": ex.get("level"), "uid": ex.get("unique_id")}
        })
    return rows

def load_local_odyssey(cfg):
    T = make_templates(ANS_TAG)
    rows = []
    for ex in _read_json_any(cfg["path"]):
        q = ex.get("problem") or ex.get("question") or ""
        ans = ex.get("answer")
        rows.append({
            "task": "math",
            "dataset": "Odyssey",
            "split": "test",
            "prompt": normalize_prompt(T["math_prompt"].format(q=q)),
            "answer": ans,
            "meta": {}
        })
    return rows

def load_local_olympiadbench(cfg):
    T = make_templates(ANS_TAG)
    rows = []
    for ex in _read_json_any(cfg["path"]):
        q = ex.get("question") or ex.get("problem") or ""
        fa = ex.get("final_answer")
        ans = fa[0] if isinstance(fa, (list, tuple)) and fa else (fa or ex.get("answer"))
        rows.append({
            "task": "math",
            "dataset": "OlympiadBench",
            "split": "test",
            "prompt": normalize_prompt(T["math_prompt"].format(q=q)),
            "answer": ans,
            "meta": {}
        })
    return rows

def load_local_aime2024(cfg):
    T = make_templates(ANS_TAG)
    rows = []
    for ex in _read_json_any(cfg["path"]):
        q = ex.get("problem") or ""
        ans = ex.get("answer")
        rows.append({
            "task": "math",
            "dataset": "AIME2024",
            "split": "test",
            "prompt": normalize_prompt(T["math_prompt"].format(q=q)),
            "answer": ans,
            "meta": {"index": ex.get("index"), "url": ex.get("url")}
        })
    return rows

# --------------------------
# ARC-AGI-2 (logic) loader
# --------------------------
def _encode_grid_ascii(grid):
    lines = []
    for row in grid:
        lines.append(" ".join(str(int(x)) for x in row))
    return "\n".join(lines)

def _format_arc_prompt(fewshots, q_grid):
    head = (
        "You will see several solved examples, each with an input grid and its output grid. "
        "Infer the transformation rule, then apply the same rule to the QUESTION grid.\n"
        f"Output format: the VERY LAST line must be `{ANS_TAG} <json_2d_array>` with no extra text.\n\n"
    )
    parts = [head]
    for i, ex in enumerate(fewshots, 1):
        parts.append(f"Example {i} – Input:\n{_encode_grid_ascii(ex['input'])}")
        parts.append(f"Output:\n{_encode_grid_ascii(ex['output'])}\n")
    parts.append(f"QUESTION:\n{_encode_grid_ascii(q_grid)}")
    parts.append("\nReturn the answer grid as a JSON 2D integer array.")
    return "\n".join(parts)

def load_logic_arc_agi_2(cfg):
    name_candidates = ["arc-agi-community/arc-agi-2", "arc-agi-2"]
    ds = None; last_err = None
    for n in name_candidates:
        try:
            ds = load_dataset(n)
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"Cannot load ARC-AGI-2 from HF. Last error: {last_err}")

    split = cfg.get("split", "train")
    if split not in ds:
        split = list(ds.keys())[0]
    k_shot = int(cfg.get("k_fewshots", -1))

    out_rows = []
    for idx, ex in enumerate(ds[split]):
        fewshots = ex.get("fewshots") or []
        questions = ex.get("question") or []
        fs = fewshots if (k_shot < 0 or k_shot >= len(fewshots)) else fewshots[:k_shot]
        for qi, q in enumerate(questions):
            q_inp = q.get("input")
            if q_inp is None:
                continue
            prompt = normalize_prompt(_format_arc_prompt(fs, q_inp))
            gt = q.get("output")
            ans = json.dumps(gt) if gt is not None else None
            out_rows.append({
                "task": "logic",
                "dataset": "arc-agi-2",
                "split": split,
                "prompt": prompt,
                "answer": ans,
                "meta": {"arc_k_fewshots": len(fs), "arc_idx": idx, "arc_qi": qi}
            })
    return out_rows

# --------------------------
# Loader registry
# --------------------------
LOADERS = {
    # HF
    "gsm8k": lambda cfg: load_math_gsm8k(split=cfg.get("split","test")),
    "mbpp": lambda cfg: load_coding_mbpp(split=cfg.get("split","test")),
    "openai_humaneval": lambda cfg: load_coding_humaneval(split=cfg.get("split","test")),
    "arc-agi-2": lambda cfg: load_logic_arc_agi_2(cfg),
    "arc-agi-community/arc-agi-2": lambda cfg: load_logic_arc_agi_2(cfg),
    # Local JSONs
    "local_math500": lambda cfg: load_local_math500(cfg),
    "local_odyssey": lambda cfg: load_local_odyssey(cfg),
    "local_olympiadbench": lambda cfg: load_local_olympiadbench(cfg),
    "local_aime2024": lambda cfg: load_local_aime2024(cfg),
}

# --------------------------
# Section builder
# --------------------------
def build_from_section(section_cfg: Dict[str, Any], task_name: str,
                       length_edges: List[int], target_count: int) -> List[Dict[str, Any]]:
    rows_all: List[Dict[str, Any]] = []
    for ds_cfg in section_cfg.get("datasets", []):
        name = ds_cfg["name"]
        # alias
        if name == "human_eval":
            name = "openai_humaneval"
        if name not in LOADERS:
            print(f"[WARN] unknown dataset name: {name}, skip", file=sys.stderr); continue

        rows = LOADERS[name](ds_cfg)
        maxn = int(ds_cfg.get("max_per_dataset", 10**9))
        random.shuffle(rows)
        rows = rows[:maxn]
        for r in rows:
            r["task"] = task_name
        rows_all.extend(rows)

    rows_all = dedup_keep_order(rows_all, key="prompt")
    rows_all = stratified_sample(rows_all, target_count, edges=length_edges, key="prompt")

    # Attach phase hints (kept in EN; downstream can ignore)
    phase_tpl = ["Plan:", "Derivation:", "Verification:"]
    for i, r in enumerate(rows_all):
        r["id"] = f"{r['dataset']}::{r['split']}::{i}"
        r.setdefault("meta", {})
        r["meta"].update({
            "phase_template": phase_tpl,
            "lang": "en",
            "answer_tag": ANS_TAG
        })
    return rows_all

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    random.seed(cfg.get("seed", 42))

    # allow overriding answer tag via config
    global ANS_TAG
    ANS_TAG = cfg.get("answer_tag", ANS_TAG)

    length_edges = cfg.get("length_buckets", [20, 80, 200])
    out_path = Path(cfg.get("output_path", "data/probe/probe.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    target = int(cfg.get("samples_per_task", 300))

    if "math" in cfg:
        all_rows += build_from_section(cfg["math"], "math", length_edges, target)
    if "coding" in cfg:
        all_rows += build_from_section(cfg["coding"], "coding", length_edges, target)
    if "logic" in cfg:
        all_rows += build_from_section(cfg["logic"], "logic", length_edges, target)

    all_rows = dedup_keep_order(all_rows, key="prompt")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(all_rows)} probes to {out_path}  (answer_tag='{ANS_TAG}')")

if __name__ == "__main__":
    main()