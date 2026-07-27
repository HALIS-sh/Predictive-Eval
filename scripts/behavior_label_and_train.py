#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_behavior_label_and_train.py

目标：
  1) 对模型在 probe 上的生成结果做“推理行为”分析——四类习惯：
       - decomposition / stepwise (分步拆解)
       - verification / checking (自检/校验)
       - backtracking / correction (回退/纠错)
       - reflection / strategy (反思/策略说明)
  2) 用一组可解释的规则做初始探测（presence + counts）
  3) （可选）在有少量人工/LLM 标注的前提下，训练一个轻量 classifier
     对这些规则进行校正，得到更稳定的 habit 预测。

输入：
  - generations_path: 包含模型在 probe 上生成的答案，要求至少列：
        [id, model_name, task, output]
    id: 与 probe.jsonl 中对齐的样本 id
    model_name: 模型标识符，例如 "Qwen2.5-1.5B-Instruct"
    task: "math" / "coding" / "logic" / ...
    output: 模型在该 probe 上的生成文本（包含完整 chain-of-thought）

  - （可选）label_path：带有四习惯真值标注的 CSV，用于训练轻量分类器：
        [id, decomp, verify, backtrack, reflect]
    取值 0/1，多模型共享同一标注（只与样本文本绑定）。

输出：
  1) behavior_sample_feats.parquet
     每 (model_name, id) 一行，包含：
        - 四习惯 presence & counts
        - 分段（early/mid/late）上的 counts
        - 一些结构性统计（行数、步骤数等）

  2) behavior_model_task_feats.parquet
     每 (model_name, task) 一行，包含：
        - 各习惯在该 task 下的出现比例（有/无）
        - 平均 counts
        - 早段 vs 末段的趋势指标（e.g., verify_late_ratio）

  3) （可选）behavior_clf.json
     若提供 label_path，则训练 LogisticRegression，用规则特征 → 习惯真值，
     存下每个习惯的权重与评估指标（precision/recall/F1）。

用法示例：

  # 仅用规则打行为特征
  python scripts/03_behavior_label_and_train.py \
    --generations_path data/outputs/generations.parquet \
    --output_sample_feats data/outputs/behavior_sample_feats.parquet \
    --output_model_feats data/outputs/behavior_model_task_feats.parquet

  # 若有人工/LLM 标注，用于训练/校正
  python scripts/03_behavior_label_and_train.py \
    --generations_path data/outputs/generations.parquet \
    --label_path data/outputs/behavior_labels.csv \
    --output_sample_feats data/outputs/behavior_sample_feats.parquet \
    --output_model_feats data/outputs/behavior_model_task_feats.parquet \
    --clf_out data/outputs/behavior_clf.json
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


# ------------------------- CLI ------------------------- #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--generations_path", type=str, required=True,
        help="Parquet file with model generations: columns=[id, model_name, task, output]"
    )
    ap.add_argument(
        "--label_path", type=str, default=None,
        help="Optional CSV with habit labels: columns=[id, decomp, verify, backtrack, reflect]"
    )
    ap.add_argument(
        "--output_sample_feats", type=str, required=True,
        help="Output parquet for per-sample behavior features"
    )
    ap.add_argument(
        "--output_model_feats", type=str, required=True,
        help="Output parquet for per-(model, task) behavior features"
    )
    ap.add_argument(
        "--clf_out", type=str, default=None,
        help="Optional JSON to save trained habit classifiers' weights/metrics"
    )
    return ap.parse_args()


# ------------------------- 规则定义 ------------------------- #

# 为了便于稍后展示，我们用 dataclass 封装四个习惯的规则关键词
@dataclass
class HabitRuleConfig:
    name: str
    # 触发“分步/结构化思考”的片段
    decomp_patterns: List[str]
    # 触发“验证/检查”的片段
    verify_patterns: List[str]
    # 触发“回退/纠错”的片段
    backtrack_patterns: List[str]
    # 触发“反思/策略/总结”的片段
    reflect_patterns: List[str]


def default_habit_rules() -> HabitRuleConfig:
    # 所有匹配在 lowercased 文本上进行
    return HabitRuleConfig(
        name="default",
        decomp_patterns=[
            r"\bstep\b", r"step\s*\d+", r"\bfirst\b", r"\bsecond\b", r"\bthird\b",
            r"then we", r"next we", r"subproblem", r"sub-goal", r"subgoal"
        ],
        verify_patterns=[
            r"check", r"double[- ]check", r"verify", r"let'?s see if",
            r"confirm", r"ensure that", r"to make sure", r"re[- ]check"
        ],
        backtrack_patterns=[
            r"wait", r"that'?s wrong", r"that was wrong", r"this is wrong",
            r"i made a mistake", r"mistake", r"let'?s try again",
            r"backtrack", r"reconsider", r"restart", r"correct this"
        ],
        reflect_patterns=[
            r"in conclusion", r"to summarize", r"summary", r"overall",
            r"our plan", r"the key idea", r"strategy", r"we can see that",
            r"this means that", r"hence", r"therefore"
        ],
    )


# ------------------------- 文本切分 & 规则探测 ------------------------- #

def split_into_phases(text: str, num_phases: int = 3) -> Dict[str, str]:
    """
    简单按行数把文本切成 early/mid/late 三段。
    如果行数太少，会自动退化（early=前半，late=后半）。
    """
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return {"early": "", "mid": "", "late": ""}

    n = len(lines)
    if n < 3:
        # 两行以内就干脆前半/后半
        half = n // 2 or 1
        early = "\n".join(lines[:half])
        late = "\n".join(lines[half:])
        return {"early": early, "mid": "", "late": late}

    # 正常分三段
    k = n // num_phases
    early = "\n".join(lines[:k])
    mid = "\n".join(lines[k: 2 * k])
    late = "\n".join(lines[2 * k:])
    return {"early": early, "mid": mid, "late": late}


def count_patterns(text: str, patterns: List[str]) -> int:
    """
    在 lowercased 文本上，统计所有 regex pattern 的匹配总和。
    """
    if not text:
        return 0
    text_l = text.lower()
    total = 0
    for pat in patterns:
        try:
            m = re.findall(pat, text_l)
            total += len(m)
        except re.error:
            # 正则写错也不影响整体
            continue
    return total


def extract_behavior_features_for_text(
    text: str,
    rules: HabitRuleConfig,
) -> Dict[str, Any]:
    """
    对单条生成文本，计算四种习惯的 presence + counts，以及 early/mid/late 分布。
    """
    phases = split_into_phases(text)
    early, mid, late = phases["early"], phases["mid"], phases["late"]

    # 分步：
    decomp_cnt_all = count_patterns(text, rules.decomp_patterns)
    decomp_cnt_early = count_patterns(early, rules.decomp_patterns)
    decomp_cnt_mid = count_patterns(mid, rules.decomp_patterns)
    decomp_cnt_late = count_patterns(late, rules.decomp_patterns)

    # 校验：
    verify_cnt_all = count_patterns(text, rules.verify_patterns)
    verify_cnt_early = count_patterns(early, rules.verify_patterns)
    verify_cnt_mid = count_patterns(mid, rules.verify_patterns)
    verify_cnt_late = count_patterns(late, rules.verify_patterns)

    # 回退：
    backtrack_cnt_all = count_patterns(text, rules.backtrack_patterns)
    backtrack_cnt_early = count_patterns(early, rules.backtrack_patterns)
    backtrack_cnt_mid = count_patterns(mid, rules.backtrack_patterns)
    backtrack_cnt_late = count_patterns(late, rules.backtrack_patterns)

    # 反思/策略：
    reflect_cnt_all = count_patterns(text, rules.reflect_patterns)
    reflect_cnt_early = count_patterns(early, rules.reflect_patterns)
    reflect_cnt_mid = count_patterns(mid, rules.reflect_patterns)
    reflect_cnt_late = count_patterns(late, rules.reflect_patterns)

    # 一些结构性统计：行数、token 近似数
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    num_lines = len(lines)
    approx_tokens = len(text.split())

    feats = {
        # 简单结构特征
        "num_lines": num_lines,
        "approx_tokens": approx_tokens,
        # Decomposition
        "decomp_cnt_all": decomp_cnt_all,
        "decomp_cnt_early": decomp_cnt_early,
        "decomp_cnt_mid": decomp_cnt_mid,
        "decomp_cnt_late": decomp_cnt_late,
        "decomp_present": int(decomp_cnt_all > 0),
        # Verification
        "verify_cnt_all": verify_cnt_all,
        "verify_cnt_early": verify_cnt_early,
        "verify_cnt_mid": verify_cnt_mid,
        "verify_cnt_late": verify_cnt_late,
        "verify_present": int(verify_cnt_all > 0),
        # Backtracking
        "backtrack_cnt_all": backtrack_cnt_all,
        "backtrack_cnt_early": backtrack_cnt_early,
        "backtrack_cnt_mid": backtrack_cnt_mid,
        "backtrack_cnt_late": backtrack_cnt_late,
        "backtrack_present": int(backtrack_cnt_all > 0),
        # Reflection
        "reflect_cnt_all": reflect_cnt_all,
        "reflect_cnt_early": reflect_cnt_early,
        "reflect_cnt_mid": reflect_cnt_mid,
        "reflect_cnt_late": reflect_cnt_late,
        "reflect_present": int(reflect_cnt_all > 0),
    }
    return feats


# ------------------------- 轻量 classifier 训练（可选） ------------------------- #

HABIT_NAMES = ["decomp", "verify", "backtrack", "reflect"]


def build_feature_matrix(sample_df: pd.DataFrame) -> np.ndarray:
    """
    将 rule-based 特征抽成一个矩阵 X，用于训练轻量 LogisticRegression。
    这里我们只用 counts + 简单结构特征，避免维度过高。
    """
    feat_cols = [
        "num_lines", "approx_tokens",
        "decomp_cnt_all", "decomp_cnt_early", "decomp_cnt_mid", "decomp_cnt_late",
        "verify_cnt_all", "verify_cnt_early", "verify_cnt_mid", "verify_cnt_late",
        "backtrack_cnt_all", "backtrack_cnt_early", "backtrack_cnt_mid", "backtrack_cnt_late",
        "reflect_cnt_all", "reflect_cnt_early", "reflect_cnt_mid", "reflect_cnt_late",
    ]
    # 缺失填 0
    X = sample_df[feat_cols].fillna(0.0).values.astype(float)
    return X, feat_cols


def train_habit_classifiers(
    sample_feats: pd.DataFrame,
    label_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    若提供 label_path，则对四个习惯分别训练 LogisticRegression：
      输入：rule-based feature
      输出：habit presence (0/1)
    返回每个习惯的系数和评价指标。
    """
    # label_df 列: [id, decomp, verify, backtrack, reflect]
    label_df = label_df.copy()
    label_df["id"] = label_df["id"].astype(str)

    sample = sample_feats.copy()
    sample["id"] = sample["id"].astype(str)

    merged = sample.merge(label_df, on="id", how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping ids between sample feats and label CSV.")

    X, feat_cols = build_feature_matrix(merged)

    clf_info: Dict[str, Any] = {}
    for habit in HABIT_NAMES:
        y = merged[habit].astype(int).values
        if len(np.unique(y)) < 2:
            print(f"[WARN] Habit '{habit}': labels are not both 0/1, skip classifier training.")
            continue

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        pred = clf.predict(X)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )

        clf_info[habit] = {
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "features": feat_cols,
            "metrics_train": {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            },
        }
        print(
            f"[INFO] Trained classifier for habit '{habit}': "
            f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}"
        )

    return clf_info


# ------------------------- 聚合到 (model, task) 级别 ------------------------- #

def aggregate_to_model_task(sample_feats: pd.DataFrame) -> pd.DataFrame:
    """
    将 per-sample 的行为特征聚合到 (model_name, task) 级别。
    输出指标包括：
      - 各习惯 present 比例（多少比例的样本触发了该习惯）
      - 各习惯 counts 的平均值
      - Verification 在 late 段出现的比例（末段验证习惯）
      - 平均行数、平均长度
    """
    group_cols = ["model_name", "task"]
    dfs = sample_feats.copy()

    agg_dict = {
        "num_lines": "mean",
        "approx_tokens": "mean",
        # presence：用 mean 表示“触发比例”
        "decomp_present": "mean",
        "verify_present": "mean",
        "backtrack_present": "mean",
        "reflect_present": "mean",
        # counts：用 mean 表示平均强度
        "decomp_cnt_all": "mean",
        "verify_cnt_all": "mean",
        "backtrack_cnt_all": "mean",
        "reflect_cnt_all": "mean",
        # late verification ratio：verify_cnt_late / verify_cnt_all
    }

    # 先算基本均值聚合
    agg = dfs.groupby(group_cols).agg(agg_dict).reset_index()
    # 重命名部分列名便于理解
    agg = agg.rename(columns={
        "num_lines": "avg_num_lines",
        "approx_tokens": "avg_approx_tokens",
        "decomp_present": "decomp_present_ratio",
        "verify_present": "verify_present_ratio",
        "backtrack_present": "backtrack_present_ratio",
        "reflect_present": "reflect_present_ratio",
        "decomp_cnt_all": "decomp_cnt_all_mean",
        "verify_cnt_all": "verify_cnt_all_mean",
        "backtrack_cnt_all": "backtrack_cnt_all_mean",
        "reflect_cnt_all": "reflect_cnt_all_mean",
    })

    # 额外计算一个“后段验证比例”指标：在有 verify 的样本中，late 占比
    def late_verify_ratio(sub: pd.DataFrame) -> float:
        has_any = sub["verify_cnt_all"] > 0
        sub = sub[has_any]
        if sub.empty:
            return float("nan")
        ratios = sub["verify_cnt_late"] / (sub["verify_cnt_all"] + 1e-6)
        return float(ratios.mean())

    ratios = []
    keys = []
    for (m, t), sub in dfs.groupby(group_cols):
        r = late_verify_ratio(sub)
        keys.append((m, t))
        ratios.append(r)

    ratio_df = pd.DataFrame(keys, columns=group_cols)
    ratio_df["verify_late_ratio"] = ratios

    agg = agg.merge(ratio_df, on=group_cols, how="left")

    return agg


# ------------------------- 主流程 ------------------------- #

def main():
    args = parse_args()

    gen_path = Path(args.generations_path)
    out_sample = Path(args.output_sample_feats)
    out_model = Path(args.output_model_feats)
    clf_out = Path(args.clf_out) if args.clf_out else None
    label_path = Path(args.label_path) if args.label_path else None

    print(f"[INFO] Loading generations from {gen_path}")
    df = pd.read_parquet(gen_path)

    required_cols = {"id", "model_name", "task", "output"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Generations parquet must contain columns {required_cols}, got {df.columns}")

    df["id"] = df["id"].astype(str)
    df["model_name"] = df["model_name"].astype(str)
    df["task"] = df["task"].astype(str)
    df["output"] = df["output"].fillna("")

    rules = default_habit_rules()

    # -------- 1. 对每条样本做规则探测，生成 sample-level 特征 -------- #
    sample_feat_rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        text = row["output"]
        feats = extract_behavior_features_for_text(text, rules)
        rec = {
            "id": row["id"],
            "model_name": row["model_name"],
            "task": row["task"],
        }
        rec.update(feats)
        sample_feat_rows.append(rec)

        if (idx + 1) % 1000 == 0:
            print(f"[INFO] Processed {idx + 1} samples for behavior features...")

    sample_feats = pd.DataFrame(sample_feat_rows)
    out_sample.parent.mkdir(parents=True, exist_ok=True)
    sample_feats.to_parquet(out_sample, index=False)
    print(f"[OK] Wrote per-sample behavior features to {out_sample} (rows={len(sample_feats)})")

    # -------- 2. 可选：训练轻量 classifier 用标注做校正 -------- #
    clf_info: Optional[Dict[str, Any]] = None
    if label_path is not None and label_path.exists():
        print(f"[INFO] Loading labels from {label_path} for classifier training")
        label_df = pd.read_csv(label_path)
        # 确认列
        expected_label_cols = {"id", "decomp", "verify", "backtrack", "reflect"}
        if not expected_label_cols.issubset(label_df.columns):
            raise ValueError(
                f"Label CSV must contain columns {expected_label_cols}, "
                f"got {label_df.columns}"
            )
        clf_info = train_habit_classifiers(sample_feats, label_df)

        if clf_out is not None:
            clf_out.parent.mkdir(parents=True, exist_ok=True)
            with open(clf_out, "w", encoding="utf-8") as f:
                json.dump(clf_info, f, ensure_ascii=False, indent=2)
            print(f"[OK] Saved habit classifier info to {clf_out}")
    else:
        print("[INFO] No label_path provided or file not exists; skip classifier training.")

    # （当前脚本先不对 sample_feats 做 classifier 校正预测，
    #  后续你可以根据 clf_info 把 presence 替换成 classifier 输出的 prob>0.5）

    # -------- 3. 聚合到 (model_name, task) 级别 -------- #
    model_task_feats = aggregate_to_model_task(sample_feats)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model_task_feats.to_parquet(out_model, index=False)
    print(f"[OK] Wrote per-(model, task) behavior features to {out_model} "
          f"(rows={len(model_task_feats)})")

    print("[DONE] 03_behavior_label_and_train.py finished.")


if __name__ == "__main__":
    main()