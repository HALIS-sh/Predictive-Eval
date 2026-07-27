#!/usr/bin/env bash
set -e

# 1) 原始 verl ckpt 根目录（有 global_step_xxx 子目录）
SRC=/data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params

# 2) 想把 HF 格式模型放在哪里
OUT_ROOT=/data/wenhesun/hf_ckpts/qwen3_8b_grpo

mkdir -p "$OUT_ROOT"

# 3) 要导出的若干 step（自行修改）
STEPS="100 600 1100 1600 2100 2600 3100"

for step in $STEPS; do
  actor_dir=${SRC}/global_step_${step}/actor
  target_dir=${OUT_ROOT}/global_step_${step}

  if [ ! -f "${actor_dir}/fsdp_config.json" ]; then
    echo "[WARN] skip step ${step}: ${actor_dir}/fsdp_config.json not found"
    continue
  fi

  echo
  echo "[INFO] Merging FSDP ckpt step=${step}"
  echo "       local_dir  = ${actor_dir}"
  echo "       target_dir = ${target_dir}"

  mkdir -p "$target_dir"

  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$actor_dir" \
    --target_dir "$target_dir" \
    --private

  echo "[OK] exported HF model for step=${step} -> ${target_dir}"
done