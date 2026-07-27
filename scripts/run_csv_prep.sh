#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 MODEL_NAME MODEL_PATH IS_ANCHOR(0/1)"
  echo "Example: $0 Qwen3-8B /data/.../Qwen3-8B 1"
  exit 1
fi

MODEL_NAME="$1"      # e.g., Qwen3-8B, Llama-3.2-1B-Instruct
MODEL_PATH="$2"      # e.g., /data/.../model_dir
IS_ANCHOR="$3"       # 1 = anchor 模型；0 = 非 anchor

BASE_DIR="/data/wenhesun/Predictive-Eval"

# 注意：这里要和 build_probe 的 output_path 保持一致
# 你现在 build_probe 的默认输出就是 data/probe/probe.jsonl
PROBE_PATH="${BASE_DIR}/data/probe/probe.jsonl"

# 当前模型的推断输出
INFER_PATH="${BASE_DIR}/data/infer/${MODEL_NAME}.probe.parquet"

# anchor vocab
VOCAB_PATH="${BASE_DIR}/data/probe/task_vocab.anchor_math.json"

# build_probe 的配置文件
PROBE_CONFIG="${BASE_DIR}/configs/data.yaml"

echo "================ CSV 训练前半 pipeline ================"
echo "[INFO] MODEL_NAME = ${MODEL_NAME}"
echo "[INFO] MODEL_PATH = ${MODEL_PATH}"
echo "[INFO] IS_ANCHOR  = ${IS_ANCHOR}"
echo "[INFO] PROBE_PATH = ${PROBE_PATH}"
echo "[INFO] INFER_PATH = ${INFER_PATH}"
echo "[INFO] VOCAB_PATH = ${VOCAB_PATH}"
echo "======================================================="

########################################
# STEP 1: build_probe （只需一次）
########################################
if [ -f "${PROBE_PATH}" ]; then
  echo "[STEP 1] Probe already exists at ${PROBE_PATH}, skip build_probe."
else
  echo "[STEP 1] Building probe using config: ${PROBE_CONFIG}"
  python scripts/build_probe.py --config "${PROBE_CONFIG}"
fi

########################################
# STEP 2: run_inference （按你本地的 CLI）
########################################
echo "[STEP 2] Running probe inference for model ${MODEL_NAME} ..."
python scripts/run_inference.py \
  --model_path "${MODEL_PATH}" \
  --probe_path "${PROBE_PATH}" \
  --output_path "${INFER_PATH}" \
  --batch_size 4 \
  --max_length 2048

########################################
# STEP 3: learn_task_vocab（只在 anchor 上跑）
########################################
if [ "${IS_ANCHOR}" = "1" ]; then
  echo "[STEP 3] Anchor model: learning task vocab to ${VOCAB_PATH}"
  python scripts/learn_task_vocab.py \
    --input_path "${INFER_PATH}" \
    --output_path "${VOCAB_PATH}" \
    --tasks math \
    --use_answer_only \
    --min_freq 10 \
    --top_k_per_task 300
else
  echo "[STEP 3] Non-anchor model: skip learn_task_vocab (reuse anchor vocab: ${VOCAB_PATH})"
fi

echo "[DONE] run_csv_prep.sh finished for model ${MODEL_NAME}"