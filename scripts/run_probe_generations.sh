#!/usr/bin/env bash
set -euo pipefail

# 项目根目录（假设脚本在 scripts/ 目录下）
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

PROBE_PATH="$ROOT_DIR/data/probe/probe.jsonl"
OUT_ROOT="$ROOT_DIR/data/completions/model"

BATCH_SIZE=4
MAX_NEW_TOKENS=512
TEMP=0.2
TOP_P=0.95

echo "================ Generate on probe for multiple models ================"
echo "[INFO] ROOT_DIR    = ${ROOT_DIR}"
echo "[INFO] PROBE_PATH  = ${PROBE_PATH}"
echo "[INFO] OUT_ROOT    = ${OUT_ROOT}"
echo "======================================================================"

# 在这里列出要跑的模型：
# 每一行格式："<family> <model_name> <model_path>"
# family 用来对应你 data/completions/model 下的一级目录（meta-llama / Qwen 等）
MODELS=(
  "meta-llama Llama-3.2-1B-Instruct /data/wenhesun/model/meta-llama/Llama-3.2-1B-Instruct"
  "Qwen       Qwen2.5-1.5B-Instruct   /data/wenhesun/model/Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen       Qwen2.5-7B              /data/wenhesun/model/Qwen/Qwen2.5-7B"
  "Qwen       Qwen2.5-Math-7B         /data/wenhesun/model/Qwen/Qwen2.5-Math-7B"
  "Qwen       Qwen3-8B                /data/wenhesun/model/Qwen/Qwen3-8B"
)

for entry in "${MODELS[@]}"; do
  # 按空格拆成三个变量：family / model_name / model_path
  read -r FAMILY MODEL_NAME MODEL_PATH <<< "${entry}"

  OUT_DIR="${OUT_ROOT}/${FAMILY}/${MODEL_NAME}/probe"
  OUT_PATH="${OUT_DIR}/probe_generations.parquet"

  echo
  echo "------------------------------------------------------------"
  echo "[INFO] FAMILY      = ${FAMILY}"
  echo "[INFO] MODEL_NAME  = ${MODEL_NAME}"
  echo "[INFO] MODEL_PATH  = ${MODEL_PATH}"
  echo "[INFO] OUT_PATH    = ${OUT_PATH}"
  echo "------------------------------------------------------------"

  mkdir -p "${OUT_DIR}"

  python "${ROOT_DIR}/scripts/generate_on_probe.py" \
    --model_path  "${MODEL_PATH}" \
    --model_name  "${MODEL_NAME}" \
    --probe_path  "${PROBE_PATH}" \
    --output_path "${OUT_PATH}" \
    --batch_size  "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMP}" \
    --top_p "${TOP_P}"
done

echo
echo "[DONE] All models have generated probe completions."