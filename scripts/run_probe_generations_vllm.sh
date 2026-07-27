#!/usr/bin/env bash
set -euo pipefail

# 使用 vLLM 在 probe 上生成推理结果
# 仅需改下面 MODEL 列表中的 model_name / model_path 即可。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE_PATH="${REPO_ROOT}/data/probe/probe.jsonl"
OUT_ROOT="${REPO_ROOT}/data/completions/model"

BATCH_SIZE=8
MAX_NEW_TOKENS=768
TEMP=0.0
TOP_P=1.0
TOP_K=-1
GPU_UTIL=0.50
MAX_MODEL_LEN=10240

run_one_model() {
  local model_name="$1"
  local model_path="$2"

  # 从 model_path 推断出 vendor 子目录，例如：
  #   /data/.../model/meta-llama/Llama-3.2-1B-Instruct
  #     -> vendor = meta-llama
  #   /data/.../model/Qwen/Qwen3-8B
  #     -> vendor = Qwen
  local vendor
  vendor="$(basename "$(dirname "${model_path}")")"

  local out_dir="${OUT_ROOT}/${vendor}/${model_name}"
  local out_path="${out_dir}/probe_generations.parquet"

  echo "======================================================"
  echo "[INFO] model_name = ${model_name}"
  echo "[INFO] model_path = ${model_path}"
  echo "[INFO] vendor     = ${vendor}"
  echo "[INFO] probe_path = ${PROBE_PATH}"
  echo "[INFO] out_path   = ${out_path}"
  echo "======================================================"

  python "${REPO_ROOT}/scripts/generate_on_probe_vllm.py" \
    --model_path "${model_path}" \
    --model_name "${model_name}" \
    --probe_path "${PROBE_PATH}" \
    --output_path "${out_path}" \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMP} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --gpu_memory_utilization ${GPU_UTIL} \
    --max_model_len ${MAX_MODEL_LEN}
}

# -------- 在这里列出要跑的模型 --------
# 可以按需要增删；路径是你当前机器上的真实路径。

run_one_model "Llama-3.2-1B-Instruct" "/data/wenhesun/model/meta-llama/Llama-3.2-1B-Instruct"

run_one_model "Qwen2.5-1.5B-Instruct" "/data/wenhesun/model/Qwen/Qwen2.5-1.5B-Instruct"
run_one_model "Qwen2.5-7B"            "/data/wenhesun/model/Qwen/Qwen2.5-7B"
run_one_model "Qwen2.5-Math-7B"       "/data/wenhesun/model/Qwen/Qwen2.5-Math-7B"
run_one_model "Qwen3-8B"              "/data/wenhesun/model/Qwen/Qwen3-8B"

echo "[DONE] All models finished probe generation with vLLM."