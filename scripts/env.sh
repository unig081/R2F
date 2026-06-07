#!/usr/bin/env bash
# This file is meant to be sourced. Do not mutate the caller's shell options.

export R2F_ROOT="${R2F_ROOT:-/mnt/data1/zxc/R2F}"
export R2F_CONDA_ENV_NAME="${R2F_CONDA_ENV_NAME:-zxc_r2f}"
export R2F_CONDA_ENV="${R2F_CONDA_ENV:-/mnt/data1/conda_env/${R2F_CONDA_ENV_NAME}}"
export R2F_OUTPUT_DIR="${R2F_OUTPUT_DIR:-${R2F_ROOT}/results}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${R2F_ROOT}/cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${R2F_ROOT}/cache/transformers}"
export HF_HOME="${HF_HOME:-${R2F_ROOT}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${R2F_ROOT}/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${R2F_ROOT}/cache/pip}"
export PYTHONNOUSERSITE=1
export PIP_USER=false
export PYTHONPATH="${R2F_ROOT}/src:${PYTHONPATH:-}"

for p in \
  "${R2F_ROOT}/models/llama_3_2_1B_instruct_tofu" \
  "/mnt/data1/zxc/models/llama_3_2_1B_instruct_tofu" \
  "/mnt/data1/zxc/handoff/junior_llama_tofu_eval_bundle/models/llama_3_2_1B_instruct_tofu" \
  "/mnt/data1/zxc/zxc_r2f/models/llama_3_2_1B_instruct_tofu"; do
  if [ -d "${p}" ] && [ -z "${R2F_SOURCE_MODEL:-}" ]; then
    export R2F_SOURCE_MODEL="${p}"
  fi
done

for p in \
  "${R2F_ROOT}/models/llama_3_2_3B_instruct_tofu" \
  "/mnt/data1/zxc/models/llama_3_2_3B_instruct_tofu" \
  "/mnt/data1/zxc/handoff/junior_llama_tofu_eval_bundle/models/llama_3_2_3B_instruct_tofu" \
  "/mnt/data1/zxc/zxc_r2f/models/llama_3_2_3B_instruct_tofu"; do
  if [ -d "${p}" ] && [ -z "${R2F_TARGET_MODEL:-}" ]; then
    export R2F_TARGET_MODEL="${p}"
  fi
done

for p in \
  "${R2F_ROOT}/datasets/tofu/forget05.json" \
  "/mnt/data1/zxc/datasets/tofu/forget05.json" \
  "/mnt/data1/zxc/handoff/junior_llama_tofu_eval_bundle/datasets/tofu/forget05.json" \
  "/mnt/data1/zxc/zxc_r2f/datasets/tofu/forget05.json"; do
  if [ -f "${p}" ] && [ -z "${R2F_FORGET_FILE:-}" ]; then
    export R2F_FORGET_FILE="${p}"
  fi
done

for p in \
  "${R2F_ROOT}/datasets/tofu/retain95.json" \
  "/mnt/data1/zxc/datasets/tofu/retain95.json" \
  "/mnt/data1/zxc/handoff/junior_llama_tofu_eval_bundle/datasets/tofu/retain95.json" \
  "/mnt/data1/zxc/zxc_r2f/datasets/tofu/retain95.json"; do
  if [ -f "${p}" ] && [ -z "${R2F_RETAIN_FILE:-}" ]; then
    export R2F_RETAIN_FILE="${p}"
  fi
done

mkdir -p \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${R2F_ROOT}/logs" \
  "${R2F_ROOT}/reports" \
  "${R2F_ROOT}/results"
