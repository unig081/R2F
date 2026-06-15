#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2F_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export R2F_ROOT
source "${SCRIPT_DIR}/env.sh"

CONFIG="${1:-${R2F_ROOT}/configs/r2f_tofu.yaml}"

"${SCRIPT_DIR}/run_tofu_r2f_01_lora.sh" "${CONFIG}"
"${SCRIPT_DIR}/run_tofu_r2f_02_dense_samples.sh" "${CONFIG}"
"${SCRIPT_DIR}/run_tofu_r2f_03_train_decoder.sh" "${CONFIG}"
"${SCRIPT_DIR}/run_tofu_r2f_04_apply_r2f.sh" "${CONFIG}"
"${SCRIPT_DIR}/run_tofu_r2f_05_eval.sh" "${CONFIG}"
