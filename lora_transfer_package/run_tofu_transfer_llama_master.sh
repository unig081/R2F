#!/usr/bin/env bash
# ===========================================================================
# TOFU LoRA Transfer - Master Pipeline
# ===========================================================================
# 一键运行 LLaMA 3.2 1B→3B TOFU LoRA 迁移实验 (含匈牙利头匹配对比)
#
# 超参数空间: forget_split × {actmap_base, actmap_hungarian}
#   共 3×2 = 6 个实验
#
# GPU分配: cuda:0 和 cuda:1 各跑3个transfer任务
#
# 用法: bash scripts/run_tofu_transfer_llama_master.sh
# ===========================================================================
set -euo pipefail

ROOT="/mnt/data1/syn/Unlearning_lora"
PYTHON="/mnt/data1/conda_env/syn_env/bin/python"
METHOD_DIR="${ROOT}/lora_transfer_package/method1_ng_v2_actmap"
TMP_DIR="${ROOT}/tmp"
LOG_DIR="${ROOT}/logs/lora_transfer"
mkdir -p "${LOG_DIR}" "${TMP_DIR}"

SRC_MODEL="${ROOT}/models/llama_3_2_1B_instruct_tofu"
TGT_MODEL="${ROOT}/models/llama_3_2_3B_instruct_tofu"
R_L_DIR="${TMP_DIR}/act_align_llama_1B_3B"
GENERIC_TEXTS="${TMP_DIR}/generic_texts_llama.json"
OUTPUT_BASE="${ROOT}/lora_transfer_output/llama_1B_to_3B"

echo "============================================"
echo "TOFU LoRA Transfer Pipeline"
echo "  Source: llama_3_2_1B (16L, h=2048, inter=8192)"
echo "  Target: llama_3_2_3B (28L, h=3072, inter=8192)"
echo "  GPUs: cuda:0, cuda:1"
echo "============================================"

# ===========================================================================
# Phase 1: Prepare generic texts for R_l computation
# ===========================================================================
if [ ! -f "${GENERIC_TEXTS}" ]; then
    echo "[Phase 1a] Preparing generic texts..."
    ${PYTHON} "${METHOD_DIR}/prepare_generic_texts.py" \
        --input "${ROOT}/datasets/tofu/retain95.json" \
        --n 100 \
        --output "${GENERIC_TEXTS}"
    echo "  Done: ${GENERIC_TEXTS}"
else
    echo "[Phase 1a] Generic texts exist: ${GENERIC_TEXTS}"
fi

# ===========================================================================
# Phase 1b: Collect activations → R_l matrices (needs GPU)
# ===========================================================================
if [ ! -f "${R_L_DIR}/R_l_0.pt" ]; then
    echo "[Phase 1b] Collecting activations for R_l..."
    CUDA_VISIBLE_DEVICES=0 ${PYTHON} "${METHOD_DIR}/collect_activations.py" \
        --old_model "${SRC_MODEL}" \
        --new_model "${TGT_MODEL}" \
        --data "${GENERIC_TEXTS}" \
        --n_samples 64 \
        --max_length 128 \
        --batch_size 2 \
        --reg 1e-3 \
        --layer_mapping_mode fixed \
        --output_dir "${R_L_DIR}" \
        2>&1 | tee "${LOG_DIR}/collect_activations_llama.log"
    echo "  Done: ${R_L_DIR}"
else
    echo "[Phase 1b] R_l matrices exist: ${R_L_DIR}"
fi

echo ""
echo "============================================"
echo "Phase 2: Launching 6 transfer experiments"
echo "============================================"

# ── Helper function ──
run_transfer() {
    local forget_split="$1"
    local use_hungarian="$2"
    local gpu_idx="$3"

    local method_name
    if [ "${use_hungarian}" = "true" ]; then
        method_name="actmap_hungarian"
    else
        method_name="actmap_base"
    fi

    local src_lora="${ROOT}/checkpoints/tofu_gagd/llama_3_2_1B_tofu_${forget_split}/adapter"
    local output_dir="${OUTPUT_BASE}/${forget_split}/${method_name}"
    local log_file="${LOG_DIR}/transfer_llama_${forget_split}_${method_name}.log"

    if [ -f "${output_dir}/adapter_model.safetensors" ]; then
        echo "[$(date '+%F %T')] SKIP ${forget_split}/${method_name} (exists)" | tee -a "${log_file}"
        return
    fi

    echo "[$(date '+%F %T')] START ${forget_split}/${method_name} on GPU ${gpu_idx}" | tee "${log_file}"

    local hungarian_flag=""
    if [ "${use_hungarian}" = "true" ]; then
        hungarian_flag="--hungarian_heads"
    fi

    CUDA_VISIBLE_DEVICES="${gpu_idx}" ${PYTHON} \
        "${METHOD_DIR}/transfer_lora_actmap_generic.py" \
        --src_model "${SRC_MODEL}" \
        --tgt_model "${TGT_MODEL}" \
        --src_lora "${src_lora}" \
        --output_dir "${output_dir}" \
        --r_l_dir "${R_L_DIR}" \
        --dtype bfloat16 \
        ${hungarian_flag} \
        >> "${log_file}" 2>&1

    echo "[$(date '+%F %T')] DONE  ${forget_split}/${method_name}" | tee -a "${log_file}"
}

# Launch all 6 jobs in parallel
# GPU 0: forget01_base, forget05_base, forget10_base
run_transfer "forget01" "false" "0" &
PID1=$!
sleep 5

run_transfer "forget05" "false" "0" &
PID2=$!
sleep 5

run_transfer "forget10" "false" "0" &
PID3=$!

# GPU 1: forget01_hungarian, forget05_hungarian, forget10_hungarian
sleep 3
run_transfer "forget01" "true" "1" &
PID4=$!
sleep 5

run_transfer "forget05" "true" "1" &
PID5=$!
sleep 5

run_transfer "forget10" "true" "1" &
PID6=$!

echo ""
echo "All 6 transfer jobs launched:"
echo "  GPU 0: PID ${PID1} (forget01_base), PID ${PID2} (forget05_base), PID ${PID3} (forget10_base)"
echo "  GPU 1: PID ${PID4} (forget01_hung), PID ${PID5} (forget05_hung), PID ${PID6} (forget10_hung)"
echo "  Logs: ${LOG_DIR}/"
echo ""
echo "Waiting for all jobs to finish..."

wait ${PID1} ${PID2} ${PID3} ${PID4} ${PID5} ${PID6}

echo ""
echo "============================================"
echo "Phase 2 complete! Results:"
echo "============================================"
for split in forget01 forget05 forget10; do
    for method in actmap_base actmap_hungarian; do
        d="${OUTPUT_BASE}/${split}/${method}"
        if [ -f "${d}/adapter_model.safetensors" ]; then
            size=$(du -sh "${d}/adapter_model.safetensors" | cut -f1)
            echo "  ✓ ${split}/${method}  (${size})"
        else
            echo "  ✗ ${split}/${method}  (MISSING - check logs)"
        fi
    done
done

echo ""
echo "Next: evaluation"
echo "  python lora_transfer_package/common/evaluate_lora_with_judge.py ..."
