#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# vLLM V1 Disaggregated Serving Entrypoint
#
# Roles:
#   standalone   -> normal vLLM
#   kv_producer  -> Prefill worker
#   kv_consumer  -> Decode worker
# ============================================================

: "${MODEL_NAME:=Qwen/Qwen2.5-7B-Instruct-AWQ}"
: "${GPU_MEMORY_UTILIZATION:=0.80}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=32}"
: "${KV_CACHE_DTYPE:=fp8}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${VLLM_KV_ROLE:=standalone}"

echo "================================================"
echo " Starting vLLM V1"
echo "================================================"
echo " Model:              ${MODEL_NAME}"
echo " KV Role:            ${VLLM_KV_ROLE}"
echo " GPU Memory:         ${GPU_MEMORY_UTILIZATION}"
echo " Max Model Length:   ${MAX_MODEL_LEN}"
echo " Max Sequences:      ${MAX_NUM_SEQS}"
echo " KV Cache Dtype:     ${KV_CACHE_DTYPE}"
echo " Address:            ${HOST}:${PORT}"
echo "================================================"

BASE_ARGS=(
    "${MODEL_NAME}"

    "--quantization" "awq"

    "--kv-cache-dtype" "${KV_CACHE_DTYPE}"

    "--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}"

    "--max-model-len" "${MAX_MODEL_LEN}"

    "--max-num-seqs" "${MAX_NUM_SEQS}"

    "--enable-prefix-caching"

    "--host" "${HOST}"

    "--port" "${PORT}"
)

# ------------------------------------------------------------
# KV Transfer Configuration
# ------------------------------------------------------------

if [[ "${VLLM_KV_ROLE}" == "kv_producer" ]]; then

    echo "Role: PREFILL / KV PRODUCER"

    KV_TRANSFER_CONFIG='{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer"
    }'

    BASE_ARGS+=(
        "--kv-transfer-config"
        "${KV_TRANSFER_CONFIG}"
    )

elif [[ "${VLLM_KV_ROLE}" == "kv_consumer" ]]; then

    echo "Role: DECODE / KV CONSUMER"

    KV_TRANSFER_CONFIG='{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }'

    BASE_ARGS+=(
        "--kv-transfer-config"
        "${KV_TRANSFER_CONFIG}"
    )

else

    echo "Role: STANDALONE"

fi

echo ""
echo "Starting:"
echo "vllm serve ${MODEL_NAME} ..."
echo ""

exec vllm serve "${BASE_ARGS[@]}"