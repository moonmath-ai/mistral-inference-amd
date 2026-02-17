#!/bin/bash

# Defaults
MODEL="mistralai/Mistral-7B-v0.3"
IS_MULTI=false
DO_PROFILE=false

# --- Optimization Environment Variables ---
# Forces the use of AITER kernels for MoE (The 2.5x speedup source)
export VLLM_ROCM_USE_AITER=1
# Enables optimized Multi-Head Attention for ROCm
export VLLM_ROCM_USE_AITER_MHA=1
# Critical for V1 engine to use fused kernels on MI300X
export VLLM_USE_V1_ENGINE=0
# Disables NUMA balancing if possible (helps with latency)
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA

GPU_DEVICES="0"

# Parse flags
while getopts "m:g:p" flag; do
    case "${flag}" in
        m) MODEL=${OPTARG} ;;
        g) GPU_DEVICES=${OPTARG} ;; # Now correctly captures "0" or "0,1"
        p) DO_PROFILE=true ;;
        *) echo "Usage: $0 -m <model> -g <device_ids> [-p]" ; exit 1 ;;
    esac
done

# Logic: Determine TP_SIZE based on number of commas in GPU_DEVICES
# If GPU_DEVICES is "0,1", count is 2. If "0", count is 1.
NUM_GPUS=$(echo $GPU_DEVICES | tr -cd ',' | wc -c)
TP_SIZE=$((NUM_GPUS + 1))

if [ "$TP_SIZE" -gt 1 ]; then
    echo "Configuration: Multi-GPU ($TP_SIZE Cards) on devices $GPU_DEVICES"
    if [[ "$MODEL" == *"8x"* || "$MODEL" == *"Mixtral"* ]]; then
        EP_FLAG="--enable-expert-parallel"
    fi
else
    echo "Configuration: Single-GPU on device $GPU_DEVICES"
    EP_FLAG=""
fi

# Profiler Logic
PROFILER_FLAG=""
if [ "$DO_PROFILE" = true ]; then
    VLLM_TORCH_PROFILER_DIR="$(pwd)/output/profiles/vllm_traces_$(date +%Y%m%d_%H%M)"
    echo "Profiling enabled, outputs will be in $VLLM_TORCH_PROFILER_DIR"
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"

    PROFILER_ARGS=(
        "--profiler-config"
        "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$VLLM_TORCH_PROFILER_DIR\"}"
    )
fi

echo "Running: $MODEL"
echo "---------------------------------------"

export VLLM_RPC_TIMEOUT=1800000
CUDA_VISIBLE_DEVICES=$GPU_DEVICES vllm serve mistralai/"$MODEL" \
    --tensor-parallel-size $TP_SIZE \
    $EP_FLAG \
    --dtype bfloat16 \
    "${PROFILER_ARGS[@]}"