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
export VLLM_USE_V1_ENGINE=1 
# Disables NUMA balancing if possible (helps with latency)
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA

# Parse flags
while getopts "m:g:p" flag; do
    case "${flag}" in
        m) MODEL=${OPTARG} ;;
        g) IS_MULTI=${OPTARG} ;;
        p) DO_PROFILE=true ;;
        h) echo "Usage: $0 -m <model> -g <multi|single> (default single) [-p]" ; exit 0 ;;
    esac
done

# Convert input to lowercase to handle "Multi", "TRUE", etc.
IS_MULTI=$(echo "$IS_MULTI" | tr '[:upper:]' '[:lower:]')

# Flexible check for multi-gpu triggers
if [[ "$IS_MULTI" == "true" || "$IS_MULTI" == "multi" || "$IS_MULTI" == "yes" ]]; then
    echo "Configuration: Multi-GPU (2 Cards)"
    TP_SIZE=2
    # Only use expert parallel if it's an MoE model (contains '8x' or 'Mixtral')
    if [[ "$MODEL" == *"8x"* || "$MODEL" == *"Mixtral"* ]]; then
        EP_FLAG="--enable-expert-parallel"
    else
        EP_FLAG=""
    fi
    GPU_DEVICES="0,1"
else
    echo "Configuration: Single-GPU"
    TP_SIZE=1
    EP_FLAG=""
    GPU_DEVICES="0"
fi

# Profiler Logic
PROF_CMD=""
if [ "$DO_PROFILE" = true ]; then
    PROF_DIR="./output/profiles/vllm_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$PROF_DIR"
    echo "Profiling enabled. Results will be in: $PROF_DIR"
    # --stats gives you the kernel summary (Average duration, calls, etc)
    # --hip-trace gives you the timeline for Perfetto
    # --roctx-trace shows the vLLM internal labels
    PROF_CMD="rocprofv3 -d $PROF_DIR -r -f pftrace csv --stats -T -P 150:300:1 -- "
fi

echo "Running: $MODEL"
echo "---------------------------------------"

CUDA_VISIBLE_DEVICES=$GPU_DEVICES $PROF_CMD vllm serve mistralai/"$MODEL" \
    --tensor-parallel-size $TP_SIZE \
    $EP_FLAG \
    --dtype bfloat16