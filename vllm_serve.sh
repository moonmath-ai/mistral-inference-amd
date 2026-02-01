#!/bin/bash

# Defaults
MODEL="mistralai/Mistral-7B-v0.3"
IS_MULTI=false

# Parse flags
while getopts "m:g:" flag; do
    case "${flag}" in
        m) MODEL=${OPTARG} ;;
        g) IS_MULTI=${OPTARG} ;;
        h) echo "Usage: $0 -m <model> -g <multi|single> (default single)" ; exit 0 ;;
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

echo "Running: $MODEL"
echo "---------------------------------------"

CUDA_VISIBLE_DEVICES=$GPU_DEVICES vllm serve mistralai/"$MODEL" \
    --tensor-parallel-size $TP_SIZE \
    $EP_FLAG \
    --dtype bfloat16