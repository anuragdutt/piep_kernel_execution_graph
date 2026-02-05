#!/usr/bin/env bash
# Run Vicuna TP profiler on 2 GPUs. Use on a machine with 2 GPUs (e.g. CUDA_VISIBLE_DEVICES=0,1).
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 vicuna_tp_profile.py \
  --model lmsys/vicuna-7b-v1.5 \
  --prompt "Explain tensor parallelism in one paragraph." \
  --max-new-tokens 64 \
  --trace trace.json \
  --record-shapes \
  --shape-log linear_shapes.jsonl
