#!/usr/bin/env bash
# Run BLOOM-560M profiler (single GPU or TP on 2 GPUs).
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate

# Single GPU (works on 1080 Ti):
# python bloom_profile.py \
#   --model bigscience/bloom-560m \
#   --prompt "Explain distributed training." \
#   --max-new-tokens 64 \
#   --trace trace.json \
#   --record-shapes \
#   --shape-log bloom_shapes.jsonl

# Or 2-GPU tensor parallel (PyTorch 2.1+):
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 bloom_profile.py \
  --model bigscience/bloom-560m \
  --prompt "Explain distributed training." \
  --max-new-tokens 64 \
  --trace trace.json \
  --record-shapes \
  --shape-log bloom_shapes.jsonl
