#!/usr/bin/env python3
"""BLOOM-560M inference profiling with CUDA kernel trace (single GPU or TP on 2 GPUs).

Run with (single GPU):
  python bloom_profile.py \
    --model bigscience/bloom-560m \
    --prompt "Explain distributed training in one paragraph." \
    --max-new-tokens 64

Or with tensor parallel (2 GPUs, PyTorch 2.1+):
  torchrun --nproc_per_node=2 bloom_profile.py \
    --model bigscience/bloom-560m \
    --prompt "Explain distributed training." \
    --max-new-tokens 64

Outputs:
  - trace_rank{rank}.json (Chrome trace, one per rank) or trace.json (single GPU)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

try:
    from torch.distributed._tensor import DeviceMesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    HAS_TENSOR_PARALLEL = True
except Exception:
    HAS_TENSOR_PARALLEL = False

from transformers import AutoModelForCausalLM, AutoTokenizer


def init_distributed() -> Dict[str, int]:
    """Initialize distributed process group if using torchrun."""
    if not dist.is_available() or not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl")
        except Exception:
            # Single GPU mode: no distributed
            return {"rank": 0, "world_size": 1, "local_rank": 0}
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def apply_tensor_parallel_bloom(model: torch.nn.Module, mesh: "DeviceMesh") -> None:
    """Apply TP sharding to BLOOM transformer layers."""
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("Model does not look like BLOOM; adjust TP plan.")

    for layer in model.transformer.h:
        tp_plan = {
            # BLOOM uses self_attention.query_key_value (fused QKV) and dense (output proj)
            "self_attention.query_key_value": ColwiseParallel(),
            "self_attention.dense": RowwiseParallel(),
            # MLP
            "mlp.dense_h_to_4h": ColwiseParallel(),
            "mlp.dense_4h_to_h": RowwiseParallel(),
        }
        parallelize_module(layer, mesh, tp_plan)


def ranked_path(path: str, rank: int, single_gpu: bool) -> str:
    """Generate per-rank output path or single path for single GPU."""
    if single_gpu:
        return path
    if "{rank}" in path:
        return path.format(rank=rank)
    root, ext = os.path.splitext(path)
    return f"{root}_rank{rank}{ext}"


def register_shape_hooks(
    model: torch.nn.Module, shape_log: List[Dict[str, object]], enabled: List[bool]
) -> None:
    """Capture one input/output shape per Linear module."""
    seen: set[int] = set()

    def hook(mod: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not enabled[0]:
            return
        if id(mod) in seen:
            return
        seen.add(id(mod))
        in_shape = tuple(inputs[0].shape) if inputs else None
        out_shape = tuple(output.shape) if hasattr(output, "shape") else None
        shape_log.append(
            {
                "module": mod.__class__.__name__,
                "in_shape": in_shape,
                "out_shape": out_shape,
                "dtype": str(output.dtype) if hasattr(output, "dtype") else None,
            }
        )

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(hook)


def main() -> int:
    parser = argparse.ArgumentParser(description="BLOOM inference with CUDA kernel trace.")
    parser.add_argument("--model", default="bigscience/bloom-560m")
    parser.add_argument("--prompt", default="Explain distributed training in one paragraph.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trace", default="trace.json")
    parser.add_argument("--record-shapes", action="store_true", help="Record op input shapes in profiler.")
    parser.add_argument("--shape-log", default="", help="Write one-time Linear shapes JSONL (per rank).")
    args = parser.parse_args()

    dist_info = init_distributed()
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]
    single_gpu = world_size == 1

    device = torch.device("cuda", local_rank)

    if rank == 0:
        mode = "single GPU" if single_gpu else f"{world_size}-GPU tensor parallel"
        print(f"Running BLOOM in {mode} mode.", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    model.to(device)

    if not single_gpu and HAS_TENSOR_PARALLEL:
        mesh = DeviceMesh("cuda", list(range(world_size)))
        apply_tensor_parallel_bloom(model, mesh)
    elif not single_gpu and not HAS_TENSOR_PARALLEL:
        if rank == 0:
            print("ERROR: torchrun with >1 GPU requires PyTorch 2.1+ for tensor parallel.", file=sys.stderr)
        return 1

    shape_log: List[Dict[str, object]] = []
    shape_logging_enabled = [False]
    if args.shape_log:
        register_shape_hooks(model, shape_log, shape_logging_enabled)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # Warmup (no profiler)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8)
    torch.cuda.synchronize()

    # Profile inference
    shape_logging_enabled[0] = True
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=args.record_shapes,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        torch.cuda.synchronize()
    shape_logging_enabled[0] = False

    trace_path = ranked_path(args.trace, rank, single_gpu)
    prof.export_chrome_trace(trace_path)
    if rank == 0:
        print(f"Wrote {trace_path}")

    if args.shape_log:
        shape_path = ranked_path(args.shape_log, rank, single_gpu)
        with open(shape_path, "w", encoding="utf-8") as f:
            for item in shape_log:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        if rank == 0:
            print(f"Wrote {shape_path}")

    if not single_gpu:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
