#!/usr/bin/env python3
"""Tensor-parallel Vicuna-7B inference on 2 GPUs with CUDA kernel trace.

Run with:
  torchrun --nproc_per_node=2 vicuna_tp_profile.py \
    --model lmsys/vicuna-7b-v1.5 \
    --prompt "Tell me a joke" \
    --max-new-tokens 64

Requires PyTorch 2.1+ for torch.distributed.tensor.parallel (DeviceMesh, etc.).

Outputs (each rank):
  - trace_rank{rank}.json (Chrome trace, one per rank)
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
except Exception as exc:
    print(
        "ERROR: torch.distributed.tensor.parallel is required. "
        "Use PyTorch 2.1+.",
        file=sys.stderr,
    )
    raise exc

from transformers import AutoModelForCausalLM, AutoTokenizer


def init_distributed() -> Dict[str, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def apply_tensor_parallel(model: torch.nn.Module, mesh: DeviceMesh) -> None:
    """Apply TP sharding to LLaMA/Vicuna-style layers."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model does not look like LLaMA/Vicuna; adjust TP plan.")

    for layer in model.model.layers:
        tp_plan = {
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }
        parallelize_module(layer, mesh, tp_plan)

    # Optional: shard LM head if desired. Keeping it replicated is simpler/safer.
    # parallelize_module(model.lm_head, mesh, {"": ColwiseParallel()})


def ranked_path(path: str, rank: int) -> str:
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
    parser = argparse.ArgumentParser(description="Vicuna-7B TP inference with CUDA kernel trace.")
    parser.add_argument("--model", default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--prompt", default="Explain tensor parallelism in one paragraph.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trace", default="trace.json")
    parser.add_argument("--record-shapes", action="store_true", help="Record op input shapes in profiler.")
    parser.add_argument("--shape-log", default="", help="Write one-time Linear shapes JSONL (per rank).")
    args = parser.parse_args()

    dist_info = init_distributed()
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]

    if world_size != 2 and rank == 0:
        print(f"WARNING: expected 2 ranks, got {world_size}.", file=sys.stderr)

    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()
    model.to(device)

    mesh = DeviceMesh("cuda", list(range(world_size)))
    apply_tensor_parallel(model, mesh)

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

    trace_path = ranked_path(args.trace, rank)
    prof.export_chrome_trace(trace_path)
    if rank == 0:
        print(f"Wrote {trace_path}")

    if args.shape_log:
        shape_path = ranked_path(args.shape_log, rank)
        with open(shape_path, "w", encoding="utf-8") as f:
            for item in shape_log:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        if rank == 0:
            print(f"Wrote {shape_path}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
