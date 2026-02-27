#!/usr/bin/env python3
"""Tensor-parallel Vicuna-7B inference on 2 GPUs - unified profiling and benchmarking.

This is the SINGLE SOURCE OF TRUTH for Vicuna TP=2 inference.
Both profiling (kernel trace extraction) and benchmarking (latency/energy measurement)
use the EXACT same code path to ensure identical workloads.

Usage:
  # Profile only (extract kernel traces)
  torchrun --nproc_per_node=2 vicuna_tp_profile.py \
    --mode profile \
    --prompt "Tell me a joke" \
    --max-new-tokens 64

  # Benchmark only (measure latency)
  torchrun --nproc_per_node=2 vicuna_tp_profile.py \
    --mode benchmark \
    --warmup 5 --runs 100

  # Both profiling and benchmarking
  torchrun --nproc_per_node=2 vicuna_tp_profile.py \
    --mode both \
    --warmup 5 --runs 100

Requires PyTorch 2.1+ for torch.distributed.tensor.parallel (DeviceMesh, etc.).

Outputs:
  - trace_rank{rank}.json (Chrome trace, one per rank) [if mode=profile or both]
  - benchmark_results.json (timing stats) [if mode=benchmark or both]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
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
        "ERROR: torch.distributed.tensor.parallel is required. Use PyTorch 2.1+.",
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
    """Capture EVERY invocation of all Linear layers.

    CRITICAL: For TP-sharded tensors (DTensor), we extract LOCAL shard shapes,
    not global shapes, because the actual CUDA kernels operate on local shards.

    FIXED: Removed 'seen' set to log ALL invocations (prefill + decode steps),
    not just first occurrence of each module.
    """

    def get_local_shape(tensor):
        """Extract local shard shape from DTensor, or regular shape from Tensor."""
        if tensor is None:
            return None
        # Check if this is a DTensor (TP-sharded)
        if hasattr(tensor, "_local_tensor"):
            # DTensor: return the LOCAL shard shape (what kernels actually see)
            return tuple(tensor._local_tensor.shape)
        elif hasattr(tensor, "shape"):
            # Regular tensor: return shape directly
            return tuple(tensor.shape)
        else:
            return None

    def hook(
        mod: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        if not enabled[0]:
            return

        in_shape = get_local_shape(inputs[0]) if inputs else None
        out_shape = get_local_shape(output)

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


def run_generation(model, inputs, max_new_tokens: int):
    """
    Single source of truth for generation.
    Both profiling and benchmarking call this EXACT function.

    Uses greedy decoding (do_sample=False) for deterministic, reproducible results.
    """
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for reproducibility
            use_cache=True,
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vicuna-7B TP=2 inference - unified profiling and benchmarking."
    )
    parser.add_argument("--model", default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--mode",
        choices=["profile", "benchmark", "both"],
        default="profile",
        help="Mode: profile (trace kernels), benchmark (measure latency), or both",
    )
    parser.add_argument(
        "--prompt",
        default="Explain tensor parallelism in one paragraph.",
        help="Input prompt (default: simple prompt for consistency)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of timed benchmark runs (default: 100, only for benchmark mode)",
    )
    parser.add_argument("--trace", default="trace.json", help="Output trace file path")
    parser.add_argument(
        "--shape-log",
        default="shape_log.jsonl",
        help="Write Linear layer shapes JSONL (default: shape_log.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output JSON file for benchmark results",
    )
    args = parser.parse_args()

    # CRITICAL: Load tokenizer BEFORE distributed init to avoid corruption
    # Known issue: HuggingFace tokenizers can get corrupted after torch.distributed.init_process_group()
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    print(f"✓ Tokenizer loaded successfully")

    dist_info = init_distributed()
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]

    if world_size != 2 and rank == 0:
        print(f"WARNING: expected 2 ranks, got {world_size}.", file=sys.stderr)

    device = torch.device("cuda", local_rank)

    if rank == 0:
        print("=" * 70)
        print("Vicuna-7B Tensor Parallel Inference")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Mode: {args.mode}")
        print(f"TP size: {world_size}")
        print(f"Prompt: '{args.prompt}'")  # DEBUG: Show actual prompt
        print(f"Max new tokens: {args.max_new_tokens}")
        print(f"Warmup runs: {args.warmup}")
        if args.mode in ["benchmark", "both"]:
            print(f"Timed runs: {args.runs}")
        print(f"Precision: FP16")
        print("=" * 70)

    # Load model with tensor parallelism
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()
    model.to(device)

    mesh = DeviceMesh("cuda", list(range(world_size)))
    apply_tensor_parallel(model, mesh)

    if rank == 0:
        print("✓ Model loaded with TP sharding")

    # Setup shape logging hooks (only for profiling mode)
    shape_log: List[Dict[str, object]] = []
    shape_logging_enabled = [False]
    if args.mode in ["profile", "both"]:
        register_shape_hooks(model, shape_log, shape_logging_enabled)

    # Tokenize input (SAME for both modes)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_tokens = inputs["input_ids"].shape[1]

    if rank == 0:
        print(f"\nTokenization:")
        print(f"  Prompt: '{args.prompt}'")
        print(f"  Input tokens: {input_tokens}")
        print(f"  Token IDs: {inputs['input_ids'][0].tolist()}")

    # Synchronize before starting
    dist.barrier()

    # ========================================================================
    # WARMUP (always done, regardless of mode)
    # ========================================================================
    if rank == 0:
        print(f"\nWarmup ({args.warmup} runs)...")

    for i in range(args.warmup):
        outputs = run_generation(model, inputs, args.max_new_tokens)
        torch.cuda.synchronize()
        if rank == 0 and (i + 1) % max(1, args.warmup // 2) == 0:
            print(f"  Warmup {i + 1}/{args.warmup}")

    dist.barrier()

    # ========================================================================
    # PROFILING MODE - Extract kernel traces
    # ========================================================================
    if args.mode in ["profile", "both"]:
        if rank == 0:
            print("\n[PROFILE MODE] Extracting kernel traces...")

        shape_logging_enabled[0] = True
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,  # Avoid distributed profiler bug
        ) as prof:
            outputs = run_generation(model, inputs, args.max_new_tokens)
            torch.cuda.synchronize()
        shape_logging_enabled[0] = False

        # Save trace
        trace_path = ranked_path(args.trace, rank)
        prof.export_chrome_trace(trace_path)
        if rank == 0:
            print(f"✓ Trace saved: {trace_path}")

        # Save shape log
        shape_path = ranked_path(args.shape_log, rank)
        with open(shape_path, "w", encoding="utf-8") as f:
            for item in shape_log:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        if rank == 0:
            print(f"✓ Shapes saved: {shape_path}")

        dist.barrier()

    # ========================================================================
    # BENCHMARK MODE - Measure latency and energy
    # ========================================================================
    if args.mode in ["benchmark", "both"]:
        if rank == 0:
            print(f"\n[BENCHMARK MODE] Running {args.runs} timed iterations...")

        latencies_ms = []
        start_timestamp = datetime.now()

        for i in range(args.runs):
            torch.cuda.synchronize()
            start_time = time.time()

            outputs = run_generation(model, inputs, args.max_new_tokens)

            torch.cuda.synchronize()
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            latencies_ms.append(latency_ms)

            if rank == 0 and (i + 1) % max(1, args.runs // 10) == 0:
                print(f"  Run {i + 1}/{args.runs}: {latency_ms:.2f} ms")

        end_timestamp = datetime.now()
        total_duration_s = (end_timestamp - start_timestamp).total_seconds()

        # Calculate statistics (rank 0 only)
        if rank == 0:
            mean_ms = sum(latencies_ms) / len(latencies_ms)
            min_ms = min(latencies_ms)
            max_ms = max(latencies_ms)
            std_ms = (
                sum((x - mean_ms) ** 2 for x in latencies_ms) / len(latencies_ms)
            ) ** 0.5

            print("\n" + "=" * 70)
            print("BENCHMARK RESULTS")
            print("=" * 70)
            print(f"Total runs: {args.warmup} warmup + {args.runs} timed")
            print(f"Mean latency: {mean_ms:.2f} ms")
            print(f"Min latency: {min_ms:.2f} ms")
            print(f"Max latency: {max_ms:.2f} ms")
            print(f"Std deviation: {std_ms:.2f} ms")
            print(f"Output tokens: {outputs.shape[1]}")
            print(f"Total duration: {total_duration_s:.2f} s")
            print("=" * 70)

            # Save results
            output_data = {
                "model": args.model,
                "tensor_parallel_size": world_size,
                "prompt": args.prompt,
                "input_tokens": input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "output_tokens": outputs.shape[1],
                "warmup_runs": args.warmup,
                "timed_runs": args.runs,
                "start_timestamp": start_timestamp.isoformat(),
                "end_timestamp": end_timestamp.isoformat(),
                "total_duration_s": total_duration_s,
                "stats": {
                    "mean_ms": mean_ms,
                    "min_ms": min_ms,
                    "max_ms": max_ms,
                    "std_ms": std_ms,
                },
                "latencies_ms": latencies_ms,
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\n✓ Results saved to: {args.output}")

        dist.barrier()

    # Cleanup
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
