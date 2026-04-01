#!/usr/bin/env python3
"""Classify kernels into 4 tiers based on replay method.

Tier 1: Memory ops     (cudaMemcpy, cudaMemset, memmove)
Tier 2: Matmul         (cuBLAS GEMM/GEMV — dominates compute energy)
Tier 3: Torch ops      (flash attention, RMS/layer norm, rotary, softmax,
                        elementwise, scan/reduce, unrecognised GPU kernels)
Tier 4: Comms          (NCCL AllReduce/AllGather/ReduceScatter + CUDA stream
                        and event sync primitives that represent
                        communication overhead in tensor-parallel workloads:
                        cudaStreamSynchronize, cudaStreamWaitEvent,
                        cudaDeviceSynchronize, cudaEventRecord, cudaEventQuery)

Reads: unique_kernels_compute.jsonl + shape_mapping.json
Writes: kernel_signatures.json

Usage:
    python classify_kernels.py --input ../../vicuna/unique_kernels_compute.jsonl --shapes ../../vicuna/shape_mapping.json
"""

import argparse
import json
import re
from typing import Dict, List, Any, Optional
from collections import Counter


def load_shape_mapping(shapes_file: str) -> Dict[str, Any]:
    """
    Load shape_mapping.json which contains GEMM configurations.

    Returns dict with:
    - gemm_configs: list of {name, M, N, K, count, in_shape, out_shape}
    - model metadata: tensor_parallel_size, hidden_size, etc.
    """
    try:
        with open(shapes_file, "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Warning: Could not find {shapes_file}, using grid-based estimates")
        return {"gemm_configs": []}


def classify_kernel(kernel: Dict[str, Any]) -> int:
    """
    Classify a kernel into one of 4 tiers.

    Returns:
        1: Memory ops  (cudaMemcpy, cudaMemset, memmove)
        2: Matmul      (cuBLAS GEMM/GEMV)
        3: Torch ops   (flash attention, norms, rotary, elementwise, etc.)
        4: Comms       (NCCL AllReduce/AllGather/ReduceScatter + CUDA stream/event
                        sync primitives that represent TP communication overhead)
    """
    name = kernel["name"].lower()

    # Tier 1: Memory ops
    if "memcpy" in name or "memset" in name or "memmove" in name:
        return 1

    # Tier 4: NCCL / communication (must be before Tier 2 so "nccl" doesn't match elsewhere)
    if (
        "nccl" in name
        or "allreduce" in name
        or "allgather" in name
        or "reducescatter" in name
    ):
        return 4

    # Tier 4: CUDA stream/event sync primitives — these represent communication
    # overhead in TP workloads (compute stream blocking on AllReduce completion).
    _COMMS_SYNC_APIS = {
        "cudastreamsynchronize",
        "cudastreamwaitevent",
        "cudadevicesynchronize",
        "cudaeventrecord",
        "cudaeventquery",
    }
    if name in _COMMS_SYNC_APIS:
        return 4

    # Tier 2: cuBLAS kernels - Updated with Ampere patterns
    if any(
        pattern in name
        for pattern in [
            "gemm",
            "gemv",
            "splitkreduce",
            "sgemm",
            "dgemm",
            "hgemm",
            "maxwell_sgemm",  # Pascal (1080Ti)
            "ampere_fp16_s16816gemm",  # Ampere (A6000) - NEW
            "sm80_xmma",  # Ampere Tensor Core ops - NEW
            "volta_",
            "turing_",
        ]
    ):
        return 2

    # Tier 3: Everything else (PyTorch native, CUB, cudaLaunchKernel, etc.)
    return 3


def extract_memcpy_params(kernel: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for Tier 1 (memcpy/memset) kernels."""
    name = kernel["name"]
    args = kernel.get("args", {})
    sig = kernel.get("signature", {})

    params = {"bytes": args.get("bytes", sig.get("bytes", 0)), "kind": "unknown"}

    # Determine memcpy kind
    if "HtoD" in name or ("Host" in name and "Device" in name):
        params["kind"] = "HtoD"
    elif "DtoH" in name or ("Device" in name and "Host" in name):
        params["kind"] = "DtoH"
    elif "DtoD" in name or "Device -> Device" in name:
        params["kind"] = "DtoD"
    elif "memset" in name.lower():
        params["kind"] = "memset"

    return params


def match_gemm_to_shape(
    kernel_count: int, gemm_configs: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Match a GEMM kernel to its shape config based on invocation count.

    Strategy:
    - qkv_proj: 96 invocations (32 layers × 3 projections)
    - ffn_gate_up: 64 invocations (32 layers × 2 projections)
    - o_proj: 32 invocations (32 layers × 1)
    - ffn_down: 32 invocations (32 layers × 1)
    - lm_head: 1 invocation
    """
    for config in gemm_configs:
        if config["count"] == kernel_count:
            return config

    # If exact match fails, return None (will use grid-based estimate)
    return None


def extract_gemm_params(
    kernel: Dict[str, Any], shape_mapping: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract parameters for Tier 2 (cuBLAS GEMM/GEMV) kernels."""
    sig = kernel.get("signature", {})
    args = kernel.get("args", {})
    grid = sig.get("grid", [1, 1, 1])
    block = sig.get("block", [1, 1, 1])
    name = kernel["name"]

    params = {
        "grid": grid,
        "block": block,
        "shared_memory": sig.get("shared memory", 0),
        "dtype": "fp16",  # Vicuna uses FP16
        "operation": "unknown",
    }

    # Detect GPU architecture from kernel name
    if "ampere" in name.lower() or "sm80" in name.lower():
        params["arch"] = "ampere"
    elif "maxwell" in name.lower():
        params["arch"] = "pascal"
    elif "volta" in name.lower():
        params["arch"] = "volta"
    elif "turing" in name.lower():
        params["arch"] = "turing"
    else:
        params["arch"] = "unknown"

    # Try to extract tile sizes from kernel name
    # e.g., ampere_fp16_s16816gemm_fp16_64x64 -> tile_m=64, tile_n=64
    tile_match = re.search(r"(\d+)x(\d+)", name)
    if tile_match:
        params["tile_m"] = int(tile_match.group(1))
        params["tile_n"] = int(tile_match.group(2))

    # Match kernel to shape based on invocation count
    gemm_configs = shape_mapping.get("gemm_configs", [])
    if gemm_configs:
        matched_shape = match_gemm_to_shape(kernel["count"], gemm_configs)
        if matched_shape:
            params["M"] = matched_shape["M"]
            params["N"] = matched_shape["N"]
            params["K"] = matched_shape["K"]
            params["operation"] = matched_shape["name"]
            params["in_shape"] = matched_shape["in_shape"]
            params["out_shape"] = matched_shape["out_shape"]
        else:
            # No exact match - could be a partial execution or splitK kernel
            # Use grid-based estimate and set operation from kernel name
            if "gemv" in name.lower() or "gemmk1" in name.lower():
                params["operation"] = "gemv"
            else:
                params["operation"] = "gemm"
            if "tile_m" in params and "tile_n" in params:
                params["M"] = grid[0] * params["tile_m"]
                params["N"] = grid[1] * params["tile_n"]
                params["K"] = shape_mapping.get("hidden_size", 4096)
            elif "gemv" in name.lower() or "gemmk1" in name.lower():
                params["M"] = grid[0] * block[0] if block else 256
                params["N"] = 1
                params["K"] = shape_mapping.get("hidden_size", 4096)
    else:
        # Fallback to grid-based estimate
        if "gemv" in name.lower() or "gemmk1" in name.lower():
            params["operation"] = "gemv"
            params["M"] = grid[0] * block[0] if block else 256
            params["N"] = 1
            params["K"] = 4096
        else:
            params["operation"] = "gemm"
            if "tile_m" in params and "tile_n" in params:
                params["M"] = grid[0] * params["tile_m"]
                params["N"] = grid[1] * params["tile_n"]
                params["K"] = 4096  # Default for Vicuna-7B

    return params


def extract_nccl_params(kernel: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for Tier 4 (NCCL/communication) kernels."""
    args = kernel.get("args", {})
    sig = kernel.get("signature", {})
    # From kernel_launch_params: In msg nelems, Out msg nelems, dtype, Group size, Process Group Ranks
    in_nelems = args.get("In msg nelems")
    if in_nelems is None and "args" in kernel:
        in_nelems = kernel["args"].get("In msg nelems")
    # Fallback from signature if present
    if in_nelems is None and sig:
        in_nelems = sig.get("In msg nelems", 94208)
    return {
        "nelems": in_nelems if in_nelems is not None else 94208,
        "dtype": args.get("dtype", "Half"),
        "group_size": args.get("Group size", 2),
        "collective": args.get("Collective name", "allreduce"),
        "grid": sig.get("grid", [4, 1, 1]),
        "block": sig.get("block", [512, 1, 1]),
        "shared_memory": sig.get("shared memory", 88416),
    }


def extract_libtorch_params(kernel: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for Tier 3 (libtorch) kernels."""
    name = kernel["name"]
    sig = kernel.get("signature", {})

    params = {
        "grid": sig.get("grid", [1, 1, 1]),
        "block": sig.get("block", [1, 1, 1]),
        "shared_memory": sig.get("shared memory", 0),
        "operation": "unknown",
    }

    # Identify the operation type from kernel name (order: more specific first)
    n = name.lower()
    # CUDA runtime API / driver calls - not GPU kernels we can replay; use minimal proxy
    if n.startswith("cuda") or n.startswith("culaunch"):
        params["operation"] = "cuda_api"
    elif "pytorch_flash" in n or "flash_fwd_kernel" in n:
        params["operation"] = "flash_attention"
    elif "layer_norm" in n or "layernorm" in n:
        params["operation"] = "layer_norm"
    elif "rms_norm" in n or "rmsnorm" in n:
        params["operation"] = "rms_norm"
    elif "rope" in n or "rotary" in n:
        params["operation"] = "rotary_embedding"
    elif "softmax" in n:
        params["operation"] = "softmax"
    elif "indexselect" in n or "index_select" in n:
        params["operation"] = "index_select"
    elif "gelu" in n:
        params["operation"] = "gelu"
    elif "silu" in n:
        params["operation"] = "silu"
    elif (
        "reduce_kernel" in n
        or "reduction_prod_kernel" in n
        or "argmax" in n
        or "maxnan" in n
    ):
        params["operation"] = "reduce"
    elif "devicescan" in n or "scan" in n:
        params["operation"] = "scan"
    elif (
        "cudafunctor_add" in n
        or "cudafunctoronself_add" in n
        or "cudafunctoronother_add" in n
        or ("add" in n and "cudafunctor" in n)
    ):
        params["operation"] = "add"
    elif (
        "mulfunctor" in n
        or ("binaryfunctor" in n and "mul" in n)
        or ("bunaryfunctor" in n and "mul" in n)
    ):
        params["operation"] = "mul"
    elif "fillfunctor" in n:
        params["operation"] = "fill"
    elif "direct_copy" in n or "direct_copy_kernel" in n:
        params["operation"] = "elementwise"
    elif "masked_fill" in n:
        params["operation"] = "elementwise"
    elif (
        "compare" in n
        or "compareeq" in n
        or "comparefunctor" in n
        or "bunaryfunctor" in n
    ):
        params["operation"] = "elementwise"
    elif "tanh" in n:
        params["operation"] = "elementwise"
    elif "arange" in n:
        params["operation"] = "elementwise"
    elif "pow_tensor" in n:
        params["operation"] = "elementwise"
    elif "catarray" in n or "catarr" in n:
        params["operation"] = "elementwise"
    elif "elementwise" in n:
        params["operation"] = "elementwise"
    elif (
        "rsqrt" in n
        or "cos_kernel" in n
        or "sin_kernel" in n
        or "neg_kernel" in n
        or "bitwise_not" in n
    ):
        params["operation"] = "elementwise"
    elif "float16_copy" in n or "direct_copy" in n:
        params["operation"] = "elementwise"

    return params


def classify_all_kernels(
    input_path: str, output_path: str, shapes_path: Optional[str] = None
):
    """
    Read unique_kernels_compute.jsonl, classify all kernels, and write to JSON.
    """
    print(f"Reading kernels from {input_path}...")
    kernels = []
    with open(input_path, "r") as f:
        for line in f:
            kernels.append(json.loads(line))

    print(f"Found {len(kernels)} unique kernels")

    # Load shape mapping if provided
    shape_mapping = {}
    if shapes_path:
        print(f"Loading shape mapping from {shapes_path}...")
        shape_mapping = load_shape_mapping(shapes_path)
        gemm_configs = shape_mapping.get("gemm_configs", [])
        print(f"Found {len(gemm_configs)} GEMM configurations:")
        for config in gemm_configs:
            print(
                f"  {config['name']:15s}: M={config['M']:3d}, N={config['N']:5d}, K={config['K']:5d}  (×{config['count']})"
            )

    # Classify and extract parameters
    classified = []
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    tier_invocations = {1: 0, 2: 0, 3: 0, 4: 0}

    for kernel in kernels:
        tier = classify_kernel(kernel)
        tier_counts[tier] += 1
        tier_invocations[tier] += kernel["count"]

        # Extract tier-specific parameters
        if tier == 1:
            params = extract_memcpy_params(kernel)
        elif tier == 2:
            params = extract_gemm_params(kernel, shape_mapping)
        elif tier == 4:
            params = extract_nccl_params(kernel)
        else:
            params = extract_libtorch_params(kernel)

        classified.append(
            {
                "name": kernel["name"],
                "tier": tier,
                "count": kernel["count"],
                "signature": kernel.get("signature", {}),
                "params": params,
            }
        )

    # Write output
    output = {
        "summary": {
            "total_kernels": len(kernels),
            "model": shape_mapping.get("model", "unknown"),
            "tensor_parallel_size": shape_mapping.get("tensor_parallel_size", 1),
            "tier1_memory_ops": {
                "unique": tier_counts[1],
                "invocations": tier_invocations[1],
                "percentage": f"{100.0 * tier_invocations[1] / sum(tier_invocations.values()):.1f}%",
            },
            "tier2_matmul": {
                "unique": tier_counts[2],
                "invocations": tier_invocations[2],
                "percentage": f"{100.0 * tier_invocations[2] / sum(tier_invocations.values()):.1f}%",
            },
            "tier3_torch_ops": {
                "unique": tier_counts[3],
                "invocations": tier_invocations[3],
                "percentage": f"{100.0 * tier_invocations[3] / sum(tier_invocations.values()):.1f}%",
            },
            "tier4_comms": {
                "unique": tier_counts[4],
                "invocations": tier_invocations[4],
                "percentage": f"{100.0 * tier_invocations[4] / sum(tier_invocations.values()):.1f}%",
            },
        },
        "kernels": classified,
    }

    total_inv = sum(tier_invocations.values())
    print(f"\nClassification summary:")
    print(
        f"  Tier 1 (memory_ops):  {tier_counts[1]} unique, {tier_invocations[1]} invocations ({100.0 * tier_invocations[1] / total_inv:.1f}%)"
    )
    print(
        f"  Tier 2 (matmul):      {tier_counts[2]} unique, {tier_invocations[2]} invocations ({100.0 * tier_invocations[2] / total_inv:.1f}%)"
    )
    print(
        f"  Tier 3 (torch_ops):   {tier_counts[3]} unique, {tier_invocations[3]} invocations ({100.0 * tier_invocations[3] / total_inv:.1f}%)"
    )
    print(
        f"  Tier 4 (comms):       {tier_counts[4]} unique, {tier_invocations[4]} invocations ({100.0 * tier_invocations[4] / total_inv:.1f}%)"
    )

    print(f"\nWriting to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ Classification complete! Output written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify kernels into tiers for replay"
    )
    parser.add_argument(
        "--input",
        default="../../vicuna/unique_kernels_compute.jsonl",
        help="Input JSONL file with unique kernels",
    )
    parser.add_argument(
        "--output",
        default="../data/kernel_signatures.json",
        help="Output JSON file with classified kernels",
    )
    parser.add_argument(
        "--shapes",
        default="../../vicuna/shape_mapping.json",
        help="Shape mapping JSON file for accurate GEMM dimensions",
    )

    args = parser.parse_args()
    classify_all_kernels(args.input, args.output, args.shapes)


if __name__ == "__main__":
    main()
