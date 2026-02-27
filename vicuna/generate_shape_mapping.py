#!/usr/bin/env python3
"""Generate shape mapping from Linear layer shape logs for C++ kernel replay.

This script processes shape_log_rank*.jsonl files and creates a JSON mapping
of GEMM dimensions (M, N, K) for accurate kernel replay in C++.
"""

import json
import os
import sys


def main():
    # Load shape logs from BOTH ranks (TP=2 means different shapes per rank!)
    rank0_file = "shape_log_rank0.jsonl"
    rank1_file = "shape_log_rank1.jsonl"

    if not os.path.exists(rank0_file):
        print(f"ERROR: {rank0_file} not found. Run profiling first.", file=sys.stderr)
        return 1
    if not os.path.exists(rank1_file):
        print(f"ERROR: {rank1_file} not found. Run profiling first.", file=sys.stderr)
        return 1

    print("=" * 80)
    print("Generating Shape Mapping for Kernel Replay (TP=2)")
    print("=" * 80)

    # Load shapes from both ranks
    with open(rank0_file, "r") as f:
        shapes_rank0 = [json.loads(line) for line in f]
    with open(rank1_file, "r") as f:
        shapes_rank1 = [json.loads(line) for line in f]

    print(f"\nLoaded {len(shapes_rank0)} Linear layer shapes from rank 0")
    print(f"Loaded {len(shapes_rank1)} Linear layer shapes from rank 1")

    # For TP=2, we need to analyze BOTH ranks because tensor parallelism
    # splits tensors across GPUs (different K or N per rank)
    # Use rank 0 for analysis (both ranks see all operations)
    shapes = shapes_rank0

    # Count unique GEMM configurations
    gemm_configs = {}
    for s in shapes:
        if s["in_shape"] and s["out_shape"] and len(s["in_shape"]) == 3:
            batch, seq_len, in_feat = s["in_shape"]
            _, _, out_feat = s["out_shape"]

            # GEMM dimensions: M=batch*seq_len, N=out_features, K=in_features
            M = batch * seq_len
            N = out_feat
            K = in_feat

            key = (M, N, K)
            if key not in gemm_configs:
                gemm_configs[key] = {
                    "count": 1,
                    "in_shape": s["in_shape"],
                    "out_shape": s["out_shape"],
                }
            else:
                gemm_configs[key]["count"] += 1

    print(f"\nFound {len(gemm_configs)} unique GEMM configurations:")
    print("-" * 80)
    print(f"{'M':>5} {'N':>6} {'K':>6}  {'Count':>6}  Input Shape -> Output Shape")
    print("-" * 80)

    gemm_list = []
    for (M, N, K), info in sorted(gemm_configs.items(), key=lambda x: -x[1]["count"]):
        print(
            f"{M:5d} {N:6d} {K:6d}  {info['count']:6d}  {str(info['in_shape']):20s} -> {str(info['out_shape'])}"
        )

        # Determine operation type from TP=2 sharded dimensions
        # NOTE: These are SHARDED dimensions (after TP split on 2 GPUs)
        # Vicuna-7B: hidden=4096, intermediate=11008, num_heads=32, head_dim=128
        # ColwiseParallel splits output N: qkv 4096->2048, gate/up 11008->5504
        # RowwiseParallel splits input K: o_proj 4096->2048, down 11008->5504
        if K == 4096 and N == 2048:
            op_name = "qkv_proj"  # Full N=4096 -> Sharded N=2048 (q,k,v projections)
        elif K == 4096 and N == 5504:
            op_name = "ffn_gate_up"  # Full: K=4096, N=11008 -> Sharded N=5504
        elif K == 2048 and N == 4096:
            op_name = "o_proj"  # Full: K=4096 -> Sharded K=2048
        elif K == 5504 and N == 4096:
            op_name = "ffn_down"  # Full: K=11008 -> Sharded K=5504
        elif K == 4096 and N == 32000:
            op_name = "lm_head"  # Not sharded (or replicated)
        else:
            op_name = "unknown"

        gemm_list.append(
            {
                "name": op_name,
                "M": M,
                "N": N,
                "K": K,
                "count": info["count"],
                "in_shape": info["in_shape"],
                "out_shape": info["out_shape"],
            }
        )

    # Create shape mapping
    shape_mapping = {
        "model": "lmsys/vicuna-7b-v1.5",
        "tensor_parallel_size": 2,
        "prompt": "Explain tensor parallelism in one paragraph.",
        "max_new_tokens": 64,
        "batch_size": 1,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "note": "GEMM dimensions: M×K @ K×N = M×N where M=batch*seq_len. "
        "IMPORTANT: These are TP=2 SHARDED dimensions (each GPU sees half the tensors). "
        "Vicuna-7B: hidden=4096, intermediate=11008, num_heads=32, head_dim=128. "
        "ColwiseParallel splits N (qkv: 4096->2048, gate/up: 11008->5504). "
        "RowwiseParallel splits K (o_proj: 4096->2048, down: 11008->5504).",
        "gemm_configs": gemm_list,
    }

    # Write to file
    output_file = "shape_mapping.json"
    with open(output_file, "w") as f:
        json.dump(shape_mapping, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Shape mapping saved to: {output_file}")
    print("=" * 80)
    print("\nKey GEMM operations for C++ replay (TP=2 SHARDED dimensions):")
    print("-" * 80)

    for cfg in gemm_list:
        if cfg["name"] != "unknown":
            print(
                f"  {cfg['name']:15s}: M={cfg['M']:3d}, N={cfg['N']:5d}, K={cfg['K']:5d}  (×{cfg['count']:3d})"
            )

    print("\n⚠️  IMPORTANT: These are TP=2 sharded dimensions (per-GPU)!")
    print("   Each GPU runs kernels with these shapes (not full model shapes).")
    print("\nThis mapping will be used by classify_kernels.py to assign")
    print("M/N/K dimensions to extracted GEMM kernels for accurate replay.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
