#!/usr/bin/env python3
"""Classify BLOOM kernels into 3 tiers based on replay method.

Tier 1: Direct CUDA Runtime (cudaMemcpy, cudaMemset) - 68 kernels
Tier 2: Direct cuBLAS API (cublasGemmEx) - 23 kernels  
Tier 3: libtorch Fallback (torch ops) - 53 kernels

Reads: bloom_unique_kernels_compute.jsonl
Writes: kernel_signatures.json

Usage:
    python classify_kernels.py [--input bloom_unique_kernels_compute.jsonl] [--output kernel_signatures.json]
"""

import argparse
import json
import re
from typing import Dict, List, Any, Tuple
from collections import Counter

def load_bloom_shapes(shapes_file: str) -> List[Tuple[int, int, int]]:
    """
    Load BLOOM Linear layer shapes and extract (M, K, N) for GEMM.
    
    Returns list of (M, K, N) tuples where:
    - M = batch * seqlen
    - K = in_features
    - N = out_features
    """
    shapes = []
    try:
        with open(shapes_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get('module') == 'Linear':
                    in_shape = data['in_shape']   # [batch, seq, in_features]
                    out_shape = data['out_shape'] # [batch, seq, out_features]
                    
                    M = in_shape[0] * in_shape[1]  # batch * seqlen
                    K = in_shape[2]                # in_features
                    N = out_shape[2]               # out_features
                    
                    shapes.append((M, K, N))
    except FileNotFoundError:
        print(f"Warning: Could not find {shapes_file}, using grid-based estimates")
        return []
    
    return shapes

def classify_kernel(kernel: Dict[str, Any]) -> int:
    """
    Classify a kernel into one of 3 tiers.
    
    Returns:
        1: CUDA Runtime (cudaMemcpy, cudaMemset)
        2: cuBLAS (GEMM, GEMV)
        3: libtorch Fallback
    """
    name = kernel['name'].lower()
    
    # Tier 1: CUDA Runtime
    if 'memcpy' in name or 'memset' in name or 'memmove' in name:
        return 1
    
    # Tier 2: cuBLAS kernels
    if any(pattern in name for pattern in [
        'gemm', 'gemv', 'splitkreduce', 'sgemm', 'dgemm', 'hgemm',
        'sm80_xmma', 'maxwell_sgemm', 'volta_', 'turing_', 'ampere_'
    ]):
        return 2
    
    # Tier 3: Everything else (PyTorch native, CUB, etc.)
    return 3

def extract_memcpy_params(kernel: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for Tier 1 (memcpy/memset) kernels."""
    name = kernel['name']
    args = kernel.get('args', {})
    sig = kernel.get('signature', {})
    
    params = {
        'bytes': args.get('bytes', sig.get('bytes', 0)),
        'kind': 'unknown'
    }
    
    # Determine memcpy kind
    if 'HtoD' in name or 'Host' in name and 'Device' in name:
        params['kind'] = 'HtoD'
    elif 'DtoH' in name or 'Device' in name and 'Host' in name:
        params['kind'] = 'DtoH'
    elif 'DtoD' in name or 'Device -> Device' in name:
        params['kind'] = 'DtoD'
    elif 'memset' in name.lower():
        params['kind'] = 'memset'
    
    return params

def extract_gemm_params(kernel: Dict[str, Any], gemm_shapes: List[Tuple[int, int, int]]) -> Dict[str, Any]:
    """Extract parameters for Tier 2 (cuBLAS GEMM/GEMV) kernels."""
    sig = kernel.get('signature', {})
    args = kernel.get('args', {})
    grid = sig.get('grid', [1, 1, 1])
    block = sig.get('block', [1, 1, 1])
    
    params = {
        'grid': grid,
        'block': block,
        'shared_memory': sig.get('shared memory', 0),
        'dtype': 'fp16',  # BLOOM uses FP16
    }
    
    # Try to extract tile sizes from kernel name
    # e.g., maxwell_sgemm_fp16_128x32_tn -> tile_m=128, tile_n=32
    name = kernel['name']
    tile_match = re.search(r'(\d+)x(\d+)', name)
    if tile_match:
        params['tile_m'] = int(tile_match.group(1))
        params['tile_n'] = int(tile_match.group(2))
    
    # Use actual GEMM shapes from bloom_shapes.jsonl if available
    if gemm_shapes:
        # Match kernel to shape based on invocation count
        # BLOOM has repeating patterns of Linear layers per transformer block
        shape_counts = Counter(gemm_shapes)
        most_common_shapes = shape_counts.most_common()
        
        # For simplicity, assign the most common shape that matches the kernel type
        # GEMV kernels (gemv in name) likely correspond to smaller M
        # GEMM kernels handle the main matmuls
        if 'gemv' in name.lower():
            # GEMV is GEMM with N=1, find shapes with small M
            for (M, K, N), count in most_common_shapes:
                if M <= 10:  # Small batch dimension
                    params['M'] = M
                    params['K'] = K
                    params['N'] = 1  # GEMV has N=1
                    break
        else:
            # Regular GEMM - use most common shape
            if most_common_shapes:
                M, K, N = most_common_shapes[0][0]
                params['M'] = M
                params['K'] = K
                params['N'] = N
    
    # Fallback to grid-based estimate if no shapes available
    if 'M' not in params and 'tile_m' in params and 'tile_n' in params:
        params['M'] = grid[0] * params['tile_m']
        params['N'] = grid[1] * params['tile_n']
        params['K'] = 1024  # Default hidden size for BLOOM-560M
    
    return params

def extract_libtorch_params(kernel: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for Tier 3 (libtorch) kernels."""
    name = kernel['name']
    sig = kernel.get('signature', {})
    
    params = {
        'grid': sig.get('grid', [1, 1, 1]),
        'block': sig.get('block', [1, 1, 1]),
        'shared_memory': sig.get('shared memory', 0),
        'operation': 'unknown'
    }
    
    # Identify the operation type from kernel name (order: more specific first)
    n = name.lower()
    if 'layer_norm' in n:
        params['operation'] = 'layer_norm'
    elif 'softmax' in n:
        params['operation'] = 'softmax'
    elif 'indexselect' in n or 'index_select' in n:
        params['operation'] = 'index_select'
    elif 'gelu' in n:
        params['operation'] = 'gelu'
    elif 'reduce_kernel' in n or 'reduction_prod_kernel' in n or 'argmax' in n or 'maxnan' in n:
        params['operation'] = 'reduce'
    elif 'devicescan' in n or 'scan' in n:
        params['operation'] = 'scan'
    elif 'cudafunctor_add' in n or 'cudafunctoronself_add' in n or 'cudafunctoronother_add' in n or ('add' in n and 'cudafunctor' in n):
        params['operation'] = 'add'
    elif 'mulfunctor' in n or ('binaryfunctor' in n and 'mul' in n) or ('bunaryfunctor' in n and 'mul' in n):
        params['operation'] = 'mul'
    elif 'fillfunctor' in n:
        params['operation'] = 'fill'
    elif 'direct_copy' in n or 'direct_copy_kernel' in n:
        params['operation'] = 'elementwise'  # copy ~ elementwise
    elif 'masked_fill' in n:
        params['operation'] = 'elementwise'  # masked_fill ~ fill + mask
    elif 'compare' in n or 'compareeq' in n or 'comparefunctor' in n or 'bunaryfunctor' in n:
        params['operation'] = 'elementwise'  # compare kernels
    elif 'tanh' in n:
        params['operation'] = 'elementwise'  # tanh
    elif 'arange' in n:
        params['operation'] = 'elementwise'  # arange
    elif 'pow_tensor' in n:
        params['operation'] = 'elementwise'  # pow
    elif 'catarray' in n or 'catarr' in n:
        params['operation'] = 'elementwise'  # concat/cat copy
    elif 'elementwise' in n:
        params['operation'] = 'elementwise'
    # else remains 'unknown'
    
    return params

def classify_all_kernels(input_path: str, output_path: str, shapes_path: str = None):
    """
    Read bloom_unique_kernels_compute.jsonl, classify all kernels, and write to JSON.
    """
    print(f"Reading kernels from {input_path}...")
    kernels = []
    with open(input_path, 'r') as f:
        for line in f:
            kernels.append(json.loads(line))
    
    print(f"Found {len(kernels)} unique kernels")
    
    # Load BLOOM shapes if provided
    gemm_shapes = []
    if shapes_path:
        print(f"Loading BLOOM shapes from {shapes_path}...")
        gemm_shapes = load_bloom_shapes(shapes_path)
        print(f"Found {len(gemm_shapes)} Linear layer invocations")
        shape_counts = Counter(gemm_shapes)
        print(f"Unique GEMM shapes: {len(shape_counts)}")
        for (M, K, N), count in shape_counts.most_common(5):
            print(f"  ({M}, {K}, {N}): {count} invocations")
    
    # Classify and extract parameters
    classified = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    tier_invocations = {1: 0, 2: 0, 3: 0}
    
    for kernel in kernels:
        tier = classify_kernel(kernel)
        tier_counts[tier] += 1
        tier_invocations[tier] += kernel['count']
        
        # Extract tier-specific parameters
        if tier == 1:
            params = extract_memcpy_params(kernel)
        elif tier == 2:
            params = extract_gemm_params(kernel, gemm_shapes)
        else:
            params = extract_libtorch_params(kernel)
        
        classified.append({
            'name': kernel['name'],
            'tier': tier,
            'count': kernel['count'],
            'signature': kernel.get('signature', {}),
            'params': params
        })
    
    # Write output
    output = {
        'summary': {
            'total_kernels': len(kernels),
            'tier1_cuda_runtime': {
                'unique': tier_counts[1],
                'invocations': tier_invocations[1],
                'percentage': f"{100.0 * tier_invocations[1] / sum(tier_invocations.values()):.1f}%"
            },
            'tier2_cublas': {
                'unique': tier_counts[2],
                'invocations': tier_invocations[2],
                'percentage': f"{100.0 * tier_invocations[2] / sum(tier_invocations.values()):.1f}%"
            },
            'tier3_libtorch': {
                'unique': tier_counts[3],
                'invocations': tier_invocations[3],
                'percentage': f"{100.0 * tier_invocations[3] / sum(tier_invocations.values()):.1f}%"
            }
        },
        'kernels': classified
    }
    
    print(f"\nClassification summary:")
    print(f"  Tier 1 (CUDA Runtime): {tier_counts[1]} unique, {tier_invocations[1]} invocations ({100.0 * tier_invocations[1] / sum(tier_invocations.values()):.1f}%)")
    print(f"  Tier 2 (cuBLAS):       {tier_counts[2]} unique, {tier_invocations[2]} invocations ({100.0 * tier_invocations[2] / sum(tier_invocations.values()):.1f}%)")
    print(f"  Tier 3 (libtorch):     {tier_counts[3]} unique, {tier_invocations[3]} invocations ({100.0 * tier_invocations[3] / sum(tier_invocations.values()):.1f}%)")
    
    print(f"\nWriting to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Classification complete! Output written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Classify BLOOM kernels into tiers")
    parser.add_argument(
        "--input",
        default="../../bloom/bloom_unique_kernels_compute.jsonl",
        help="Input JSONL file with unique kernels"
    )
    parser.add_argument(
        "--output",
        default="../data/kernel_signatures.json",
        help="Output JSON file with classified kernels"
    )
    parser.add_argument(
        "--shapes",
        default="../../bloom/bloom_shapes.jsonl",
        help="BLOOM shapes JSONL file for accurate GEMM dimensions"
    )
    
    args = parser.parse_args()
    classify_all_kernels(args.input, args.output, args.shapes)

if __name__ == "__main__":
    main()
