#!/usr/bin/env python3
"""Kernel replay benchmark harness.

Replays unique kernels extracted from Vicuna-7B TP inference traces.
Measures latency (and optionally energy via external power probes).

Usage:
    python kernel_benchmark.py --kernels unique_kernels_compute.jsonl --iterations 100 --output results.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable

import torch


@dataclass
class KernelConfig:
    """Configuration for a kernel to benchmark."""
    name: str
    category: str  # "memcpy_htod", "memcpy_dtod", "memcpy_dtoh", "memset", "gemm", "gemm_relu"
    params: Dict   # bytes for mem ops; m, n, k, batch for GEMM
    original_count: int
    device: int
    grid: Optional[List[int]] = None
    block: Optional[List[int]] = None
    shared_memory: Optional[int] = None


@dataclass
class BenchmarkResult:
    """Result of benchmarking a kernel."""
    name: str
    category: str
    params: Dict
    device: int
    latency_ms: float
    latency_std_ms: float
    iterations: int
    original_count: int
    throughput_gbps: Optional[float] = None  # For memory ops
    tflops: Optional[float] = None  # For GEMM


# Tile sizes for different kernel types (decoded from kernel names)
TILE_SIZES = {
    # ampere_fp16_s16816gemm variants use 64x64 tiles
    "ampere_fp16": (64, 64),
    # cutlass_80_wmma_tensorop_f16_s161616gemm uses 16x16 tiles
    "wmma_tensorop_f16_s161616": (16, 16),
    # cutlass_80_tensorop_f16_s16816gemm_relu uses 256x64 tiles
    "tensorop_f16_s16816gemm_relu_f16_256x64": (256, 64),
    "tensorop_f16_s16816gemm": (128, 64),  # default for non-relu
}


def infer_tile_size(kernel_name: str) -> tuple:
    """Infer tile size from kernel name."""
    name_lower = kernel_name.lower()
    
    if "256x64" in name_lower:
        return (256, 64)
    elif "64x64" in kernel_name:
        return (64, 64)
    elif "wmma" in name_lower and "16x16" in name_lower:
        return (16, 16)
    elif "ampere_fp16" in name_lower:
        return (64, 64)
    else:
        return (64, 64)  # default


def infer_k_dim(kernel_name: str, m: int, n: int) -> int:
    """Infer K dimension based on kernel context.
    
    For Vicuna-7B with TP=2:
    - hidden_dim = 4096, per-GPU = 2048
    - intermediate_dim = 11008, per-GPU = 5504
    """
    # These are heuristics based on Vicuna-7B architecture
    if m <= 128 and n <= 128:
        # Small attention-related GEMM
        return 64  # head_dim
    elif m == 2048 or n == 2048:
        # Per-GPU hidden dim
        return 2048
    elif m == 4096 or n == 4096:
        # Full hidden dim
        return 4096
    else:
        # Default to hidden_dim / TP
        return 2048


def load_kernel_configs(jsonl_path: str) -> List[KernelConfig]:
    """Load kernel configs from extracted unique kernels JSONL."""
    configs = []
    
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num}: {e}", file=sys.stderr)
                continue
            
            name = rec.get("name", "")
            sig = rec.get("signature", {})
            device = sig.get("device", 0)
            count = rec.get("count", 1)
            
            grid = sig.get("grid")
            block = sig.get("block")
            shared_mem = sig.get("shared memory")
            
            # Categorize kernel
            if "Memcpy HtoD" in name:
                category = "memcpy_htod"
                params = {"bytes": sig.get("bytes", 0)}
            elif "Memcpy DtoD" in name:
                category = "memcpy_dtod"
                params = {"bytes": sig.get("bytes", 0)}
            elif "Memcpy DtoH" in name:
                category = "memcpy_dtoh"
                params = {"bytes": sig.get("bytes", 0)}
            elif "Memset" in name:
                category = "memset"
                params = {"bytes": sig.get("bytes", 0)}
            elif "gemm" in name.lower():
                # GEMM kernel - infer dimensions from grid and tile size
                if grid is None:
                    print(f"Warning: No grid for GEMM {name}", file=sys.stderr)
                    continue
                
                tile_m, tile_n = infer_tile_size(name)
                m = grid[0] * tile_m
                n = grid[1] * tile_n
                batch = grid[2] if len(grid) > 2 and grid[2] > 1 else 1
                k = infer_k_dim(name, m, n)
                
                category = "gemm_relu" if "relu" in name.lower() else "gemm"
                params = {"m": m, "n": n, "k": k, "batch": batch}
            else:
                # Unknown kernel type
                print(f"Warning: Unknown kernel type: {name}", file=sys.stderr)
                continue
            
            configs.append(KernelConfig(
                name=name,
                category=category,
                params=params,
                original_count=count,
                device=device,
                grid=grid,
                block=block,
                shared_memory=shared_mem,
            ))
    
    return configs


def create_benchmark_op(cfg: KernelConfig) -> Callable:
    """Create a callable that performs the kernel operation."""
    torch.cuda.set_device(cfg.device)
    
    if cfg.category == "memcpy_htod":
        nbytes = cfg.params["bytes"]
        src = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        return lambda: dst.copy_(src)
    
    elif cfg.category == "memcpy_dtod":
        nbytes = cfg.params["bytes"]
        src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        return lambda: dst.copy_(src)
    
    elif cfg.category == "memcpy_dtoh":
        nbytes = cfg.params["bytes"]
        src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        dst = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        return lambda: dst.copy_(src)
    
    elif cfg.category == "memset":
        nbytes = cfg.params["bytes"]
        tensor = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        return lambda: tensor.zero_()
    
    elif cfg.category in ("gemm", "gemm_relu"):
        m, n, k = cfg.params["m"], cfg.params["n"], cfg.params["k"]
        batch = cfg.params.get("batch", 1)
        
        if batch > 1:
            A = torch.randn(batch, m, k, dtype=torch.float16, device="cuda")
            B = torch.randn(batch, k, n, dtype=torch.float16, device="cuda")
        else:
            A = torch.randn(m, k, dtype=torch.float16, device="cuda")
            B = torch.randn(k, n, dtype=torch.float16, device="cuda")
        
        if cfg.category == "gemm_relu":
            return lambda: torch.relu(torch.matmul(A, B))
        else:
            return lambda: torch.matmul(A, B)
    
    else:
        raise ValueError(f"Unknown category: {cfg.category}")


def benchmark_kernel(cfg: KernelConfig, iterations: int = 100, warmup: int = 10) -> BenchmarkResult:
    """Run benchmark for a single kernel configuration."""
    torch.cuda.set_device(cfg.device)
    
    op = create_benchmark_op(cfg)
    
    # Warmup
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    
    # Collect individual timings for std calculation
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        op()
        end.record()
        torch.cuda.synchronize()
        
        timings.append(start.elapsed_time(end))
    
    latency_ms = sum(timings) / len(timings)
    latency_std = (sum((t - latency_ms) ** 2 for t in timings) / len(timings)) ** 0.5
    
    # Compute derived metrics
    throughput_gbps = None
    tflops = None
    
    if cfg.category in ("memcpy_htod", "memcpy_dtod", "memcpy_dtoh", "memset"):
        nbytes = cfg.params["bytes"]
        if latency_ms > 0:
            throughput_gbps = (nbytes / 1e9) / (latency_ms / 1000)
    
    elif cfg.category in ("gemm", "gemm_relu"):
        m, n, k = cfg.params["m"], cfg.params["n"], cfg.params["k"]
        batch = cfg.params.get("batch", 1)
        flops = 2 * batch * m * n * k  # 2 ops per multiply-add
        if latency_ms > 0:
            tflops = (flops / 1e12) / (latency_ms / 1000)
    
    return BenchmarkResult(
        name=cfg.name,
        category=cfg.category,
        params=cfg.params,
        device=cfg.device,
        latency_ms=latency_ms,
        latency_std_ms=latency_std,
        iterations=iterations,
        original_count=cfg.original_count,
        throughput_gbps=throughput_gbps,
        tflops=tflops,
    )


def run_benchmarks(configs: List[KernelConfig], iterations: int = 100, warmup: int = 10) -> List[BenchmarkResult]:
    """Run benchmarks for all kernel configurations."""
    results = []
    
    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Benchmarking: {cfg.name[:60]}...", end=" ", flush=True)
        
        try:
            result = benchmark_kernel(cfg, iterations=iterations, warmup=warmup)
            results.append(result)
            
            # Print summary
            if result.throughput_gbps is not None:
                print(f"{result.latency_ms:.4f} ms, {result.throughput_gbps:.2f} GB/s")
            elif result.tflops is not None:
                print(f"{result.latency_ms:.4f} ms, {result.tflops:.2f} TFLOPS")
            else:
                print(f"{result.latency_ms:.4f} ms")
        
        except Exception as e:
            print(f"FAILED: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Kernel replay benchmark harness.")
    parser.add_argument("--kernels", default="unique_kernels_compute.jsonl",
                        help="Path to unique kernels JSONL file")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations per kernel")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--output", default="kernel_latency_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--device", type=int, default=None,
                        help="Override device (benchmark all on this GPU)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter kernels by name substring")
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        return 1
    
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load kernel configs
    print(f"Loading kernels from: {args.kernels}")
    configs = load_kernel_configs(args.kernels)
    print(f"Loaded {len(configs)} kernel configurations")
    
    # Apply filters
    if args.filter:
        configs = [c for c in configs if args.filter.lower() in c.name.lower()]
        print(f"After filter '{args.filter}': {len(configs)} kernels")
    
    if args.device is not None:
        for c in configs:
            c.device = args.device
        print(f"Overriding device to: {args.device}")
    
    # Remove duplicates (same kernel on different GPUs, just benchmark once)
    seen = set()
    unique_configs = []
    for c in configs:
        key = (c.name, c.category, tuple(sorted(c.params.items())))
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    
    print(f"Unique kernels to benchmark: {len(unique_configs)}")
    print()
    
    # Run benchmarks
    results = run_benchmarks(unique_configs, iterations=args.iterations, warmup=args.warmup)
    
    # Save results
    output_data = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": [asdict(r) for r in results],
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print(f"Results saved to: {args.output}")
    
    # Summary table
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Category':<15} {'Count':>6} {'Avg Latency (ms)':>18} {'Metric':>20}")
    print("-" * 100)
    
    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)
    
    for cat, cat_results in sorted(by_category.items()):
        avg_lat = sum(r.latency_ms for r in cat_results) / len(cat_results)
        
        if cat_results[0].throughput_gbps is not None:
            avg_metric = sum(r.throughput_gbps for r in cat_results if r.throughput_gbps) / len(cat_results)
            metric_str = f"{avg_metric:.2f} GB/s"
        elif cat_results[0].tflops is not None:
            avg_metric = sum(r.tflops for r in cat_results if r.tflops) / len(cat_results)
            metric_str = f"{avg_metric:.2f} TFLOPS"
        else:
            metric_str = "N/A"
        
        print(f"{cat:<15} {len(cat_results):>6} {avg_lat:>18.4f} {metric_str:>20}")
    
    print("=" * 100)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
