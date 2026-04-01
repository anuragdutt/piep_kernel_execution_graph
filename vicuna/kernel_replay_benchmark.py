#!/usr/bin/env python3
"""
Python Kernel Replay Benchmark for Vicuna-7B TP=2

Replays all extracted CUDA kernels in pure Python/PyTorch, achieving
language and framework parity with the full Vicuna inference benchmark.
This eliminates the C++ vs Python overhead discrepancy that inflates the
prediction error in energy comparisons.

Tier mapping (C++ → Python):
  Tier 1 (memcpy/memset) → torch tensor ops: .fill_(), .copy_(), torch.zeros()
  Tier 2 (cuBLAS GEMM)   → torch.matmul / F.linear in fp16 (calls identical cuBLAS kernels)
  Tier 3 (libtorch ops)  → torch.* functional ops with shapes from shape_log
  Tier 4 (NCCL AllReduce) → torch.distributed.all_reduce (same NCCL backend)

Usage (single GPU, Tiers 1-3):
  python kernel_replay_benchmark.py --kernels results/<run>/kernel_signatures.json

Usage (with NCCL Tier 4, requires torchrun):
  torchrun --nproc_per_node=2 kernel_replay_benchmark.py --kernels results/<run>/kernel_signatures.json --nccl

Output:
  isolated_kernels_timing.json  (same schema as C++ kernel_benchmark output)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# CUDA APIs excluded from energy accounting (return -1.0 → caller skips kernel).
#
# Three categories are excluded:
#
#   1. One-time setup calls (cudaFuncSetAttribute, cudaOccupancyMax*):
#      fired during model load, not in the hot inference loop.
#
#   2. Inter-GPU synchronisation primitives (cudaStreamWaitEvent,
#      cudaEventRecord, cudaEventQuery):
#      In a TP run the compute stream posts a cudaStreamWaitEvent and then
#      stalls until the NCCL AllReduce signals completion.  The *host-side*
#      call itself is ~1-3 µs, but the real cost — the GPU stall duration —
#      is variable, depends on AllReduce latency, and is already fully
#      captured by the Tier 4 NCCL replay.  Assigning a fixed per-invocation
#      cost here would double-count and be wrong.
_CUDA_API_EXCLUDED = frozenset(
    {
        # Setup / query (never in hot path)
        "cudaFuncSetAttribute",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        # Inter-GPU sync (variable wait time already in AllReduce / Tier 4)
        "cudaStreamWaitEvent",
        "cudaEventRecord",
        "cudaEventQuery",
    }
)

# CUDA APIs that are CPU/driver-bound and CAN be faithfully replayed:
# measured with wall-clock time (CUDA events capture zero device-side time
# for host calls).
_CUDA_API_WALLCLOCK = frozenset(
    {
        "cudaStreamSynchronize",
        "cudaLaunchKernel",
        "cudaLaunchKernelExC",
        "cudaPeekAtLastError",
        "cudaDeviceGetAttribute",
        "cudaStreamIsCapturing",
        "cudaStreamGetCaptureInfo_v2",
        "cudaDeviceSynchronize",
        # cuLaunchKernel (lowercase cu) is NOT in C++'s known set → falls to fill proxy
    }
)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def cuda_timed(fn, warmup: int, runs: int) -> float:
    """
    Time a CUDA function using CUDA events for accurate device-side timing.
    Returns average time per call in microseconds.
    Use this for GPU kernels (GEMM, elementwise, etc.).
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return (elapsed_ms * 1000.0) / runs  # microseconds per call


def wall_timed(fn, warmup: int, runs: int) -> float:
    """
    Time a function using wall-clock time (time.perf_counter).
    Returns average time per call in microseconds.
    Use this for CPU/driver-side CUDA API calls (cudaEventQuery, etc.) where
    CUDA events capture zero device-side time.
    Mirrors C++ benchmark_cpu_us() in cuda_api_replay.cpp.
    """
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / runs  # microseconds per call


# ---------------------------------------------------------------------------
# Tier 1: memcpy / memset  →  torch tensor ops
# ---------------------------------------------------------------------------


def replay_tier1(kernel: Dict, device: torch.device, runs: int, warmup: int) -> float:
    """
    Replay memcpy/memset CUDA runtime calls via torch tensor operations.
    torch.Tensor.fill_() lowers to cudaMemset.
    torch.Tensor.copy_()  lowers to cudaMemcpy (D2D) or cudaMemcpyAsync (H2D/D2H).
    """
    params = kernel.get("params", {})
    kind = params.get("kind", "unknown")
    nbytes = params.get("bytes", 0)

    if nbytes <= 0:
        nbytes = 1024  # default if not recorded

    n_elems = max(1, nbytes // 2)  # fp16 = 2 bytes per element

    if kind in ("memset", "unknown"):
        buf = torch.zeros(n_elems, dtype=torch.float16, device=device)
        return cuda_timed(lambda: buf.fill_(0.0), warmup, runs)

    elif kind == "DtoD":
        src = torch.zeros(n_elems, dtype=torch.float16, device=device)
        dst = torch.zeros(n_elems, dtype=torch.float16, device=device)
        return cuda_timed(lambda: dst.copy_(src), warmup, runs)

    elif kind == "HtoD":
        src = torch.zeros(n_elems, dtype=torch.float16, device="cpu").pin_memory()
        dst = torch.zeros(n_elems, dtype=torch.float16, device=device)
        return cuda_timed(lambda: dst.copy_(src, non_blocking=True), warmup, runs)

    elif kind == "DtoH":
        src = torch.zeros(n_elems, dtype=torch.float16, device=device)
        dst = torch.zeros(n_elems, dtype=torch.float16, device="cpu").pin_memory()
        return cuda_timed(lambda: dst.copy_(src, non_blocking=True), warmup, runs)

    else:
        buf = torch.zeros(n_elems, dtype=torch.float16, device=device)
        return cuda_timed(lambda: buf.fill_(0.0), warmup, runs)


# ---------------------------------------------------------------------------
# Tier 2: cuBLAS GEMM  →  torch.matmul in fp16
# ---------------------------------------------------------------------------


def replay_tier2(kernel: Dict, device: torch.device, runs: int, warmup: int) -> float:
    """
    Replay cuBLAS GEMM/GEMV via torch.matmul in fp16.
    PyTorch dispatches torch.matmul to cublasGemmEx with CUBLAS_COMPUTE_16F,
    which is the identical kernel that the C++ benchmark calls.

    For a GEMM: C[M,N] = A[M,K] @ B[K,N]
    """
    params = kernel.get("params", {})
    M = params.get("M", 0)
    N = params.get("N", 0)
    K = params.get("K", 0)

    # Fallbacks matching C++ defaults
    if M <= 0 or N <= 0 or K <= 0:
        op = params.get("operation", "gemm")
        if op == "gemv":
            M, N, K = 1, 1, 4096
        else:
            M, N, K = 57, 4096, 4096

    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)

    return cuda_timed(lambda: torch.matmul(A, B), warmup, runs)


# ---------------------------------------------------------------------------
# Tier 3: libtorch ops  →  torch.* functional equivalents
# ---------------------------------------------------------------------------


def _infer_shape_from_params(params: Dict) -> Tuple[int, int, int]:
    """
    Infer [batch, seq_len, feat] from grid/block dimensions.
    Mirrors the C++ heuristic in libtorch_kernels.cpp::infer_shape_from_grid().
    """
    grid = params.get("grid", [1, 1, 1])
    block = params.get("block", [1, 1, 1])
    total_threads = grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2]

    if total_threads <= 512:
        feat = 128
    elif total_threads <= 2560:
        feat = 512
    elif total_threads <= 5120:
        feat = 1024
    elif total_threads <= 15360:
        feat = 3072
    elif total_threads <= 20480:
        feat = 4096
    else:
        feat = min(total_threads // 4, 8192)

    return 1, 5, feat


def replay_tier3(
    kernel: Dict,
    device: torch.device,
    runs: int,
    warmup: int,
    shape_map: Optional[Dict] = None,
) -> float:
    """
    Replay libtorch ops via PyTorch equivalents.
    Uses actual shapes from shape_log where available, otherwise falls back
    to the grid-based heuristic.
    """
    params = kernel.get("params", {})
    operation = params.get("operation", "elementwise")

    # Infer shape from grid/block heuristic, then refine with shape_map if available.
    # shape_map is keyed by str(feat_dim) — we look up the representative shape for
    # the feature dimension suggested by the grid heuristic.
    batch, seq, feat = _infer_shape_from_params(params)
    if shape_map and str(feat) in shape_map:
        batch, seq, feat = shape_map[str(feat)]

    feat = max(feat, 1)
    shape = (batch, seq, feat)

    # ---- operation dispatch ------------------------------------------------

    if operation == "layer_norm" or operation == "rms_norm":
        x = torch.randn(shape, dtype=torch.float16, device=device)
        w = torch.ones(feat, dtype=torch.float16, device=device)
        b = torch.zeros(feat, dtype=torch.float16, device=device)
        return cuda_timed(
            lambda: F.layer_norm(x, [feat], weight=w, bias=b), warmup, runs
        )

    elif operation == "softmax":
        x = torch.randn(shape, dtype=torch.float16, device=device)
        return cuda_timed(lambda: torch.softmax(x, dim=-1), warmup, runs)

    elif operation == "add":
        a = torch.randn(shape, dtype=torch.float16, device=device)
        b = torch.randn(shape, dtype=torch.float16, device=device)
        return cuda_timed(lambda: torch.add(a, b), warmup, runs)

    elif operation == "mul":
        a = torch.randn(shape, dtype=torch.float16, device=device)
        b = torch.randn(shape, dtype=torch.float16, device=device)
        return cuda_timed(lambda: torch.mul(a, b), warmup, runs)

    elif operation == "fill":
        x = torch.randn(shape, dtype=torch.float16, device=device)
        return cuda_timed(lambda: x.fill_(0.0), warmup, runs)

    elif operation == "index_select":
        x = torch.randn(shape, dtype=torch.float16, device=device)
        idx = torch.zeros(seq, dtype=torch.long, device=device)
        return cuda_timed(lambda: torch.index_select(x, 1, idx), warmup, runs)

    elif operation == "gelu":
        x = torch.randn(shape, dtype=torch.float16, device=device)
        return cuda_timed(lambda: F.gelu(x), warmup, runs)

    elif operation == "silu":
        x = torch.randn(shape, dtype=torch.float16, device=device)
        return cuda_timed(lambda: F.silu(x), warmup, runs)

    elif operation == "reduce":
        x = torch.randn(shape, dtype=torch.float16, device=device)
        return cuda_timed(lambda: torch.sum(x, dim=-1), warmup, runs)

    elif operation == "scan":
        # C++ benchmark_scan uses torch::kLong (int64) to match CUB scan kernel dtype
        x = torch.randint(0, 10, shape, dtype=torch.long, device=device)
        return cuda_timed(lambda: torch.cumsum(x, dim=-1), warmup, runs)

    elif operation == "flash_attention" or operation == "rotary_embedding":
        # No direct PyTorch equivalent - use elementwise proxy of same size
        a = torch.randn(shape, dtype=torch.float16, device=device)
        b = torch.randn(shape, dtype=torch.float16, device=device)
        return cuda_timed(lambda: torch.add(a, b), warmup, runs)

    elif operation == "cuda_api":
        # Mirror C++ cuda_api_replay.cpp exactly:
        #   - Excluded APIs (cudaFuncSetAttribute, cudaOccupancyMax*): return -1 → caller skips
        #   - CPU/driver-bound APIs: measure with wall-clock (CUDA events = 0 device time)
        #   - Unknown names (e.g. cuLaunchKernel): fill proxy, CUDA event ok
        import ctypes

        api_name = kernel.get("name", "")

        if api_name in _CUDA_API_EXCLUDED:
            # C++ returns -1.0 for these; excluded from results by aggregation.cpp
            return -1.0

        if api_name in _CUDA_API_WALLCLOCK:
            # CPU/driver-bound: measured with wall-clock, same as C++ benchmark_cpu_us().
            # cudaEventRecord / cudaEventQuery / cudaStreamWaitEvent are in _CUDA_API_EXCLUDED
            # above and never reach here.
            try:
                _cudart = ctypes.CDLL("libcudart.so")
            except OSError:
                _cudart = None

            if api_name == "cudaStreamSynchronize":
                # Proxy: synchronize an already-idle stream.
                # The real call is a fence; when the stream is already drained it
                # returns in ~1-5 µs.  The variable blocking time (while waiting
                # for AllReduce) is already accounted for in the Tier 4 NCCL replay.
                torch.cuda.synchronize()  # drain once before timing
                return wall_timed(lambda: torch.cuda.synchronize(), warmup, runs)

            elif api_name in ("cudaLaunchKernel", "cudaLaunchKernelExC"):
                # Proxy: measure the host-side kernel-dispatch overhead only.
                # The previous fill_() proxy executed a real GPU kernel and measured
                # wall time including GPU execution (~79 µs) — that is wrong.
                # The real cudaLaunchKernel just posts a launch descriptor to the
                # driver and returns in ~2-5 µs.  We approximate this with a ctypes
                # call to cudaPeekAtLastError (another fast driver round-trip) as a
                # stand-in for the dispatch overhead.  If ctypes is unavailable, use
                # a trivial lambda that exercises the Python→C extension boundary.
                if _cudart:
                    return wall_timed(
                        lambda: _cudart.cudaPeekAtLastError(), warmup, runs
                    )
                return wall_timed(lambda: torch.cuda.current_stream(), warmup, runs)

            elif api_name == "cudaPeekAtLastError":
                if _cudart:
                    return wall_timed(
                        lambda: _cudart.cudaPeekAtLastError(), warmup, runs
                    )
                return wall_timed(lambda: torch.cuda.check_error(0), warmup, runs)

            elif api_name == "cudaDeviceGetAttribute":
                if _cudart:
                    val = ctypes.c_int(0)
                    return wall_timed(
                        lambda: _cudart.cudaDeviceGetAttribute(
                            ctypes.byref(val),
                            16,
                            device.index,  # 16 = multiProcessorCount
                        ),
                        warmup,
                        runs,
                    )
                return wall_timed(
                    lambda: torch.cuda.get_device_properties(device), warmup, runs
                )

            elif api_name in ("cudaStreamIsCapturing", "cudaStreamGetCaptureInfo_v2"):
                # Host-side flag check; no Python equivalent — proxy as trivial host call.
                return wall_timed(lambda: torch.cuda.current_stream(), warmup, runs)

            elif api_name == "cudaDeviceSynchronize":
                # Same reasoning as cudaStreamSynchronize: measure idle-device cost only.
                torch.cuda.synchronize()  # drain once before timing
                return wall_timed(lambda: torch.cuda.synchronize(), warmup, runs)

            else:
                # Other known-but-unimplemented wallclock APIs: trivial host proxy.
                return wall_timed(lambda: torch.cuda.current_stream(), warmup, runs)

        else:
            # Unknown cuda_api (e.g. cuLaunchKernel): C++ falls to fill proxy.
            x = torch.zeros(1, dtype=torch.float16, device=device)
            return cuda_timed(lambda: x.fill_(0.0), warmup, runs)

    else:  # elementwise, unknown
        a = torch.randn(shape, dtype=torch.float16, device=device)
        b = torch.randn(shape, dtype=torch.float16, device=device)
        return cuda_timed(lambda: torch.add(a, b), warmup, runs)


# ---------------------------------------------------------------------------
# Tier 4: NCCL AllReduce  →  torch.distributed.all_reduce
# ---------------------------------------------------------------------------


def replay_tier4(kernel: Dict, device: torch.device, runs: int, warmup: int) -> float:
    """
    Replay NCCL AllReduce via torch.distributed.all_reduce.
    PyTorch uses the same NCCL backend, so the identical ncclAllReduce kernel
    is dispatched - this is true kernel-level parity.

    Requires torch.distributed to be initialized (torchrun --nproc_per_node=2).
    Falls back to a no-op if distributed is not available.
    """
    if not dist.is_available() or not dist.is_initialized():
        # Fallback: simulate with a local reduce (not accurate but safe)
        params = kernel.get("params", {})
        nelems = params.get("nelems", 94208)
        buf = torch.zeros(nelems, dtype=torch.float16, device=device)
        return cuda_timed(lambda: buf.fill_(0.0), warmup, runs)

    params = kernel.get("params", {})
    nelems = params.get("nelems", 94208)
    dtype_str = params.get("dtype", "Half").lower()
    dtype = (
        torch.float16 if "half" in dtype_str or "fp16" in dtype_str else torch.float32
    )

    buf = torch.randn(nelems, dtype=dtype, device=device)
    return cuda_timed(lambda: dist.all_reduce(buf, op=dist.ReduceOp.SUM), warmup, runs)


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

# Adaptive run counts matching C++ benchmark (calibrated for 1 Hz power logger,
# target ≥2 s per kernel so at least 2 power samples fall in each window).

TIER_RUNS = {
    1: 2_000_000,  # memcpy/memset
    2: 200_000,  # cuBLAS GEMM
    3: 1_000_000,  # libtorch ops
    4: 100_000,  # NCCL AllReduce
}
TIER_WARMUP = {
    1: 1_000,
    2: 200,
    3: 1_000,
    4: 50,
}


def benchmark_kernel(
    kernel: Dict,
    default_device: torch.device,
    shape_map: Optional[Dict],
    runs_override: Optional[int],
    warmup_override: Optional[int],
    nvml_handles: Optional[List] = None,
) -> Optional[Dict]:
    """
    Benchmark a single kernel: warmup + timed loop with wall-clock timestamps.
    Returns a timing record in the same schema as isolated_kernels_timing.json,
    or None if the kernel is excluded (single_time_us == -1.0).

    Matches C++ aggregation.cpp behavior: only kernels with avg_single_time_us >= 0
    are included in the output/energy sum. Kernels like cudaFuncSetAttribute that
    return -1.0 are excluded entirely.

    Device selection (per-kernel):
      - Tier 2 and most Tier 3 compute kernels carry signature.device ∈ {0, 1}
        (one entry per rank from extract_vicuna_kernels.py).  We run each kernel
        on the GPU that originally executed it.
      - Tier 1 (memcpy/memset), Tier 3 cuda_api, and Tier 4 NCCL kernels do not
        carry a device field; they fall back to default_device.
    """
    tier = kernel["tier"]
    runs = runs_override if runs_override is not None else TIER_RUNS.get(tier, 10_000)
    warmup = (
        warmup_override if warmup_override is not None else TIER_WARMUP.get(tier, 20)
    )

    # Resolve per-kernel device from signature.device when present.
    sig_device = kernel.get("signature", {}).get("device")
    if sig_device is not None:
        device = torch.device("cuda", int(sig_device))
    else:
        device = default_device

    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    start_wall = time.time()

    # Read NVML energy counters before benchmark (if available)
    energy_before_mj: List[int] = []
    if nvml_handles:
        torch.cuda.synchronize()
        for h in nvml_handles:
            energy_before_mj.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))

    try:
        if tier == 1:
            single_us = replay_tier1(kernel, device, runs, warmup)
        elif tier == 2:
            single_us = replay_tier2(kernel, device, runs, warmup)
        elif tier == 3:
            single_us = replay_tier3(kernel, device, runs, warmup, shape_map)
        elif tier == 4:
            single_us = replay_tier4(kernel, device, runs, warmup)
        else:
            single_us = 0.0
    except Exception as e:
        print(f"  WARNING: kernel failed ({e}): {kernel['name'][:80]}", file=sys.stderr)
        single_us = 0.0

    # Read NVML energy counters after benchmark
    energy_after_mj: List[int] = []
    if nvml_handles:
        torch.cuda.synchronize()
        for h in nvml_handles:
            energy_after_mj.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))

    end_wall = time.time()
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Mirror C++ aggregation.cpp: skip kernels where avg_single_time_us < 0
    # (these are excluded APIs like cudaFuncSetAttribute that return -1.0).
    if single_us < 0.0:
        return None

    invocation_count = kernel.get("count", 1)
    total_us = single_us * invocation_count

    # Compute NVML energy
    gpu_energy_mj = 0.0  # energy for this kernel's device
    system_energy_mj = 0.0  # sum across all GPUs
    if energy_before_mj and energy_after_mj:
        for i, (before, after) in enumerate(zip(energy_before_mj, energy_after_mj)):
            delta = float(after - before)
            system_energy_mj += delta
            # Attribute GPU energy to the specific device this kernel ran on
            if i == (device.index or 0):
                gpu_energy_mj = delta

    # Per-execution energy = total_energy / benchmark_runs
    gpu_energy_per_exec_mj = gpu_energy_mj / runs if runs > 0 else 0.0
    system_energy_per_exec_mj = system_energy_mj / runs if runs > 0 else 0.0

    return {
        "name": kernel["name"],
        "tier": tier,
        "invocation_count": invocation_count,
        "benchmark_runs": runs,
        "single_time_us": single_us,
        "total_time_us": total_us,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "wall_time_s": end_wall - start_wall,
        "gpu_energy_mj": gpu_energy_mj,
        "gpu_energy_per_exec_mj": gpu_energy_per_exec_mj,
        "system_energy_mj": system_energy_mj,
        "system_energy_per_exec_mj": system_energy_per_exec_mj,
        "has_metrics": bool(nvml_handles),
        "system_metrics": {},
    }


# ---------------------------------------------------------------------------
# Shape map: build from shape_log to give Tier 3 ops real tensor shapes
# ---------------------------------------------------------------------------


def build_shape_map(shape_log_path: Optional[str]) -> Dict[str, Tuple[int, int, int]]:
    """
    Build a map of feat_dim → (batch, seq_len, feat) from shape_log_rank0.jsonl.
    Used as a lookup table in replay_tier3 to find a representative shape for
    a given feature dimension inferred from the kernel's grid/block dimensions.

    The shape log records Linear layer I/O shapes; we aggregate by feature dimension
    and return the most common (batch, seq, feat) tuple for each feature size seen.
    This gives Tier 3 ops realistic tensor sizes rather than purely grid-based guesses.
    """
    if not shape_log_path or not os.path.exists(shape_log_path):
        return {}

    shapes_by_feat: Dict[int, List[Tuple[int, int, int]]] = {}
    with open(shape_log_path) as f:
        for line in f:
            entry = json.loads(line)
            s = entry.get("in_shape")
            if s and len(s) == 3:
                batch, seq, feat = s
                feat = max(feat, 1)
                shapes_by_feat.setdefault(feat, []).append((batch, seq, feat))

    # Return the most frequent shape per feature dim (key = str(feat))
    result: Dict[str, Tuple[int, int, int]] = {}
    for feat, shapes in shapes_by_feat.items():
        # Pick the most common (batch, seq) pair for this feature dim
        from collections import Counter

        most_common = Counter(shapes).most_common(1)[0][0]
        result[str(feat)] = most_common
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Python kernel replay benchmark for Vicuna-7B TP=2."
    )
    parser.add_argument(
        "--kernels",
        default="kernel_signatures.json",
        help="Path to kernel_signatures.json",
    )
    parser.add_argument(
        "--shape-log",
        default=None,
        help="Path to shape_log_rank0.jsonl for accurate Tier 3 shapes",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for isolated_kernels_timing.json",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override benchmark run count for all tiers",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override warmup count for all tiers",
    )
    parser.add_argument(
        "--nccl",
        action="store_true",
        help="Enable Tier 4 NCCL replay (requires torchrun --nproc_per_node=2)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Which tiers to benchmark (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--nvml",
        action="store_true",
        default=True,
        help="Enable NVML energy measurement (default: True if pynvml available)",
    )
    parser.add_argument(
        "--no-nvml",
        action="store_true",
        help="Disable NVML energy measurement",
    )
    args = parser.parse_args()

    # Distributed init for Tier 4 NCCL
    rank = 0
    if args.nccl:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda", args.device)

    if rank == 0:
        print("=" * 70)
        print("Python Kernel Replay Benchmark")
        print("=" * 70)
        print(f"Kernels:    {args.kernels}")
        print(f"Device:     {device} (default; compute kernels use signature.device)")
        print(f"Tiers:      {args.tiers}")
        print(f"NCCL mode:  {args.nccl}")
        if args.runs:
            print(f"Runs:       {args.runs} (override)")
        else:
            print(f"Runs:       {TIER_RUNS} (per tier)")

    # Initialize NVML for energy measurement
    nvml_handles: Optional[List] = None
    use_nvml = HAS_NVML and args.nvml and not args.no_nvml
    if use_nvml:
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]
            if rank == 0:
                print(f"NVML:       enabled ({gpu_count} GPUs)")
        except Exception as e:
            if rank == 0:
                print(f"NVML:       failed ({e}), energy measurement disabled")
            nvml_handles = None
    else:
        if rank == 0:
            reason = "not installed" if not HAS_NVML else "disabled via --no-nvml"
            print(f"NVML:       disabled ({reason})")

    if rank == 0:
        print("=" * 70)

    # Load kernel signatures
    if not os.path.exists(args.kernels):
        print(f"ERROR: kernels file not found: {args.kernels}", file=sys.stderr)
        return 1

    with open(args.kernels) as f:
        sigs = json.load(f)
    kernels = sigs["kernels"]

    # Build shape map from shape_log
    shape_map = build_shape_map(args.shape_log)
    if rank == 0 and shape_map:
        print(f"Shape map: {len(shape_map)} feature dims loaded from {args.shape_log}")

    # Filter tiers
    kernels_to_run = [k for k in kernels if k["tier"] in args.tiers]
    if rank == 0:
        print(
            f"\nBenchmarking {len(kernels_to_run)} kernels "
            f"(of {len(kernels)} total, tiers {args.tiers})..."
        )
        print()

    # --- Main benchmark loop ---
    results = []
    tier_totals = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    global_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    for i, kernel in enumerate(kernels_to_run):
        tier = kernel["tier"]

        # Skip Tier 4 if not in NCCL mode
        if tier == 4 and not args.nccl:
            if rank == 0:
                print(
                    f"  [{i + 1:3d}/{len(kernels_to_run)}] SKIP (Tier 4, no --nccl): "
                    f"{kernel['name'][:60]}"
                )
            continue

        if rank == 0:
            sig_dev = kernel.get("signature", {}).get("device")
            dev_label = f"cuda:{sig_dev}" if sig_dev is not None else str(device)
            print(
                f"  [{i + 1:3d}/{len(kernels_to_run)}] T{tier} "
                f"dev={dev_label}  count={kernel.get('count', 1):6d}  {kernel['name'][:60]}"
            )

        rec = benchmark_kernel(kernel, device, shape_map, args.runs, args.warmup, nvml_handles)

        # None means the kernel was excluded (e.g. cudaFuncSetAttribute returns -1.0).
        # Match C++ aggregation.cpp: skip from results and energy sum.
        if rec is None:
            if rank == 0:
                print(f"           EXCLUDED (single_time_us=-1.0, not counted)")
            continue

        results.append(rec)
        tier_totals[tier] = tier_totals.get(tier, 0.0) + rec["total_time_us"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        if rank == 0:
            energy_str = ""
            if nvml_handles:
                energy_str = f"  gpu_e={rec['gpu_energy_per_exec_mj']:.4f} mJ/exec"
            print(
                f"           single={rec['single_time_us']:.4f} us  "
                f"total={rec['total_time_us']:.2f} us  "
                f"wall={rec['wall_time_s']:.2f}s{energy_str}"
            )

    global_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # --- Summary ---
    predicted_total_us = sum(r["total_time_us"] for r in results)
    predicted_total_ms = predicted_total_us / 1000.0

    if rank == 0:
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        tier_names = {1: "CUDA Runtime", 2: "cuBLAS GEMM", 3: "libtorch", 4: "NCCL"}
        for t in [1, 2, 3, 4]:
            if tier_counts[t] > 0:
                pct = (
                    tier_totals[t] / predicted_total_us * 100
                    if predicted_total_us
                    else 0
                )
                print(
                    f"Tier {t} ({tier_names[t]:15s}): "
                    f"{tier_counts[t]:3d} kernels, "
                    f"{tier_totals[t] / 1000:.2f} ms ({pct:.1f}%)"
                )
        print(f"\nPredicted total (1 inference): {predicted_total_ms:.3f} ms")
        print("=" * 70)

        # --- Save output (same schema as C++ isolated_kernels_timing.json) ---
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "isolated_kernels_timing.json")

        output = {
            "start_timestamp": global_start,
            "end_timestamp": global_end,
            "num_runs": len(results),
            "predicted_total_us": predicted_total_us,
            "predicted_total_ms": predicted_total_ms,
            "tier1_count": tier_counts[1],
            "tier1_total_us": tier_totals[1],
            "tier2_count": tier_counts[2],
            "tier2_total_us": tier_totals[2],
            "tier3_count": tier_counts[3],
            "tier3_total_us": tier_totals[3],
            "tier4_count": tier_counts[4],
            "tier4_total_us": tier_totals[4],
            "kernels": results,
        }

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {out_path}")

    # Cleanup NVML
    if nvml_handles:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    if args.nccl and dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
