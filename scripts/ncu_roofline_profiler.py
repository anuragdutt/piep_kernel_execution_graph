#!/usr/bin/env python3
"""
NCU Roofline Profiler — Collect hardware performance counters for kernel classification.

For each unique GPU-compute kernel (Tiers 2 & 3) in kernel_signatures.json,
this script:
  1. Generates a minimal Python driver that fires the kernel once
  2. Runs it under `ncu --csv --metrics ...` to collect hardware counters
  3. Parses the CSV output to extract FLOPs, DRAM bytes, and kernel duration
  4. Computes arithmetic intensity (FLOP/byte) and classifies as
     compute_bound or memory_bound against the device roofline

Usage:
  python scripts/ncu_roofline_profiler.py \
      --kernels results/dataset_.../7b/kernel_signatures.json \
      --shape-log results/dataset_.../7b/shape_log_rank0.jsonl \
      --output results/dataset_.../7b/roofline_metrics.json

RTX A6000 (GA102) specs (tensor cores disabled):
  FP16 CUDA-core peak: 77.4 TFLOPS
  DRAM bandwidth:      768 GB/s
  Ridge point AI:      ~100.8 FLOP/byte
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# ── Device roofline ceilings ──────────────────────────────────────────────────
# RTX A6000 (GA102 Ampere) — Empirical Boundaries
PEAK_FP16_TFLOPS = 108.8       # TFLOPS (Empirically measured max compute)
PEAK_BW_GBS = 590.3            # GB/s (Empirically measured max DRAM bandwidth)
PEAK_FP16_FLOPS = PEAK_FP16_TFLOPS * 1e12   # FLOP/s
PEAK_BW_BYTES = PEAK_BW_GBS * 1e9           # bytes/s
RIDGE_POINT_AI = PEAK_FP16_FLOPS / PEAK_BW_BYTES

# ── NCU metrics to collect ────────────────────────────────────────────────────
NCU_METRICS = [
    # FP32 FLOP instruction counters
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    # FP16 FLOP instruction counters
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    # Tensor core (should be 0 since we disabled them)
    "sm__inst_executed_pipe_tensor_op_hmma.sum",
    # Generic FP16 instruction counter (fallback)
    "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
    # DRAM bytes
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    # L1 cache bytes (for hierarchical roofline)
    "l1tex__t_bytes.sum",
    # Kernel duration
    "gpu__time_duration.sum",
]

# Tiers that are pure memory ops (no FLOPs to measure → auto-classify)
MEMORY_OP_TIERS = {1}

# CUDA APIs that ncu cannot profile (host-side only)
SKIP_OPERATIONS = {"cuda_api"}


def parse_ncu_csv(csv_text: str) -> Dict[str, Dict[str, float]]:
    """Parse ncu --csv output into {kernel_name: {metric_name: value}}."""
    results: Dict[str, Dict[str, float]] = {}

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        kernel = row.get("Kernel Name", "")
        metric = row.get("Metric Name", "")
        value_str = row.get("Metric Value", "0")

        if not kernel or not metric:
            continue

        # ncu uses commas in numbers like "1,234,567"
        try:
            value = float(value_str.replace(",", ""))
        except ValueError:
            value = 0.0

        if kernel not in results:
            results[kernel] = {}
        results[kernel][metric] = value

    return results


def compute_roofline_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Compute roofline classification from raw ncu hardware counters."""

    # ── FP32 FLOPs ────────────────────────────────────────────────────────
    fp32_fadd = metrics.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0)
    fp32_fmul = metrics.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0)
    fp32_ffma = metrics.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)
    fp32_flops = fp32_fadd + fp32_fmul + 2 * fp32_ffma  # FMA = 2 FLOPs

    # ── FP16 FLOPs ────────────────────────────────────────────────────────
    fp16_hadd = metrics.get("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum", 0)
    fp16_hmul = metrics.get("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum", 0)
    fp16_hfma = metrics.get("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", 0)
    fp16_flops = fp16_hadd + fp16_hmul + 2 * fp16_hfma

    # FP16 instructions can pack 2 ops per thread instruction on some architectures
    # Use the generic FP16 counter as a cross-check
    fp16_generic = metrics.get("sm__sass_thread_inst_executed_op_fp16_pred_on.sum", 0)

    # ── Tensor core FLOPs ─────────────────────────────────────────────────
    tc_hmma = metrics.get("sm__inst_executed_pipe_tensor_op_hmma.sum", 0)
    # Each HMMA warp instruction performs a certain number of FLOPs depending on
    # the MMA shape. For m16n8k8 FP16: each instruction = 16*8*8*2 = 2048 FLOPs
    # per warp instruction. We flag if TC is used unexpectedly.
    tc_flops = tc_hmma * 2048  # conservative estimate

    total_flops = fp32_flops + fp16_flops + tc_flops

    # If SASS counters are 0 but generic FP16 counter is nonzero,
    # cuBLAS may be using warp-level intrinsics that bypass per-thread counters.
    # Use the generic counter as a fallback (each is 2 ops for packed FP16).
    if total_flops == 0 and fp16_generic > 0:
        total_flops = fp16_generic * 2  # packed half2 operations

    # ── DRAM bytes ────────────────────────────────────────────────────────
    dram_bytes = metrics.get("dram__bytes.sum", 0)
    dram_read = metrics.get("dram__bytes_read.sum", 0)
    dram_write = metrics.get("dram__bytes_write.sum", 0)

    # L1 cache bytes
    l1_bytes = metrics.get("l1tex__t_bytes.sum", 0)

    # ── Duration ──────────────────────────────────────────────────────────
    duration_ns = metrics.get("gpu__time_duration.sum", 0)
    duration_s = duration_ns / 1e9 if duration_ns > 0 else 1e-12

    # ── Derived metrics ───────────────────────────────────────────────────
    arithmetic_intensity = total_flops / dram_bytes if dram_bytes > 0 else float("inf")
    achieved_gflops = total_flops / duration_s / 1e9 if duration_s > 0 else 0
    achieved_bw_gbs = dram_bytes / duration_s / 1e9 if duration_s > 0 else 0

    pct_peak_compute = (achieved_gflops / (PEAK_FP16_TFLOPS * 1e3)) * 100 if PEAK_FP16_TFLOPS > 0 else 0
    pct_peak_bw = (achieved_bw_gbs / PEAK_BW_GBS) * 100 if PEAK_BW_GBS > 0 else 0

    if arithmetic_intensity > RIDGE_POINT_AI:
        boundedness = "compute_bound"
    else:
        boundedness = "memory_bound"

    return {
        "fp32_flops": fp32_flops,
        "fp16_flops": fp16_flops,
        "tc_flops": tc_flops,
        "total_flops": total_flops,
        "dram_bytes": dram_bytes,
        "dram_read_bytes": dram_read,
        "dram_write_bytes": dram_write,
        "l1_bytes": l1_bytes,
        "duration_ns": duration_ns,
        "arithmetic_intensity": arithmetic_intensity,
        "achieved_gflops": achieved_gflops,
        "achieved_bw_gbs": achieved_bw_gbs,
        "pct_peak_compute": pct_peak_compute,
        "pct_peak_bw": pct_peak_bw,
        "boundedness": boundedness,
        "tc_hmma_instructions": tc_hmma,
        "tc_unexpected": tc_hmma > 0,  # flag: we disabled tensor cores
    }


def generate_kernel_driver(kernel: Dict, shape_log_path: Optional[str], device_idx: int) -> str:
    """Generate a minimal Python script that fires one kernel under ncu profiling."""

    tier = kernel["tier"]
    kernel_json = json.dumps(kernel)
    shape_log_arg = f'"{shape_log_path}"' if shape_log_path else "None"

    script = f'''#!/usr/bin/env python3
"""Auto-generated ncu driver for kernel: {kernel["name"][:60]}"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vicuna"))
import torch
import torch.cuda.profiler as profiler

# Disable tensor cores
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# Physical device is isolated via CUDA_VISIBLE_DEVICES, so logical device is always 0
device = torch.device("cuda", 0)
torch.cuda.set_device(device)

kernel = json.loads({repr(kernel_json)})

# Import replay functions
from kernel_replay_benchmark import (
    replay_tier1, replay_tier2, replay_tier3, build_shape_map
)

shape_map = build_shape_map({shape_log_arg})

# Warmup (outside profiler)
torch.cuda.synchronize(device)
try:
    if {tier} == 1:
        replay_tier1(kernel, device, 2, 1)
    elif {tier} == 2:
        replay_tier2(kernel, device, 2, 1)
    elif {tier} == 3:
        replay_tier3(kernel, device, 2, 1, shape_map)
except Exception as e:
    print(f"Warmup failed: {{e}}", file=sys.stderr)
    sys.exit(1)

torch.cuda.synchronize(device)

# Profile: fire exactly once under ncu
profiler.start()
if {tier} == 1:
    replay_tier1(kernel, device, 1, 0)
elif {tier} == 2:
    replay_tier2(kernel, device, 1, 0)
elif {tier} == 3:
    replay_tier3(kernel, device, 1, 0, shape_map)
torch.cuda.synchronize(device)
profiler.stop()
'''
    return script


def run_ncu_for_kernel(
    kernel: Dict,
    shape_log_path: Optional[str],
    device_idx: int,
    ncu_bin: str = "ncu",
) -> Optional[Dict[str, Dict[str, float]]]:
    """Run ncu on a generated kernel driver and return parsed metrics."""

    # Write temporary driver script
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="ncu_driver_",
        dir=os.path.join(os.path.dirname(__file__)),
        delete=False,
    ) as f:
        driver_code = generate_kernel_driver(kernel, shape_log_path, device_idx)
        f.write(driver_code)
        driver_path = f.name

    try:
        metrics_str = ",".join(NCU_METRICS)
        cmd = [
            ncu_bin,
            "--csv",
            "--profile-from-start", "off",
            "--metrics", metrics_str,
            "--target-processes", "all",
            sys.executable, driver_path,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_idx)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per kernel
            env=env,
        )

        if result.returncode != 0:
            # Print stderr for debugging but don't fail
            stderr_snippet = result.stderr[-500:] if result.stderr else "(empty)"
            print(f"    ncu warning (rc={result.returncode}): {stderr_snippet[:200]}",
                  file=sys.stderr)

        # Parse CSV from stdout
        csv_lines = []
        for line in result.stdout.split("\n"):
            if line.startswith('"') or line.startswith("ID,") or line.startswith('"ID"'):
                csv_lines.append(line)

        if not csv_lines:
            return None

        csv_text = "\n".join(csv_lines)
        return parse_ncu_csv(csv_text)

    except subprocess.TimeoutExpired:
        print(f"    ncu TIMEOUT for kernel: {kernel['name'][:60]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"    ncu ERROR: {e}", file=sys.stderr)
        return None
    finally:
        try:
            os.unlink(driver_path)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NCU Roofline Profiler — collect hardware counters for kernel classification"
    )
    parser.add_argument(
        "--kernels", required=True,
        help="Path to kernel_signatures.json (same as replay benchmark input)"
    )
    parser.add_argument(
        "--shape-log", default=None,
        help="Path to shape_log_rank0.jsonl for Tier 3 shape inference"
    )
    parser.add_argument(
        "--output", default="roofline_metrics.json",
        help="Output JSON file (default: roofline_metrics.json)"
    )
    parser.add_argument(
        "--tiers", nargs="+", type=int, default=[2, 3],
        help="Tiers to profile with ncu (default: 2 3)"
    )
    parser.add_argument(
        "--ncu-bin", default="ncu",
        help="Path to ncu binary (default: ncu in PATH)"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device index (default: 0)"
    )
    args = parser.parse_args()

    # Load kernel signatures
    with open(args.kernels) as f:
        all_kernels = json.load(f)

    if isinstance(all_kernels, dict) and "kernels" in all_kernels:
        all_kernels = all_kernels["kernels"]

    # Filter to requested tiers
    ncu_tiers = set(args.tiers)
    ncu_kernels = []
    skip_kernels = []
    for k in all_kernels:
        tier = k.get("tier", 0)
        operation = k.get("params", {}).get("operation", "")

        if tier in MEMORY_OP_TIERS:
            skip_kernels.append(k)
        elif operation in SKIP_OPERATIONS:
            skip_kernels.append(k)
        elif tier in ncu_tiers:
            ncu_kernels.append(k)
        else:
            skip_kernels.append(k)

    print("=" * 70)
    print("NCU Roofline Profiler")
    print("=" * 70)
    print(f"Kernels file:    {args.kernels}")
    print(f"Total kernels:   {len(all_kernels)}")
    print(f"To profile:      {len(ncu_kernels)} (Tiers {sorted(ncu_tiers)})")
    print(f"Auto-classified: {len(skip_kernels)} (memory ops / CUDA APIs)")
    print(f"Device:          cuda:{args.device}")
    print(f"Ridge point AI:  {RIDGE_POINT_AI:.1f} FLOP/byte")
    print("=" * 70)

    results = []

    # Auto-classify memory ops / CUDA APIs
    for k in skip_kernels:
        tier = k.get("tier", 0)
        operation = k.get("params", {}).get("operation", "")
        count = k.get("count", 1)

        if tier in MEMORY_OP_TIERS:
            boundedness = "memory_op"
        elif operation in SKIP_OPERATIONS:
            boundedness = "host_api"
        else:
            boundedness = "skipped"

        results.append({
            "kernel_name": k["name"],
            "tier": tier,
            "invocation_count": count,
            "boundedness": boundedness,
            "arithmetic_intensity": 0.0,
            "achieved_gflops": 0.0,
            "achieved_bw_gbs": 0.0,
            "pct_peak_compute": 0.0,
            "pct_peak_bw": 0.0,
            "total_flops": 0,
            "dram_bytes": 0,
            "duration_ns": 0,
            "ncu_profiled": False,
        })

    # Profile each kernel with ncu
    for i, k in enumerate(ncu_kernels):
        name = k["name"]
        tier = k.get("tier", 0)
        count = k.get("count", 1)
        operation = k.get("params", {}).get("operation", "unknown")

        # Resolve device from kernel signature
        sig_device = k.get("signature", {}).get("device", args.device)
        device_idx = int(sig_device) if sig_device is not None else args.device

        short_name = name[:70] + "..." if len(name) > 70 else name
        print(f"\n  [{i+1:3d}/{len(ncu_kernels)}] T{tier} dev=cuda:{device_idx}  {short_name}")

        ncu_result = run_ncu_for_kernel(k, args.shape_log, device_idx, args.ncu_bin)

        if ncu_result is None or len(ncu_result) == 0:
            print(f"           ⚠ No ncu data — marking as memory_bound (fallback)")
            results.append({
                "kernel_name": name,
                "tier": tier,
                "invocation_count": count,
                "boundedness": "memory_bound",
                "arithmetic_intensity": 0.0,
                "achieved_gflops": 0.0,
                "achieved_bw_gbs": 0.0,
                "pct_peak_compute": 0.0,
                "pct_peak_bw": 0.0,
                "total_flops": 0,
                "dram_bytes": 0,
                "duration_ns": 0,
                "ncu_profiled": False,
            })
            continue

        # ncu may capture multiple kernels (warmup leak, PyTorch internals).
        # Find the one that best matches our target kernel name.
        best_kernel = None
        best_match_score = -1
        for ncu_kernel_name, ncu_metrics in ncu_result.items():
            # Simple substring matching — our kernel name is a substring of ncu's demangled name
            score = 0
            # Check if the kernel name (possibly truncated) appears in the ncu output
            short_target = name.split("(")[0].strip()  # strip template args for matching
            if short_target in ncu_kernel_name or ncu_kernel_name in name:
                score = 100
            elif any(part in ncu_kernel_name for part in name.split("::")[-2:] if len(part) > 5):
                score = 50

            # Prefer kernels with real work (nonzero bytes or flops)
            dram = ncu_metrics.get("dram__bytes.sum", 0)
            dur = ncu_metrics.get("gpu__time_duration.sum", 0)
            if dram > 0 or dur > 100:
                score += 10

            if score > best_match_score:
                best_match_score = score
                best_kernel = ncu_metrics

        if best_kernel is None:
            # Fall back to first kernel with largest duration
            best_kernel = max(ncu_result.values(),
                              key=lambda m: m.get("gpu__time_duration.sum", 0))

        roofline = compute_roofline_metrics(best_kernel)

        print(f"           AI={roofline['arithmetic_intensity']:.2f} FLOP/byte  "
              f"GFLOP/s={roofline['achieved_gflops']:.1f}  "
              f"BW={roofline['achieved_bw_gbs']:.1f} GB/s  "
              f"→ {roofline['boundedness'].upper()}"
              f"{'  ⚠TC!' if roofline['tc_unexpected'] else ''}")

        results.append({
            "kernel_name": name,
            "tier": tier,
            "invocation_count": count,
            "ncu_profiled": True,
            **roofline,
        })

    # Write results
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "device_specs": {
                "name": "NVIDIA RTX A6000",
                "arch": "GA102 (Ampere)",
                "peak_fp16_tflops": PEAK_FP16_TFLOPS,
                "peak_bw_gbs": PEAK_BW_GBS,
                "ridge_point_ai": RIDGE_POINT_AI,
                "tensor_cores": "disabled",
            },
            "kernels": results,
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("ROOFLINE SUMMARY")
    print("=" * 70)
    profiled = [r for r in results if r.get("ncu_profiled")]
    auto = [r for r in results if not r.get("ncu_profiled")]
    compute_bound = [r for r in profiled if r["boundedness"] == "compute_bound"]
    memory_bound = [r for r in profiled if r["boundedness"] == "memory_bound"]
    tc_warn = [r for r in profiled if r.get("tc_unexpected")]

    print(f"  NCU-profiled:     {len(profiled)} kernels")
    print(f"  Auto-classified:  {len(auto)} kernels")
    print(f"  Compute-bound:    {len(compute_bound)}")
    print(f"  Memory-bound:     {len(memory_bound)}")
    if tc_warn:
        print(f"  ⚠ Tensor core usage detected in {len(tc_warn)} kernels (unexpected!)")
    print(f"\nOutput: {output_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
