#!/usr/bin/env python3
"""
collect_ncu_metrics.py — Collect 21 regression-model NCU metrics for all compute
kernels in kernel_dataset_roofline.csv for one Vicuna model variant.

Uses the kernel-replay infrastructure (kernel_replay_benchmark.py) already present
in the repo: for each unique kernel in the CSV it generates a minimal Python driver
that replays the kernel once under ncu, then parses the CSV output.

Tier 1 (memcpy/memset), Tier 4 (NCCL), and cuda_api kernels are skipped (NaN row).

Usage:
  python scripts/collect_ncu_metrics.py \\
      --model_key 7b \\
      --dataset_dir results/dataset_20260327_205511 \\
      --existing_csv results/dataset_20260327_205511/kernel_dataset_roofline.csv \\
      --output_csv /tmp/ncu_7b.csv \\
      --device 0

  # Run all three models in parallel on separate GPUs:
  python scripts/collect_ncu_metrics.py --model_key 7b  --device 0 &
  python scripts/collect_ncu_metrics.py --model_key 13b --device 1 &
  python scripts/collect_ncu_metrics.py --model_key 33b --device 2 &
  wait
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

import pandas as pd

# ── NCU metric map (qualified NCU name → output column name) ─────────────────
METRICS_MAP: List[Tuple[str, str]] = [
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed",                "sm_throughput_pct"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed",              "dram_throughput_pct"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active",               "achieved_occupancy_pct"),
    ("sm__warps_active.avg.per_cycle_active",                           "warp_activity"),
    ("smsp__warps_eligible.avg.per_cycle_active",                       "warp_eligible"),
    ("smsp__issue_active.avg.per_cycle_active",                         "issue_activity"),
    ("smsp__inst_executed.avg.per_cycle_active",                        "ipc"),
    ("smsp__inst_executed.sum",                                         "inst_executed"),
    ("smsp__sass_thread_inst_executed_op_memory_pred_on.sum",           "mem_inst"),
    ("dram__bytes_read.sum",                                            "dram_read_bytes"),
    ("dram__bytes_write.sum",                                           "dram_write_bytes"),
    ("lts__t_sectors.avg.pct_of_peak_sustained_elapsed",                "l2_throughput_pct"),
    ("l1tex__data_bank_conflicts_pipe_lsu.sum",                         "shared_bank_conflicts"),
    ("smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",   "fma_utilization_pct"),
    ("smsp__average_warp_latency_per_inst_issued.ratio",                "avg_warp_latency"),
    ("smsp__warp_issue_stalled_membar_per_warp_active.pct",             "stall_mem_dep_pct"),
    ("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",    "stall_long_sb_pct"),
    ("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",   "stall_short_sb_pct"),
    ("smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct", "stall_math_pipe_pct"),
    ("smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",       "stall_mio_throttle_pct"),
    ("smsp__warp_issue_stalled_not_selected_per_warp_active.pct",       "stall_not_selected_pct"),
]

NCU_METRICS_STR = ",".join(ncu for ncu, _ in METRICS_MAP)
NCU_NAME_TO_SHORT = {ncu: short for ncu, short in METRICS_MAP}
SHORT_NAMES = [short for _, short in METRICS_MAP]
NAN_ROW = {short: float("nan") for short in SHORT_NAMES}

NCU_SECTIONS = [
    "SpeedOfLight_HierarchicalSingleRooflineChart",
    "MemoryWorkloadAnalysis",
]

MODEL_BY_KEY = {
    "7b":  "lmsys/vicuna-7b-v1.5",
    "13b": "lmsys/vicuna-13b-v1.5",
    "33b": "lmsys/vicuna-33b-v1.3",
}

# Tiers / operations that produce no GPU kernel → skip NCU, return NaN
SKIP_TIERS = {1, 4}
SKIP_OPS   = {"cuda_api"}


# ── Driver generation ─────────────────────────────────────────────────────────

def _generate_kernel_driver(kernel: Dict, shape_log_path: Optional[str], device_idx: int) -> str:
    """
    Generate a minimal Python script that fires one kernel under NCU profiling.
    Mirrors ncu_roofline_profiler.py::generate_kernel_driver exactly, so the
    same replay infrastructure is used.
    """
    tier = kernel["tier"]
    kernel_json = json.dumps(kernel)
    shape_log_arg = f'"{shape_log_path}"' if shape_log_path else "None"

    return f'''#!/usr/bin/env python3
"""Auto-generated NCU driver for: {kernel["name"][:60]}"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vicuna"))

import torch
import torch.cuda.profiler as profiler

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

device = torch.device("cuda", 0)  # CUDA_VISIBLE_DEVICES restricts to target GPU
torch.cuda.set_device(device)

kernel = json.loads({repr(kernel_json)})

from kernel_replay_benchmark import (
    replay_tier1, replay_tier2, replay_tier3, build_shape_map
)

shape_map = build_shape_map({shape_log_arg})

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


# ── NCU CSV parsing ───────────────────────────────────────────────────────────

def _parse_ncu_csv(stdout: str) -> Dict[str, Dict[str, float]]:
    """
    Parse ncu --csv output (long format) into {kernel_name: {metric_name: value}}.
    NCU emits ==PROF== lines on stderr; stdout is the CSV.  Multiple invocations
    of the same kernel produce multiple rows — callers should average across them.
    """
    results: Dict[str, Dict[str, float]] = {}

    # Strip ==PROF== prefix lines that sometimes bleed into stdout
    csv_lines = [l for l in stdout.splitlines()
                 if l.startswith('"') or (l and l[0].isdigit())]
    if not csv_lines:
        return results

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    invoc_counts: Dict[str, int] = {}

    for row in reader:
        kname  = row.get("Kernel Name", "").strip()
        metric = row.get("Metric Name", "").strip()
        val_s  = row.get("Metric Value", "").strip()

        if not kname or not metric or metric not in NCU_NAME_TO_SHORT:
            continue

        try:
            value = float(val_s.replace(",", ""))
        except ValueError:
            value = float("nan")

        if kname not in results:
            results[kname] = {}
            invoc_counts[kname] = 0

        short = NCU_NAME_TO_SHORT[metric]
        # Running mean accumulation
        n = invoc_counts[kname]
        prev = results[kname].get(short, 0.0)
        results[kname][short] = (prev * n + value) / (n + 1)

    # Finalize invocation counts (increment once per (kernel, metric) group)
    # The simple mean above handles it; just return the accumulated dict.
    return results


def _best_kernel_match(ncu_results: Dict[str, Dict[str, float]], target_name: str) -> Optional[Dict[str, float]]:
    """
    Pick the NCU-captured kernel that best matches target_name.
    Strategy mirrors ncu_roofline_profiler.py: exact > substring > partial.
    Falls back to the entry with the most non-zero metrics.
    """
    if not ncu_results:
        return None

    # Exact match
    if target_name in ncu_results:
        return ncu_results[target_name]

    stem = target_name.split("(")[0].strip()
    best_name = None
    best_score = -1

    for ncu_kname, metrics in ncu_results.items():
        score = 0
        if stem in ncu_kname or ncu_kname in target_name:
            score = 100
        elif any(p in ncu_kname for p in target_name.split("::")[-2:] if len(p) > 5):
            score = 50

        # Prefer kernels with real work
        non_zero = sum(1 for v in metrics.values() if v and v == v and v != 0)
        score += non_zero

        if score > best_score:
            best_score = score
            best_name = ncu_kname

    if best_name and best_score > 0:
        return ncu_results[best_name]

    # Last resort: entry with most non-zero values
    return max(ncu_results.values(), key=lambda m: sum(1 for v in m.values() if v and v == v and v != 0))


# ── NCU runner ────────────────────────────────────────────────────────────────

def _run_ncu_for_kernel(
    kernel: Dict,
    shape_log_path: Optional[str],
    device_idx: int,
    ncu_bin: str,
    scripts_dir: str,
) -> Dict[str, float]:
    """
    Run ncu on the generated driver and return {short_name: value} dict.
    Returns NAN_ROW on failure.
    """
    driver_code = _generate_kernel_driver(kernel, shape_log_path, device_idx)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="ncu_regr_driver_",
        dir=scripts_dir, delete=False,
    ) as f:
        f.write(driver_code)
        driver_path = f.name

    try:
        section_flags: List[str] = []
        for s in NCU_SECTIONS:
            section_flags += ["--section", s]

        cmd = [
            ncu_bin,
            "--csv",
            "--profile-from-start", "off",
            "--kernel-name-base", "demangled",
            "--metrics", NCU_METRICS_STR,
            *section_flags,
            "--target-processes", "all",
            sys.executable, driver_path,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_idx)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180, env=env,
        )

        if result.returncode != 0 and not result.stdout.strip():
            print(f"    ncu error (rc={result.returncode}): "
                  f"{result.stderr[-300:].strip()}", file=sys.stderr)
            return dict(NAN_ROW)

        ncu_data = _parse_ncu_csv(result.stdout)
        if not ncu_data:
            return dict(NAN_ROW)

        best = _best_kernel_match(ncu_data, kernel["name"])
        if best is None:
            return dict(NAN_ROW)

        # Fill any missing metrics with NaN
        row = dict(NAN_ROW)
        row.update(best)
        return row

    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: {kernel['name'][:60]}", file=sys.stderr)
        return dict(NAN_ROW)
    except Exception as exc:
        print(f"    ERROR: {exc}", file=sys.stderr)
        return dict(NAN_ROW)
    finally:
        try:
            os.unlink(driver_path)
        except OSError:
            pass


# ── Static GPU features ───────────────────────────────────────────────────────

def _get_gpu_statics(device_idx: int) -> Dict[str, Any]:
    """Query static GPU properties for the target device."""
    import torch
    props = torch.cuda.get_device_properties(device_idx)
    sm_count = props.multi_processor_count

    # Clock speeds and bandwidth via nvidia-smi
    try:
        smi = subprocess.run(
            ["nvidia-smi",
             f"--id={device_idx}",
             "--query-gpu=clocks.max.sm,clocks.max.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        parts = [p.strip() for p in smi.stdout.strip().split(",")]
        gpu_clock_mhz     = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        gpu_mem_clock_mhz = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    except Exception:
        gpu_clock_mhz, gpu_mem_clock_mhz = 0, 0

    # Theoretical peak bandwidth: 2 * mem_clock_Hz * (bus_width_bits / 8) bytes/s
    try:
        bus_width = props.memory_bus_width  # bits
        gpu_mem_bandwidth_gbs = round(
            2.0 * gpu_mem_clock_mhz * 1e6 * (bus_width / 8) / 1e9, 1
        )
    except Exception:
        gpu_mem_bandwidth_gbs = 0.0

    return {
        "gpu_sm_count":          sm_count,
        "gpu_clock_mhz":         gpu_clock_mhz,
        "gpu_mem_clock_mhz":     gpu_mem_clock_mhz,
        "gpu_mem_bandwidth_gbs": gpu_mem_bandwidth_gbs,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect 21 NCU regression metrics for one Vicuna model variant."
    )
    parser.add_argument(
        "--model_key", required=True, choices=list(MODEL_BY_KEY),
        help="Model size key: 7b | 13b | 33b"
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Path to the dataset directory (contains 7b/, 13b/, 33b/ subdirs)"
    )
    parser.add_argument(
        "--existing_csv", required=True,
        help="Path to kernel_dataset_roofline.csv"
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Output CSV path for collected NCU metrics"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device index (default: 0)"
    )
    parser.add_argument(
        "--ncu-bin", default="ncu",
        help="Path to ncu binary (default: ncu in PATH)"
    )
    args = parser.parse_args()

    model_name = MODEL_BY_KEY[args.model_key]
    model_dir  = os.path.join(args.dataset_dir, args.model_key)
    sigs_path  = os.path.join(model_dir, "kernel_signatures.json")
    shape_log  = os.path.join(model_dir, "shape_log_rank0.jsonl")
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isfile(sigs_path):
        print(f"ERROR: kernel_signatures.json not found: {sigs_path}", file=sys.stderr)
        return 1

    # Load kernel signatures
    with open(sigs_path) as f:
        raw = json.load(f)
    all_sigs = raw if isinstance(raw, list) else raw.get("kernels", [])
    sig_by_name: Dict[str, Dict] = {k["name"]: k for k in all_sigs}

    # Load existing CSV and get unique kernel names for this model
    df = pd.read_csv(args.existing_csv)
    model_rows = df[df["model"] == model_name]
    if model_rows.empty:
        print(f"ERROR: no rows for model '{model_name}' in {args.existing_csv}", file=sys.stderr)
        return 1
    unique_kernels: List[str] = sorted(model_rows["kernel_name"].unique())

    # GPU static features
    gpu_statics = _get_gpu_statics(args.device)

    print("=" * 70)
    print(f"NCU Metric Collection — {args.model_key}  ({model_name})")
    print("=" * 70)
    print(f"Kernels to process : {len(unique_kernels)}")
    print(f"Device             : cuda:{args.device}")
    print(f"GPU SM count       : {gpu_statics['gpu_sm_count']}")
    print(f"GPU clock          : {gpu_statics['gpu_clock_mhz']} MHz")
    print(f"Mem clock          : {gpu_statics['gpu_mem_clock_mhz']} MHz")
    print(f"Mem bandwidth      : {gpu_statics['gpu_mem_bandwidth_gbs']} GB/s")
    print(f"Output             : {args.output_csv}")
    print("=" * 70)

    records: List[Dict] = []
    n_profiled = 0
    n_skipped  = 0
    n_failed   = 0

    for idx, kname in enumerate(unique_kernels):
        sig = sig_by_name.get(kname)
        tier = sig.get("tier", 0) if sig else 0
        op   = (sig.get("params", {}).get("operation", "") if sig else "")

        skip = (tier in SKIP_TIERS) or (op in SKIP_OPS) or (sig is None)
        label = f"[{idx+1:3d}/{len(unique_kernels)}]"
        short_name = kname[:60] + "…" if len(kname) > 60 else kname

        if skip:
            reason = "no sig" if sig is None else f"tier{tier}/{op}"
            print(f"  {label} SKIP ({reason:10s})  {short_name}")
            metrics_row = dict(NAN_ROW)
            n_skipped += 1
        else:
            print(f"  {label} NCU  (tier{tier:d})        {short_name}")
            metrics_row = _run_ncu_for_kernel(
                sig, shape_log if os.path.isfile(shape_log) else None,
                args.device, args.ncu_bin, scripts_dir,
            )
            if all(v != v for v in metrics_row.values()):  # all NaN
                n_failed += 1
            else:
                n_profiled += 1

        record: Dict[str, Any] = {
            "model":       model_name,
            "kernel_name": kname,
            **metrics_row,
            **gpu_statics,
        }
        records.append(record)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    out_cols = (
        ["model", "kernel_name"]
        + SHORT_NAMES
        + ["gpu_sm_count", "gpu_clock_mhz", "gpu_mem_clock_mhz", "gpu_mem_bandwidth_gbs"]
    )
    pd.DataFrame(records)[out_cols].to_csv(args.output_csv, index=False)

    print()
    print("=" * 70)
    print(f"  Profiled : {n_profiled}")
    print(f"  Skipped  : {n_skipped}  (tier1/4/cuda_api — NaN values)")
    print(f"  Failed   : {n_failed}  (NCU error — NaN values)")
    print(f"  Output   : {args.output_csv}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
