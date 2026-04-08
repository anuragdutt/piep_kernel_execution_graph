#!/usr/bin/env python3
"""
verify_metrics.py — Verify all 21 required NCU metrics are collectable on this device.

`ncu --query-metrics` only lists base metric names, not qualified forms such as
`.avg.pct_of_peak_sustained_elapsed`.  This script runs a live NCU probe on a
trivial CUDA kernel and checks that every qualified metric name in the regression
model's required set actually appears in the output.

Usage:
  python scripts/verify_metrics.py
  python scripts/verify_metrics.py --ncu-bin /usr/local/cuda/bin/ncu
  python scripts/verify_metrics.py --device 1
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import tempfile
import textwrap
from typing import Dict, List, Optional, Set, Tuple

# ── Required metrics (ncu qualified name → short output column name) ──────────
REQUIRED_METRICS: List[Tuple[str, str]] = [
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

# Probe kernel: tiny fp16 matmul, brackets the profiled region with profiler.start/stop
_PROBE_SCRIPT = textwrap.dedent("""\
    import torch
    import torch.cuda.profiler as profiler

    device = torch.device("cuda:{device}")
    a = torch.randn(512, 512, dtype=torch.float16, device=device)
    b = torch.randn(512, 512, dtype=torch.float16, device=device)
    # Warmup (not profiled)
    torch.cuda.synchronize(device)
    _ = torch.mm(a, b)
    torch.cuda.synchronize(device)
    # Profiled region
    profiler.start()
    c = torch.mm(a, b)
    torch.cuda.synchronize(device)
    profiler.stop()
""")


def _write_probe_script(device: int) -> str:
    """Write the probe driver to a temp file and return its path."""
    script = _PROBE_SCRIPT.format(device=device)
    fd, path = tempfile.mkstemp(suffix=".py", prefix="ncu_verify_probe_")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    return path


def _parse_ncu_csv(text: str) -> Set[str]:
    """Return the set of 'Metric Name' values found in ncu --csv output."""
    found: Set[str] = set()
    # NCU prepends ==PROF== lines; skip them and find the header line.
    csv_lines = [l for l in text.splitlines()
                 if l.startswith('"') or (l and l[0].isdigit())]
    if not csv_lines:
        return found
    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    for row in reader:
        name = row.get("Metric Name", "").strip()
        if name:
            found.add(name)
    return found


def probe_metrics(
    ncu_bin: str,
    device: int,
    python_bin: str,
) -> Optional[Set[str]]:
    """Run a live NCU probe and return the set of metric names that were collected."""
    probe_path = _write_probe_script(device)
    try:
        metrics_str = ",".join(m for m, _ in REQUIRED_METRICS)
        cmd = [
            ncu_bin,
            "--csv",
            "--profile-from-start", "off",
            "--kernel-name-base", "demangled",
            "--metrics", metrics_str,
            "--target-processes", "all",
            python_bin, probe_path,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env,
        )

        if result.returncode != 0 and not result.stdout.strip():
            print(f"ERROR: ncu exited {result.returncode} with no output.")
            if result.stderr:
                print(f"  stderr: {result.stderr[:400]}")
            return None

        return _parse_ncu_csv(result.stdout)

    except FileNotFoundError:
        print(f"ERROR: ncu binary not found at '{ncu_bin}'.")
        print("  Install NVIDIA Nsight Compute or pass --ncu-bin <path>.")
        return None
    except subprocess.TimeoutExpired:
        print("ERROR: NCU probe timed out (120 s).")
        return None
    finally:
        try:
            os.unlink(probe_path)
        except OSError:
            pass


def check_sections(ncu_bin: str) -> Dict[str, bool]:
    """Check whether the two required section names are listed by ncu --list-sections."""
    required = {
        "SpeedOfLight_HierarchicalSingleRooflineChart",
        "MemoryWorkloadAnalysis",
    }
    result: Dict[str, bool] = {s: False for s in required}
    try:
        proc = subprocess.run(
            [ncu_bin, "--list-sections"],
            capture_output=True, text=True, timeout=30,
        )
        for line in proc.stdout.splitlines():
            for s in required:
                if s in line:
                    result[s] = True
    except Exception:
        pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all 21 regression-model NCU metrics are collectable via a "
            "live probe kernel.  Also checks the two --section names are available."
        )
    )
    parser.add_argument(
        "--ncu-bin", default="ncu",
        help="Path to ncu binary (default: ncu in PATH)"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device index to probe (default: 0)"
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter for the probe kernel (default: current interpreter)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("NCU Metric Verification  (live probe)")
    print("=" * 70)
    print(f"ncu binary  : {args.ncu_bin}")
    print(f"device      : {args.device}")
    print(f"python      : {args.python}")
    print()
    print("Running probe kernel (tiny fp16 matmul) under ncu…")

    collected = probe_metrics(args.ncu_bin, args.device, args.python)
    if collected is None:
        return 1

    if not collected:
        print("ERROR: NCU ran but returned no metrics.  Check ncu output format.")
        return 1

    print(f"Collected {len(collected)} distinct metric names from probe.\n")

    # ── Per-metric status ─────────────────────────────────────────────────────
    ok: List[str] = []
    missing: List[str] = []

    col_w = max(len(n) for n, _ in REQUIRED_METRICS)
    print(f"  {'STATUS':<8}  {'SHORT NAME':<28}  NCU METRIC")
    print("  " + "-" * (8 + 2 + 28 + 2 + col_w))
    for ncu_name, short_name in REQUIRED_METRICS:
        if ncu_name in collected:
            ok.append(ncu_name)
            mark, status = "✓", "OK"
        else:
            missing.append(ncu_name)
            mark, status = "✗", "MISSING"
        print(f"  {mark} {status:<6}  {short_name:<28}  {ncu_name}")

    print()
    print(f"Metrics: {len(ok)}/{len(REQUIRED_METRICS)} collected by the live probe.")

    # ── Section check ─────────────────────────────────────────────────────────
    section_status = check_sections(args.ncu_bin)
    print()
    print("Section availability:")
    all_sections_ok = True
    for section, found in section_status.items():
        mark = "✓" if found else "✗"
        print(f"  {mark}  {section}")
        if not found:
            all_sections_ok = False

    # ── Summary ───────────────────────────────────────────────────────────────
    if missing:
        print()
        print("⚠  MISSING METRICS — the following were not returned by the probe:")
        for m in missing:
            stem = m.split(".")[0]
            candidates = sorted(n for n in collected if stem in n)[:4]
            print(f"  {m}")
            if candidates:
                print(f"    Collected names with same stem: {candidates}")
            else:
                print("    (no matching names found in collected set)")
        print()
        print("Action: update the REQUIRED_METRICS list in collect_ncu_metrics.py")
        print("with the available alternatives shown above.")

    if not all_sections_ok:
        print()
        print("⚠  One or more --section names are not listed by ncu --list-sections.")
        print("   Remove the missing sections from the NCU_SECTIONS list in")
        print("   collect_ncu_metrics.py to avoid a failed run.")

    if not missing and all_sections_ok:
        print()
        print("✓  All 21 metrics and both sections verified.  Safe to run collect_ncu_metrics.py.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
