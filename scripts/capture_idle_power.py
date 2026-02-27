#!/usr/bin/env python3
"""
Capture system and GPU idle power (no workload) for fair energy comparison.

Runs the unified power logger for a short period with no benchmark, then
computes average idle power (system, GPU) and saves:
  - results/idle_power_<timestamp>.csv   (raw samples)
  - results/idle_power_stats.json         (idle_system_watts, idle_gpu_watts, duration_s)

Use these with analyze_energy_comparison.py --idle-json results/idle_power_stats.json
to subtract idle power from both full-model and isolated-kernel energy before comparing.
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Capture idle power (no workload) for energy comparison baseline."
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=60.0,
        help="Idle capture duration in seconds (default: 60)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Power sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Directory for output CSV and stats JSON",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Use power logger in GPU-only mode (no WattsUp)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    logger_script = base_dir / "kernel_replay_cpp/scripts/unified_power_logger.py"
    if not logger_script.exists():
        print(f"ERROR: {logger_script} not found", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"idle_power_{timestamp}.csv"

    cmd = [
        sys.executable,
        str(logger_script),
        "-o", str(csv_path),
        "-i", str(args.interval),
    ]
    if args.gpu_only:
        cmd.append("--gpu-only")

    print("=" * 60)
    print("Idle power capture (no workload)")
    print("=" * 60)
    print(f"Duration: {args.duration}s")
    print(f"Output CSV: {csv_path}")
    print("Starting power logger... (do not run any GPU workload)")
    print()

    proc = subprocess.Popen(cmd)
    try:
        time.sleep(args.duration)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(1)

    if not csv_path.exists():
        print("ERROR: Power log was not created", file=sys.stderr)
        return 1

    # Load and compute idle averages (same N/A handling as analyze script)
    df = pd.read_csv(csv_path, na_values=["N/A"])
    if "system_total_watts" in df.columns:
        df["system_valid"] = df["system_total_watts"].notna()
        df["system_total_watts"] = df["system_total_watts"].ffill().bfill()
        idle_system_w = float(df["system_total_watts"].mean())
    else:
        idle_system_w = None
    if "gpu_total_watts" in df.columns:
        df["gpu_total_watts"] = df["gpu_total_watts"].ffill().bfill()
        idle_gpu_w = float(df["gpu_total_watts"].mean())
    else:
        idle_gpu_w = None

    df["time"] = pd.to_datetime(df["timestamp"])
    duration_s = (df["time"].max() - df["time"].min()).total_seconds()

    stats = {
        "idle_system_watts": idle_system_w,
        "idle_gpu_watts": idle_gpu_w,
        "duration_s": duration_s,
        "samples": len(df),
        "csv": str(csv_path),
        "timestamp": timestamp,
    }
    stats_path = args.output_dir / "idle_power_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("Idle power stats:")
    print(f"  System: {idle_system_w:.2f} W" if idle_system_w is not None else "  System: N/A")
    print(f"  GPU:    {idle_gpu_w:.2f} W" if idle_gpu_w is not None else "  GPU: N/A")
    print(f"  Duration: {duration_s:.1f}s  Samples: {len(df)}")
    print(f"  Saved: {stats_path}")
    print()
    print("Run analysis with idle subtraction:")
    print(f"  python scripts/analyze_energy_comparison.py --idle-json {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
