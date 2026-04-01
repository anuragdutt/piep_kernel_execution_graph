#!/usr/bin/env python3
"""
Analyze energy consumption: Isolated kernels vs Full model inference

Compares isolated kernel-level energy predictions against actual full-model
inference energy measured with WattsUp + nvidia-smi.

Energy calculation:
1. For each kernel: Total energy during benchmark = avg_power × (end - start)
2. Energy per single execution = total / benchmark_runs
3. Energy for full inference = energy_per_execution × invocation_count
4. Predicted total = sum of all kernel energies

Optional: subtract idle power for a fairer comparison. Capture idle with:
  python scripts/capture_idle_power.py -d 60
then run:
  python scripts/analyze_energy_comparison.py --idle-json results/idle_power_stats.json
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime


def parse_timestamp(ts_str):
    """Parse timestamp string to datetime"""
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")


def load_isolated_timing(path: Path):
    """Load isolated kernel timing data"""
    with open(path) as f:
        return json.load(f)


def load_full_model_energy(path: Path):
    """Load full model energy measurements"""
    with open(path) as f:
        return json.load(f)


def load_full_model_benchmark(path: Path):
    """Load full model benchmark data"""
    with open(path) as f:
        return json.load(f)


def load_power_data(csv_path):
    """Load power data from unified power logger CSV and handle N/A values.

    System total = pm1 + pm2 (only set when both WattsUp meters report; same poll interval
    as GPU but serial reads often miss one meter). We mark originally-valid system rows
    for careful integration; filled values used only for interpolation across gaps.
    """
    df = pd.read_csv(csv_path, na_values=["N/A"])
    df["time"] = pd.to_datetime(df["timestamp"])
    # Mark system power rows that were originally valid (before fill)
    df["system_valid"] = df["system_total_watts"].notna()
    df["system_total_watts"] = df["system_total_watts"].ffill().bfill()
    df["gpu_total_watts"] = df["gpu_total_watts"].ffill().bfill()
    return df


def _integrate_system_energy_valid_aware(samples, start_time, end_time):
    """
    Integrate system power over [start_time, end_time] using only originally-valid
    samples. Between two valid samples use trapezoidal; across gaps use linear
    interpolation (avg of last valid before and first valid after). Avoids spreading
    one sample over long N/A gaps.
    """
    t = samples["time"]
    P = samples["system_total_watts"]
    valid = samples["system_valid"]
    valid_idx = [i for i in range(len(samples)) if valid.iloc[i]]
    if not valid_idx:
        # No valid system sample in window: use filled value for whole window (fallback)
        return float(P.iloc[0]) * (end_time - start_time).total_seconds()

    energy_j = 0.0
    # Leading segment: window start -> first valid sample
    i0 = valid_idx[0]
    energy_j += float(P.iloc[i0]) * (t.iloc[i0] - start_time).total_seconds()
    # Between consecutive valid samples: trapezoidal
    for k in range(len(valid_idx) - 1):
        i, j = valid_idx[k], valid_idx[k + 1]
        dt_s = (t.iloc[j] - t.iloc[i]).total_seconds()
        energy_j += (float(P.iloc[i]) + float(P.iloc[j])) / 2.0 * dt_s
    # Trailing segment: last valid sample -> window end
    i1 = valid_idx[-1]
    energy_j += float(P.iloc[i1]) * (end_time - t.iloc[i1]).total_seconds()
    return energy_j


def calculate_energy_for_window(power_df, start_time, end_time):
    """
    Calculate energy for a time window. GPU: trapezoidal integration (samples are
    dense). System: valid-aware integration (only use rows where both WattsUp meters
    reported; interpolate across N/A gaps) so we don't over/under-weight sparse samples.
    Returns (system_energy_j, gpu_energy_j, num_samples)
    """
    mask = (power_df["time"] >= start_time) & (power_df["time"] <= end_time)
    samples = power_df[mask].copy()

    if len(samples) == 0:
        return None, None, 0

    duration_s = (end_time - start_time).total_seconds()
    samples = samples.sort_values("time").reset_index(drop=True)

    if len(samples) == 1:
        system_energy_j = float(samples["system_total_watts"].iloc[0]) * duration_s
        gpu_energy_j = float(samples["gpu_total_watts"].iloc[0]) * duration_s
        return system_energy_j, gpu_energy_j, 1

    # System: integrate using only valid samples and interpolate across gaps
    system_energy_j = _integrate_system_energy_valid_aware(
        samples, start_time, end_time
    )

    # GPU: trapezoidal (GPU power is sampled every poll, rarely N/A)
    samples["dt"] = samples["time"].diff().dt.total_seconds()
    samples.loc[samples.index[0], "dt"] = samples.loc[samples.index[1], "dt"]
    samples["gpu_energy_j"] = samples["gpu_total_watts"] * samples["dt"]
    gpu_energy_j = samples["gpu_energy_j"].sum()

    return system_energy_j, gpu_energy_j, len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Compare isolated kernel energy vs full model energy."
    )
    parser.add_argument(
        "--isolated-timing",
        type=Path,
        default=None,
        help="Path to isolated_kernels_timing.json from kernel_replay_benchmark.py",
    )
    parser.add_argument(
        "--isolated-power",
        type=Path,
        default=None,
        help="Path to replay_power.csv recorded during the isolated kernel benchmark",
    )
    parser.add_argument(
        "--full-model-energy",
        type=Path,
        default=None,
        help="Path to full_model_energy.json from run_experiment.sh",
    )
    parser.add_argument(
        "--full-model-benchmark",
        type=Path,
        default=None,
        help="Path to full_model_benchmark.json from vicuna_tp_profile.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write energy_comparison_report.json (default: current dir)",
    )
    parser.add_argument(
        "--idle-json",
        type=Path,
        default=None,
        help="JSON with idle_system_watts, idle_gpu_watts (from capture_idle_power.py)",
    )
    parser.add_argument(
        "--idle-csv",
        type=Path,
        default=None,
        help="CSV from idle power capture; compute mean system/gpu as idle (alternative to --idle-json)",
    )
    args = parser.parse_args()

    # All four required paths must be provided explicitly — no more auto-discovery
    # from hardcoded legacy locations.  run_experiment.sh always passes them.
    missing = []
    if args.isolated_timing is None:
        missing.append("--isolated-timing")
    if args.isolated_power is None:
        missing.append("--isolated-power")
    if args.full_model_energy is None:
        missing.append("--full-model-energy")
    if args.full_model_benchmark is None:
        missing.append("--full-model-benchmark")
    if missing:
        print(
            f"ERROR: required arguments not provided: {', '.join(missing)}",
            file=sys.stderr,
        )
        print(
            "  These are set automatically by run_experiment.sh; if running manually "
            "pass each path explicitly.",
            file=sys.stderr,
        )
        return 1

    isolated_timing_path = args.isolated_timing
    isolated_power_path = args.isolated_power
    full_model_energy_path = args.full_model_energy
    full_model_benchmark_path = args.full_model_benchmark

    # Load idle power if provided (for fair comparison: subtract baseline from both sides)
    idle_system_w = None
    idle_gpu_w = None
    if args.idle_json and args.idle_json.exists():
        with open(args.idle_json) as f:
            idle_stats = json.load(f)
        idle_system_w = idle_stats.get("idle_system_watts")
        idle_gpu_w = idle_stats.get("idle_gpu_watts")
        print(
            f"Idle power (from {args.idle_json}): system={idle_system_w} W, gpu={idle_gpu_w} W"
        )
    elif args.idle_csv and args.idle_csv.exists():
        idle_df = load_power_data(args.idle_csv)
        idle_system_w = float(idle_df["system_total_watts"].mean())
        idle_gpu_w = float(idle_df["gpu_total_watts"].mean())
        print(
            f"Idle power (from {args.idle_csv}): system={idle_system_w:.2f} W, gpu={idle_gpu_w:.2f} W"
        )
    use_idle = idle_system_w is not None and idle_gpu_w is not None
    if use_idle:
        print(
            "  → Will subtract idle power from both full-model and isolated for comparison."
        )
    print()

    print("=" * 80)
    print("Energy Comparison: Isolated Kernels vs Full Model")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    timing_data = load_isolated_timing(isolated_timing_path)
    full_model_energy = load_full_model_energy(full_model_energy_path)
    full_model_benchmark = load_full_model_benchmark(full_model_benchmark_path)

    print(f"  Isolated kernels: {len(timing_data['kernels'])} kernels")
    print(
        f"  Full model energy: {full_model_energy['energy_wh']['system']:.4f} Wh (system)"
    )
    print(
        f"                     {full_model_energy['energy_wh']['gpu_total']:.4f} Wh (GPU)"
    )
    print(f"  Full model runs: {full_model_benchmark['timed_runs']}")
    print()

    # Load power data
    print("Loading power measurements...")
    power_df = load_power_data(isolated_power_path)
    print(f"  Isolated benchmark: {len(power_df)} power samples")
    print(
        f"    Duration: {(power_df['time'].max() - power_df['time'].min()).total_seconds():.1f}s"
    )
    print(f"    Avg system power: {power_df['system_total_watts'].mean():.2f}W")
    print(f"    Avg GPU power: {power_df['gpu_total_watts'].mean():.2f}W")
    print()

    # Calculate average power during isolated benchmarks (for estimation fallback)
    avg_system_power = power_df["system_total_watts"].mean()
    avg_gpu_power = power_df["gpu_total_watts"].mean()

    # Calculate per-kernel energy
    print("Calculating per-kernel energy...")
    kernel_energies = []
    skipped_no_timestamps = []  # name, tier, invocation_count (contribute 0 unless we estimate from timing)
    kernels_no_power_samples = []  # name, tier, duration_s, invocation_count (we use avg power; run longer to get measured)
    total_predicted_system = 0.0
    total_predicted_gpu = 0.0
    total_predicted_system_active = 0.0
    total_predicted_gpu_active = 0.0
    measured_count = 0
    estimated_count = 0
    skipped_count = 0

    for i, kernel in enumerate(timing_data["kernels"]):
        invocation_count = kernel["invocation_count"]
        benchmark_runs = kernel["benchmark_runs"]
        single_time_us = kernel.get("single_time_us")
        name = kernel.get("name", "?")
        tier = kernel.get("tier", 0)

        if "start_timestamp" not in kernel or "end_timestamp" not in kernel:
            # No timestamps: estimate from single_time_us if available so we don't leave at zero
            skipped_count += 1
            skipped_no_timestamps.append(
                {"name": name, "tier": tier, "invocation_count": invocation_count}
            )
            if single_time_us is not None and benchmark_runs and benchmark_runs > 0:
                benchmark_duration_s = single_time_us * benchmark_runs / 1e6
                system_energy_j = avg_system_power * benchmark_duration_s
                gpu_energy_j = avg_gpu_power * benchmark_duration_s
                energy_per_exec_system = system_energy_j / benchmark_runs
                energy_per_exec_gpu = gpu_energy_j / benchmark_runs
                energy_for_inference_system = energy_per_exec_system * invocation_count
                energy_for_inference_gpu = energy_per_exec_gpu * invocation_count
                total_predicted_system += energy_for_inference_system
                total_predicted_gpu += energy_for_inference_gpu
                if use_idle:
                    assert idle_system_w is not None and idle_gpu_w is not None
                    system_active_j = max(
                        0.0, system_energy_j - idle_system_w * benchmark_duration_s
                    )
                    gpu_active_j = max(
                        0.0, gpu_energy_j - idle_gpu_w * benchmark_duration_s
                    )
                    total_predicted_system_active += (
                        system_active_j / benchmark_runs
                    ) * invocation_count
                    total_predicted_gpu_active += (
                        gpu_active_j / benchmark_runs
                    ) * invocation_count
                estimated_count += 1
                kernel_energies.append(
                    {
                        "name": name,
                        "tier": tier,
                        "invocation_count": invocation_count,
                        "benchmark_runs": benchmark_runs,
                        "benchmark_total_system_j": system_energy_j,
                        "benchmark_total_gpu_j": gpu_energy_j,
                        "energy_per_exec_system_j": energy_per_exec_system,
                        "energy_per_exec_gpu_j": energy_per_exec_gpu,
                        "energy_for_inference_system_j": energy_for_inference_system,
                        "energy_for_inference_gpu_j": energy_for_inference_gpu,
                        "power_samples": 0,
                        "method": "estimated_no_timestamps",
                    }
                )
            continue

        start = parse_timestamp(kernel["start_timestamp"])
        end = parse_timestamp(kernel["end_timestamp"])
        duration_s = (end - start).total_seconds()

        # Calculate energy for this kernel's benchmark window
        system_energy_j, gpu_energy_j, num_samples = calculate_energy_for_window(
            power_df, start, end
        )

        if system_energy_j is None or num_samples < 1:
            # No power samples in window (e.g. window too short for 1s polling) - estimate from avg power
            system_energy_j = avg_system_power * duration_s
            gpu_energy_j = avg_gpu_power * duration_s
            method = "estimated"
            estimated_count += 1
            kernels_no_power_samples.append(
                {
                    "name": name,
                    "tier": tier,
                    "duration_s": round(duration_s, 2),
                    "invocation_count": invocation_count,
                }
            )
        else:
            method = "measured"
            measured_count += 1

        # Energy per single execution
        energy_per_exec_system = system_energy_j / benchmark_runs
        energy_per_exec_gpu = gpu_energy_j / benchmark_runs

        # Energy for full inference (scale by invocation count)
        energy_for_inference_system = energy_per_exec_system * invocation_count
        energy_for_inference_gpu = energy_per_exec_gpu * invocation_count

        total_predicted_system += energy_for_inference_system
        total_predicted_gpu += energy_for_inference_gpu
        if use_idle:
            assert idle_system_w is not None and idle_gpu_w is not None
            assert gpu_energy_j is not None  # reassigned above if it was None
            system_active_j = max(0.0, system_energy_j - idle_system_w * duration_s)
            gpu_active_j = max(0.0, gpu_energy_j - idle_gpu_w * duration_s)
            total_predicted_system_active += (
                system_active_j / benchmark_runs
            ) * invocation_count
            total_predicted_gpu_active += (
                gpu_active_j / benchmark_runs
            ) * invocation_count

        kernel_energies.append(
            {
                "name": name,
                "tier": kernel["tier"],
                "invocation_count": invocation_count,
                "benchmark_runs": benchmark_runs,
                "benchmark_total_system_j": system_energy_j,
                "benchmark_total_gpu_j": gpu_energy_j,
                "energy_per_exec_system_j": energy_per_exec_system,
                "energy_per_exec_gpu_j": energy_per_exec_gpu,
                "energy_for_inference_system_j": energy_for_inference_system,
                "energy_for_inference_gpu_j": energy_for_inference_gpu,
                "power_samples": num_samples if system_energy_j is not None else 0,
                "method": method,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(timing_data['kernels'])} kernels...")

    print(f"  Measured: {measured_count} kernels (power samples in window)")
    print(
        f"  Estimated: {estimated_count} kernels (included in total: no samples → avg×duration, or no timestamps → timing-based)"
    )
    if skipped_count > 0:
        estimated_from_timing = sum(
            1 for k in kernel_energies if k.get("method") == "estimated_no_timestamps"
        )
        print(
            f"  No timestamps: {skipped_count} kernels  ({estimated_from_timing} estimated from single_time_us, rest contribute 0)"
        )
    if kernels_no_power_samples:
        print(
            f"  Kernels with no power samples in window: {len(kernels_no_power_samples)} (run longer or increase poll rate)"
        )
    print()

    # Print kernels that had no power samples (so user can run them longer)
    if kernels_no_power_samples:
        print("=" * 80)
        print(
            "KERNELS WITH NO POWER SAMPLES (energy = avg power × duration; run longer for measured)"
        )
        print("=" * 80)
        # Sort by duration ascending so shortest (most likely to need longer run) first
        for k in sorted(kernels_no_power_samples, key=lambda x: x["duration_s"])[:30]:
            print(
                f"  Tier {k['tier']}  duration={k['duration_s']}s  inv={k['invocation_count']}  {k['name'][:60]}"
            )
        if len(kernels_no_power_samples) > 30:
            print(
                f"  ... and {len(kernels_no_power_samples) - 30} more (see report JSON)"
            )
        print()

    # Aggregate by tier (include Tier 4 for AllReduce/communication)
    tier_summary = {}
    for tier_num in [1, 2, 3, 4]:
        tier_kernels = [k for k in kernel_energies if k["tier"] == tier_num]
        tier_summary[tier_num] = {
            "count": len(tier_kernels),
            "total_invocations": sum(k["invocation_count"] for k in tier_kernels),
            "total_energy_system_j": sum(
                k["energy_for_inference_system_j"] for k in tier_kernels
            ),
            "total_energy_gpu_j": sum(
                k["energy_for_inference_gpu_j"] for k in tier_kernels
            ),
        }

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    print("Tier Breakdown:")
    print("-" * 80)
    tier_names = {
        1: "CUDA Runtime",
        2: "cuBLAS/GEMM",
        3: "libtorch",
        4: "Communication (NCCL)",
    }
    for tier_num in [1, 2, 3, 4]:
        t = tier_summary[tier_num]
        print(f"Tier {tier_num} ({tier_names[tier_num]}):")
        print(f"  Unique kernels:    {t['count']}")
        print(f"  Total invocations: {t['total_invocations']}")
        pct_sys = (
            (t["total_energy_system_j"] / total_predicted_system * 100)
            if total_predicted_system
            else 0
        )
        pct_gpu = (
            (t["total_energy_gpu_j"] / total_predicted_gpu * 100)
            if total_predicted_gpu
            else 0
        )
        print(
            f"  System energy:     {t['total_energy_system_j']:.4f} J ({pct_sys:.2f}%)"
        )
        print(f"  GPU energy:        {t['total_energy_gpu_j']:.4f} J ({pct_gpu:.2f}%)")
        print()

    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()

    # Full model energy (actual) - convert from Wh to J
    # The power window in run_distributed_benchmark_with_power.sh is [start_timestamp, end_timestamp],
    # which in vicuna_tp_profile.py is only the timed runs loop (not warmup). So total energy is for
    # timed_runs inferences; divide by timed_runs to get per-inference (same idea as isolated: total
    # energy over benchmark / number of runs).
    actual_system_j = full_model_energy["energy_wh"]["system"] * 3600
    actual_gpu_j = full_model_energy["energy_wh"]["gpu_total"] * 3600
    num_runs_in_window = full_model_benchmark[
        "timed_runs"
    ]  # power was logged only during timed runs
    actual_system_per_inference_j = actual_system_j / num_runs_in_window
    actual_gpu_per_inference_j = actual_gpu_j / num_runs_in_window

    full_model_duration_s = full_model_benchmark.get("total_duration_s")
    if use_idle and full_model_duration_s is not None:
        assert (
            idle_system_w is not None and idle_gpu_w is not None
        )  # narrowing for type checker
        actual_system_active_j = actual_system_j - idle_system_w * full_model_duration_s
        actual_gpu_active_j = actual_gpu_j - idle_gpu_w * full_model_duration_s
        actual_system_per_inference_active_j = (
            max(0.0, actual_system_active_j) / num_runs_in_window
        )
        actual_gpu_per_inference_active_j = (
            max(0.0, actual_gpu_active_j) / num_runs_in_window
        )
    else:
        actual_system_per_inference_active_j = None
        actual_gpu_per_inference_active_j = None

    tp_size = full_model_benchmark.get("tensor_parallel_size", 1)
    mean_ms = full_model_benchmark.get("stats", {}).get("mean_ms")

    print(f"Full Model (Actual - per inference):")
    print(
        f"  Tensor parallelism: TP={tp_size}  (reported latency is parallel wall time)"
    )
    if mean_ms is not None:
        print(f"  Mean latency:       {mean_ms:.2f} ms")
    print(
        f"  Total energy / {num_runs_in_window} runs  (power window = timed runs only)"
    )
    print(f"  System energy:    {actual_system_per_inference_j:.4f} J")
    print(f"  GPU energy:       {actual_gpu_per_inference_j:.4f} J")
    print()

    print(f"Isolated Kernels (Predicted - per inference):")
    print(f"  System energy:    {total_predicted_system:.4f} J")
    print(f"  GPU energy:       {total_predicted_gpu:.4f} J")
    print()

    # Calculate error
    error_system_j = abs(total_predicted_system - actual_system_per_inference_j)
    error_gpu_j = abs(total_predicted_gpu - actual_gpu_per_inference_j)
    error_system_pct = (error_system_j / actual_system_per_inference_j) * 100
    error_gpu_pct = (error_gpu_j / actual_gpu_per_inference_j) * 100

    ratio_system = total_predicted_system / actual_system_per_inference_j
    ratio_gpu = total_predicted_gpu / actual_gpu_per_inference_j

    print(f"Prediction Error:")
    print(f"  System: {error_system_j:.4f} J ({error_system_pct:.2f}%)")
    print(f"  GPU:    {error_gpu_j:.4f} J ({error_gpu_pct:.2f}%)")
    print()

    print(f"Prediction Ratio (Predicted/Actual):")
    print(f"  System: {ratio_system:.3f}x")
    print(f"  GPU:    {ratio_gpu:.3f}x")
    print()

    # Comparison after subtracting idle power (fairer: same baseline removed from both sides)
    error_system_active_pct = None
    error_gpu_active_pct = None
    if (
        use_idle
        and actual_system_per_inference_active_j is not None
        and actual_gpu_per_inference_active_j is not None
    ):
        print("=" * 80)
        print("COMPARISON (after subtracting idle power)")
        print("=" * 80)
        print()
        print(
            f"Full Model active (per inference):  System {actual_system_per_inference_active_j:.4f} J   GPU {actual_gpu_per_inference_active_j:.4f} J"
        )
        print(
            f"Isolated predicted active (per inf): System {total_predicted_system_active:.4f} J   GPU {total_predicted_gpu_active:.4f} J"
        )
        print()
        error_system_active_j = abs(
            total_predicted_system_active - actual_system_per_inference_active_j
        )
        error_gpu_active_j = abs(
            total_predicted_gpu_active - actual_gpu_per_inference_active_j
        )
        error_system_active_pct = (
            (error_system_active_j / actual_system_per_inference_active_j) * 100
            if actual_system_per_inference_active_j
            else 0
        )
        error_gpu_active_pct = (
            (error_gpu_active_j / actual_gpu_per_inference_active_j) * 100
            if actual_gpu_per_inference_active_j
            else None
        )
        ratio_system_active = (
            total_predicted_system_active / actual_system_per_inference_active_j
            if actual_system_per_inference_active_j
            else None
        )
        ratio_gpu_active = (
            total_predicted_gpu_active / actual_gpu_per_inference_active_j
            if actual_gpu_per_inference_active_j
            else None
        )
        gpu_pct_str = (
            f"{error_gpu_active_pct:.2f}%"
            if error_gpu_active_pct is not None
            else "N/A (actual active=0)"
        )
        print(
            f"Prediction Error (active only):  System {error_system_active_j:.4f} J ({error_system_active_pct:.2f}%)   GPU {error_gpu_active_j:.4f} J ({gpu_pct_str})"
        )
        if ratio_system_active is not None and ratio_gpu_active is not None:
            print(
                f"Prediction Ratio (active only):  System {ratio_system_active:.3f}x   GPU {ratio_gpu_active:.3f}x"
            )
        print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    if abs(error_system_pct) < 10 and abs(error_gpu_pct) < 10:
        print("✓ EXCELLENT: Prediction error < 10% for both system and GPU")
    elif abs(error_system_pct) < 20 and abs(error_gpu_pct) < 20:
        print("✓ GOOD: Prediction error < 20% for both system and GPU")
    elif abs(error_system_pct) < 30 and abs(error_gpu_pct) < 30:
        print(
            "⚠ MODERATE: Prediction error < 30% - acceptable but room for improvement"
        )
    else:
        print("✗ POOR: Prediction error > 30% - significant discrepancy")

    if ratio_system > 1.5 or ratio_gpu > 1.5:
        print("\n⚠ WARNING: Isolated kernels consuming MORE energy than full model!")
        print("  This suggests isolated replay runs slower than optimized inference.")
        print("  Possible causes:")
        print("    - Lack of kernel fusion in isolated replay")
        print("    - Missing pipelining and concurrent execution")
        print("    - Cold cache effects in isolated benchmarks")
        print("    - Different GPU frequency scaling behavior")
    elif ratio_system < 0.5 or ratio_gpu < 0.5:
        print("\n⚠ WARNING: Isolated kernels consuming LESS energy than full model!")
        print("  Missing components:")
        print("    - NCCL communication (all-reduce, all-gather for TP=2)")
        print("    - Framework overhead (PyTorch scheduling)")
        print("    - Memory management and allocations")
        print("    - GPU idle time between operations")

    print()

    # Top energy-consuming kernels
    print("=" * 80)
    print("TOP 10 ENERGY-CONSUMING KERNELS (GPU)")
    print("=" * 80)
    print()

    sorted_kernels = sorted(
        kernel_energies, key=lambda k: k["energy_for_inference_gpu_j"], reverse=True
    )

    print(
        f"{'Rank':<6} {'Tier':<6} {'Invocations':<12} {'GPU Energy (J)':<18} {'% of Total':<12} {'Kernel Name'}"
    )
    print("-" * 160)

    for i, kernel in enumerate(sorted_kernels[:10], 1):
        pct = (kernel["energy_for_inference_gpu_j"] / total_predicted_gpu) * 100
        name = kernel["name"][:80]  # Truncate long names
        print(
            f"{i:<6} {kernel['tier']:<6} {kernel['invocation_count']:<12} {kernel['energy_for_inference_gpu_j']:<18.6f} {pct:<12.2f} {name}"
        )

    print()

    # Save detailed report
    output_file = args.output_dir / "energy_comparison_report.json"
    output_file.parent.mkdir(exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "idle_power_subtraction": (
            {
                "idle_system_watts": idle_system_w,
                "idle_gpu_watts": idle_gpu_w,
                "full_model_duration_s": full_model_duration_s,
                "per_inference_system_active_j": actual_system_per_inference_active_j,
                "per_inference_gpu_active_j": actual_gpu_per_inference_active_j,
                "predicted_per_inference_system_active_j": total_predicted_system_active,
                "predicted_per_inference_gpu_active_j": total_predicted_gpu_active,
                "error_system_active_pct": error_system_active_pct,
                "error_gpu_active_pct": error_gpu_active_pct,
            }
            if use_idle
            else None
        ),
        "full_model_actual": {
            "tensor_parallel_size": full_model_benchmark.get("tensor_parallel_size", 1),
            "mean_latency_ms": full_model_benchmark.get("stats", {}).get("mean_ms"),
            "total_system_energy_j": actual_system_j,
            "total_gpu_energy_j": actual_gpu_j,
            "num_runs": num_runs_in_window,
            "per_inference_system_j": actual_system_per_inference_j,
            "per_inference_gpu_j": actual_gpu_per_inference_j,
        },
        "isolated_kernels_predicted": {
            "per_inference_system_j": total_predicted_system,
            "per_inference_gpu_j": total_predicted_gpu,
            "num_measured": measured_count,
            "num_estimated": estimated_count,
            "num_skipped_no_timestamps": skipped_count,
            "kernels_no_power_samples": kernels_no_power_samples,
            "skipped_no_timestamps": skipped_no_timestamps,
        },
        "prediction_error": {
            "system_j": error_system_j,
            "gpu_j": error_gpu_j,
            "system_pct": error_system_pct,
            "gpu_pct": error_gpu_pct,
        },
        "prediction_ratio": {
            "system": ratio_system,
            "gpu": ratio_gpu,
        },
        "tier_breakdown": tier_summary,
        "top_kernels_gpu": [
            {
                "rank": i,
                "name": k["name"],
                "tier": k["tier"],
                "invocations": k["invocation_count"],
                "gpu_energy_j": k["energy_for_inference_gpu_j"],
                "pct_of_total": (k["energy_for_inference_gpu_j"] / total_predicted_gpu)
                * 100,
            }
            for i, k in enumerate(sorted_kernels[:20], 1)
        ],
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Detailed report saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
