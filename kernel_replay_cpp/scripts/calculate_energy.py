#!/usr/bin/env python3
"""Calculate energy consumption from power log and benchmark timestamps.

This script matches timestamps from the power meter log with benchmark
execution times to calculate total energy consumption.

Usage:
    python calculate_energy.py \
        --power-log power.csv \
        --benchmark-result results/full_model_timing.json \
        --output results/energy_report.json
"""

import argparse
import pandas as pd
import json
from datetime import datetime
import sys

def parse_timestamp(ts_str):
    """Parse ISO timestamp to datetime.
    
    Handles both formats:
    - "2026-02-04 10:15:23.123456" (with microseconds)
    - "2026-02-04 10:15:23" (without microseconds)
    """
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse timestamp: {ts_str}")

def calculate_energy(power_log_path, benchmark_result_path, output_path):
    """
    Match power log with benchmark timestamps and calculate energy.
    
    Energy = Σ(power_i × Δt) in Joules
    
    Args:
        power_log_path: Path to power meter CSV log
        benchmark_result_path: Path to benchmark JSON result
        output_path: Path to write energy report JSON
    """
    print(f"Reading power log: {power_log_path}")
    
    # Load power log
    try:
        power_df = pd.read_csv(power_log_path)
        power_df['time'] = pd.to_datetime(power_df['time'])
        
        # Handle different formats:
        # pm.py (2 meters): time, id, pm1, pm2, sum
        # pm1.py (1 meter): time, id, power
        if 'sum' not in power_df.columns:
            if 'power' in power_df.columns:
                # Single meter format - use 'power' as total
                print("  Detected single-meter format (pm1.py)")
                power_df['sum'] = power_df['power']
                power_df['pm1'] = power_df['power']
                power_df['pm2'] = 0.0
            else:
                print(f"Error: Unknown power log format. Columns: {list(power_df.columns)}")
                sys.exit(1)
        else:
            print("  Detected dual-meter format (pm.py)")
            
    except Exception as e:
        print(f"Error reading power log: {e}")
        sys.exit(1)
    
    print(f"  Found {len(power_df)} power samples")
    print(f"  Time range: {power_df['time'].min()} to {power_df['time'].max()}")
    
    # Load benchmark results
    print(f"\nReading benchmark results: {benchmark_result_path}")
    try:
        with open(benchmark_result_path) as f:
            benchmark = json.load(f)
    except Exception as e:
        print(f"Error reading benchmark results: {e}")
        sys.exit(1)
    
    if 'start_timestamp' not in benchmark or 'end_timestamp' not in benchmark:
        print("Error: Benchmark results missing start_timestamp or end_timestamp!")
        print("Please update full_model_benchmark.cpp to record timestamps.")
        sys.exit(1)
    
    start_time = parse_timestamp(benchmark['start_timestamp'])
    end_time = parse_timestamp(benchmark['end_timestamp'])
    
    print(f"  Benchmark start: {start_time}")
    print(f"  Benchmark end:   {end_time}")
    print(f"  Duration: {(end_time - start_time).total_seconds():.3f} seconds")
    
    # Filter power samples within benchmark window
    mask = (power_df['time'] >= start_time) & (power_df['time'] <= end_time)
    samples = power_df[mask].copy()
    
    if len(samples) == 0:
        print("\nError: No power samples found in benchmark time window!")
        print(f"  Benchmark window: {start_time} to {end_time}")
        print(f"  Power log range:  {power_df['time'].min()} to {power_df['time'].max()}")
        print("\nPossible issues:")
        print("  1. Power logger not running during benchmark")
        print("  2. System clock mismatch")
        print("  3. Benchmark completed before power samples recorded")
        sys.exit(1)
    
    print(f"\n  Found {len(samples)} power samples in benchmark window")
    
    # Calculate energy (trapezoidal integration)
    # Energy = Σ(power_i × Δt) where Δt is time between samples
    samples = samples.sort_values('time')
    samples['dt'] = samples['time'].diff().dt.total_seconds()
    
    # For first sample, estimate dt from second sample
    if len(samples) > 1:
        samples.loc[samples.index[0], 'dt'] = samples.loc[samples.index[1], 'dt']
    
    # Energy = Power (W) × Time (s) = Joules
    samples['energy_j'] = samples['sum'] * samples['dt']
    
    total_energy_j = samples['energy_j'].sum()
    avg_power_w = samples['sum'].mean()
    peak_power_w = samples['sum'].max()
    min_power_w = samples['sum'].min()
    duration_s = (end_time - start_time).total_seconds()
    
    # Get num_runs from benchmark if available
    num_runs = benchmark.get('num_runs', 1)
    energy_per_run_j = total_energy_j / num_runs if num_runs > 0 else total_energy_j
    
    # Write output
    result = {
        "start_timestamp": benchmark['start_timestamp'],
        "end_timestamp": benchmark['end_timestamp'],
        "duration_seconds": duration_s,
        "num_power_samples": len(samples),
        "num_runs": num_runs,
        "power_watts": {
            "average": avg_power_w,
            "peak": peak_power_w,
            "min": min_power_w
        },
        "energy_joules": {
            "total": total_energy_j,
            "per_run": energy_per_run_j
        },
        "energy_wh": {
            "total": total_energy_j / 3600,
            "per_run": energy_per_run_j / 3600
        },
        "power_meters": {
            "pm1_avg_watts": samples['pm1'].mean(),
            "pm2_avg_watts": samples['pm2'].mean()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Energy Calculation Complete")
    print(f"{'='*60}")
    print(f"Duration:        {duration_s:.3f} seconds")
    print(f"Samples:         {len(samples)}")
    print(f"Runs:            {num_runs}")
    print(f"Average Power:   {avg_power_w:.2f} W")
    print(f"Peak Power:      {peak_power_w:.2f} W")
    print(f"Total Energy:    {total_energy_j:.2f} J ({total_energy_j / 3600:.6f} Wh)")
    print(f"Energy Per Run:  {energy_per_run_j:.3f} J ({energy_per_run_j / 3600:.9f} Wh)")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Calculate energy from power log and benchmark timestamps"
    )
    parser.add_argument(
        "--power-log",
        required=True,
        help="Power meter log CSV file (from pm.py or logpower.sh)"
    )
    parser.add_argument(
        "--full-model-result",
        required=True,
        help="Full model benchmark result JSON file"
    )
    parser.add_argument(
        "--isolated-result",
        required=True,
        help="Isolated kernels benchmark result JSON file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output energy report JSON file"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ENERGY CALCULATION FOR FULL MODEL")
    print("="*60)
    full_model_energy = calculate_energy(args.power_log, args.full_model_result, 
                                         args.output.replace('.json', '_full_model.json'))
    
    print("\n" + "="*60)
    print("ENERGY CALCULATION FOR ISOLATED KERNELS")
    print("="*60)
    isolated_energy = calculate_energy(args.power_log, args.isolated_result,
                                       args.output.replace('.json', '_isolated_kernels.json'))
    
    # Generate combined comparison report
    combined_report = {
        "full_model": full_model_energy,
        "isolated_kernels": isolated_energy,
        "comparison": {
            "energy_ratio": isolated_energy["energy_joules"]["total"] / full_model_energy["energy_joules"]["total"],
            "energy_difference_j": isolated_energy["energy_joules"]["total"] - full_model_energy["energy_joules"]["total"],
            "energy_error_percent": ((isolated_energy["energy_joules"]["total"] - full_model_energy["energy_joules"]["total"]) / 
                                    full_model_energy["energy_joules"]["total"]) * 100.0
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(combined_report, f, indent=2)
    
    print("\n" + "="*60)
    print("ENERGY COMPARISON")
    print("="*60)
    print(f"Full Model Energy:     {full_model_energy['energy_joules']['total']:.2f} J")
    print(f"Isolated Kernels Energy: {isolated_energy['energy_joules']['total']:.2f} J")
    print(f"Ratio (Isolated/Full): {combined_report['comparison']['energy_ratio']:.2f}x")
    print(f"Error:                 {combined_report['comparison']['energy_error_percent']:.1f}%")
    print(f"\nCombined report saved to: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()
