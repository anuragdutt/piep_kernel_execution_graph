#!/usr/bin/env python3
"""Calculate per-kernel energy consumption for accurate prediction.

This script:
1. Measures energy for each kernel's benchmark run (N iterations)
2. Calculates energy per single kernel execution = total / N
3. Predicts energy per inference = Σ(energy_per_kernel × invocation_count)
4. Compares with full model energy per inference

Usage:
    python calculate_per_kernel_energy.py \
        --power-log power.csv \
        --full-model-result results/full_model_timing.json \
        --isolated-result results/isolated_kernels_timing.json \
        --output results/per_kernel_energy_report.json
"""

import argparse
import pandas as pd
import json
from datetime import datetime
import sys

def parse_timestamp(ts_str):
    """Parse ISO timestamp to datetime."""
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

def calculate_energy_for_window(power_df, start_time, end_time):
    """Calculate energy for a time window. Uses trapezoidal integration when 2+ samples,
    else power * duration for a single sample."""
    mask = (power_df['time'] >= start_time) & (power_df['time'] <= end_time)
    samples = power_df[mask].copy()
    
    if len(samples) == 0:
        return None, 0
    
    duration_s = (end_time - start_time).total_seconds()
    samples = samples.sort_values('time')

    if len(samples) == 1:
        # Single sample: energy = power * full window duration
        total_energy_j = float(samples['sum'].iloc[0]) * duration_s
        return total_energy_j, 1

    # 2+ samples: trapezoidal integration
    samples['dt'] = samples['time'].diff().dt.total_seconds()
    samples.loc[samples.index[0], 'dt'] = samples.loc[samples.index[1], 'dt']
    samples['energy_j'] = samples['sum'] * samples['dt']
    total_energy_j = samples['energy_j'].sum()
    return total_energy_j, len(samples)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate per-kernel energy for accurate prediction"
    )
    parser.add_argument("--power-log", required=True, help="Power meter log CSV")
    parser.add_argument("--full-model-result", required=True, help="Full model JSON")
    parser.add_argument("--isolated-result", required=True, help="Isolated kernels JSON")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--gpu", type=int, default=None, help="Use only this GPU index from log (default: use all / as logged)")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PER-KERNEL ENERGY CALCULATION")
    print("="*70)
    
    # Load power log
    print(f"\n1. Loading power log: {args.power_log}")
    power_df = pd.read_csv(args.power_log)
    
    # Detect format and normalize to 'time' and 'sum' columns
    if 'timestamp' in power_df.columns:
        power_df['time'] = pd.to_datetime(power_df['timestamp'])
    else:
        power_df['time'] = pd.to_datetime(power_df['time'])
    
    # Handle different power log formats
    if 'power_w' in power_df.columns:
        # GPU power log format from nvidia-smi
        if 'gpu' in power_df.columns:
            if args.gpu is not None:
                power_df = power_df[power_df['gpu'] == args.gpu].copy()
                print(f"   Using GPU {args.gpu} only ({len(power_df)} rows)")
            # One row per timestamp (or sum if multiple GPUs and no --gpu filter)
            grouped = power_df.groupby('time').agg({'power_w': 'sum'}).reset_index()
            power_df = grouped.rename(columns={'power_w': 'sum'})
        else:
            power_df['sum'] = power_df['power_w']
        print("   Format: GPU power (nvidia-smi)")
    elif 'sum' not in power_df.columns:
        if 'power' in power_df.columns:
            power_df['sum'] = power_df['power']
            power_df['pm1'] = power_df['power']
            power_df['pm2'] = 0.0
        print("   Format: System power (WattsUp meter)")
    else:
        print("   Format: Dual power meter")
    
    print(f"   Power samples: {len(power_df)}")
    print(f"   Time range: {power_df['time'].min()} to {power_df['time'].max()}")
    
    # Load full model results
    print(f"\n2. Loading full model results: {args.full_model_result}")
    with open(args.full_model_result) as f:
        full_model = json.load(f)
    
    full_start = parse_timestamp(full_model['start_timestamp'])
    full_end = parse_timestamp(full_model['end_timestamp'])
    full_num_runs = full_model['num_runs']
    
    print(f"   Runs: {full_num_runs}")
    print(f"   Duration: {(full_end - full_start).total_seconds():.3f}s")
    
    # Calculate full model energy
    full_energy, full_samples = calculate_energy_for_window(power_df, full_start, full_end)
    if full_energy is None:
        print("ERROR: No power samples for full model!")
        sys.exit(1)
    
    full_energy_per_inference = full_energy / full_num_runs
    print(f"   Total energy: {full_energy:.2f} J")
    print(f"   Energy per inference: {full_energy_per_inference:.4f} J")
    print(f"   Power samples: {full_samples}")
    
    # Load isolated kernel results
    print(f"\n3. Loading isolated kernel results: {args.isolated_result}")
    with open(args.isolated_result) as f:
        isolated = json.load(f)
    
    kernels = isolated['kernels']
    print(f"   Total unique kernels: {len(kernels)}")
    
    # Calculate per-kernel energy
    print(f"\n4. Calculating per-kernel energy...")
    
    # First, calculate average power during entire isolated benchmark period
    isolated_start = parse_timestamp(isolated.get('start_timestamp', kernels[0]['start_timestamp']))
    isolated_end = parse_timestamp(isolated.get('end_timestamp', kernels[-1]['end_timestamp']))
    
    mask = (power_df['time'] >= isolated_start) & (power_df['time'] <= isolated_end)
    avg_power_w = power_df[mask]['sum'].mean() if mask.any() else power_df['sum'].mean()
    print(f"   Average power during isolated benchmarks: {avg_power_w:.2f} W")
    
    kernel_energies = []
    estimated_energies = []  # For fast kernels
    total_predicted_energy = 0.0
    total_estimated_energy = 0.0
    measured_count = 0
    estimated_count = 0
    
    for i, kernel in enumerate(kernels):
        invocation_count = kernel['invocation_count']
        single_time_us = kernel['single_time_us']
        
        if not kernel['start_timestamp'] or not kernel['end_timestamp']:
            # No timestamps - estimate based on timing
            # Energy = Power × Time
            time_per_exec_s = single_time_us / 1_000_000
            energy_per_exec = avg_power_w * time_per_exec_s
            energy_for_inference = energy_per_exec * invocation_count
            total_estimated_energy += energy_for_inference
            estimated_count += 1
            
            estimated_energies.append({
                "name": kernel['name'],
                "tier": kernel['tier'],
                "invocation_count": invocation_count,
                "single_time_us": single_time_us,
                "energy_per_inference_j": energy_for_inference,
                "method": "estimated"
            })
            continue
        
        start = parse_timestamp(kernel['start_timestamp'])
        end = parse_timestamp(kernel['end_timestamp'])
        benchmark_runs = kernel['benchmark_runs']
        duration_s = (end - start).total_seconds()
        
        # Calculate energy for this kernel's benchmark
        energy_total, num_samples = calculate_energy_for_window(power_df, start, end)
        
        if energy_total is None or num_samples < 1:
            # No power samples in window (kernel ran too fast) - estimate from average power × duration
            time_per_exec_s = single_time_us / 1_000_000
            energy_per_exec = avg_power_w * time_per_exec_s
            energy_for_inference = energy_per_exec * invocation_count
            total_estimated_energy += energy_for_inference
            estimated_count += 1
            
            estimated_energies.append({
                "name": kernel['name'],
                "tier": kernel['tier'],
                "invocation_count": invocation_count,
                "single_time_us": single_time_us,
                "energy_per_inference_j": energy_for_inference,
                "method": "estimated_insufficient_samples"
            })
            continue
        
        # Measured energy: total energy over kernel's whole benchmark window,
        # then average per single run (divide by benchmark_runs), then scale to per-inference
        energy_per_execution = energy_total / benchmark_runs  # J per single kernel run
        energy_for_inference = energy_per_execution * invocation_count  # J per inference
        total_predicted_energy += energy_for_inference
        measured_count += 1
        
        kernel_energies.append({
            "name": kernel['name'],
            "tier": kernel['tier'],
            "invocation_count": invocation_count,
            "benchmark_runs": benchmark_runs,
            "benchmark_duration_s": duration_s,
            "benchmark_total_energy_j": energy_total,
            "energy_per_execution_j": energy_per_execution,
            "energy_per_inference_j": energy_for_inference,
            "power_samples": num_samples,
            "method": "measured"
        })
    
    # Combine measured and estimated
    total_combined_energy = total_predicted_energy + total_estimated_energy
    
    print(f"   Measured: {measured_count} kernels ({total_predicted_energy:.4f} J)")
    print(f"   Estimated: {estimated_count} kernels ({total_estimated_energy:.4f} J)")
    print(f"   Total predicted energy per inference: {total_combined_energy:.4f} J")
    
    # Calculate error using combined energy
    error_j = abs(total_combined_energy - full_energy_per_inference)
    error_pct = (error_j / full_energy_per_inference) * 100.0
    
    # Generate report
    all_kernel_energies = kernel_energies + estimated_energies
    report = {
        "full_model": {
            "num_runs": full_num_runs,
            "total_energy_j": full_energy,
            "energy_per_inference_j": full_energy_per_inference,
            "duration_s": (full_end - full_start).total_seconds(),
            "power_samples": full_samples
        },
        "isolated_kernels": {
            "num_unique_kernels": len(kernels),
            "num_measured": measured_count,
            "num_estimated": estimated_count,
            "measured_energy_j": total_predicted_energy,
            "estimated_energy_j": total_estimated_energy,
            "predicted_energy_per_inference_j": total_combined_energy,
            "avg_power_during_benchmark_w": avg_power_w
        },
        "comparison": {
            "full_model_energy_j": full_energy_per_inference,
            "predicted_energy_j": total_combined_energy,
            "error_j": error_j,
            "error_percent": error_pct,
            "ratio_predicted_to_actual": total_combined_energy / full_energy_per_inference
        },
        "per_kernel_energies": sorted(all_kernel_energies, 
                                       key=lambda x: x['energy_per_inference_j'], 
                                       reverse=True)[:20]  # Top 20 contributors
    }
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("ENERGY COMPARISON SUMMARY")
    print("="*70)
    print(f"Full Model Energy (per inference):     {full_energy_per_inference:.4f} J")
    print(f"Predicted Energy (measured kernels):   {total_predicted_energy:.4f} J ({measured_count} kernels)")
    print(f"Predicted Energy (estimated kernels):  {total_estimated_energy:.4f} J ({estimated_count} kernels)")
    print(f"Total Predicted Energy:                {total_combined_energy:.4f} J")
    print(f"Error:                                  {error_j:.4f} J ({error_pct:.1f}%)")
    print(f"Ratio (Predicted/Actual):               {report['comparison']['ratio_predicted_to_actual']:.3f}x")
    print(f"\nReport saved to: {args.output}")
    print("="*70)

if __name__ == "__main__":
    main()
