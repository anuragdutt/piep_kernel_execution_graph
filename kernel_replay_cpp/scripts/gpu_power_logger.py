#!/usr/bin/env python3
"""
GPU Power Logger using nvidia-smi
Logs GPU power draw at ~100ms intervals for accurate energy measurement
"""

import subprocess
import time
import datetime
import argparse
import signal
import sys

running = True

def signal_handler(sig, frame):
    global running
    running = False

def get_gpu_power():
    """Get power draw for all GPUs via nvidia-smi (power.draw in watts)."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            powers = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        gpu_idx = int(parts[0].strip())
                        raw = parts[1].strip()
                        if raw.upper() in ('N/A', 'NA', ''):
                            power = 0.0
                        else:
                            power = float(raw)
                        powers.append((gpu_idx, power))
                    except (ValueError, IndexError):
                        continue
            return powers
    except subprocess.TimeoutExpired:
        print("nvidia-smi timeout", file=sys.stderr)
    except Exception as e:
        print(f"Error reading GPU power: {e}", file=sys.stderr)
    return []

def main():
    parser = argparse.ArgumentParser(description='GPU Power Logger')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file')
    parser.add_argument('-i', '--interval', type=float, default=0.04,
                        help='Sampling interval in seconds (default: 0.04 = 25 Hz).')
    parser.add_argument('-g', '--gpu', type=int, default=None, help='Specific GPU index to monitor (default: all)')
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"GPU Power Logger started")
    print(f"Output: {args.output}")
    print(f"Interval: {args.interval}s")
    if args.gpu is not None:
        print(f"Monitoring GPU: {args.gpu}")
    else:
        print("Monitoring all GPUs")
    
    with open(args.output, 'w') as f:
        # Header
        f.write('timestamp,gpu,power_w\n')
        f.flush()
        
        sample_count = 0
        start_time = time.time()
        
        while running:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            powers = get_gpu_power()
            
            for gpu_idx, power in powers:
                if args.gpu is None or gpu_idx == args.gpu:
                    f.write(f'{timestamp},{gpu_idx},{power:.2f}\n')
                    sample_count += 1
            
            f.flush()
            
            # Print status every 100 samples
            if sample_count % 100 == 0 and sample_count > 0:
                elapsed = time.time() - start_time
                rate = sample_count / elapsed if elapsed > 0 else 0
                print(f"  Samples: {sample_count}, Rate: {rate:.1f}/s, Last power: {powers}")
            
            time.sleep(args.interval)
    
    print(f"\nLogged {sample_count} samples to {args.output}")

if __name__ == '__main__':
    main()
