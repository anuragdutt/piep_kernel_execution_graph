#!/usr/bin/env python3
"""
Background system metrics collector for kernel benchmarking.

Samples GPU and CPU metrics at regular intervals and computes aggregated statistics.
Designed to run in parallel with C++ kernel benchmarking.

Usage:
    # Start collection in background
    python3 collect_system_metrics.py --output metrics.json --interval 0.1 &
    MONITOR_PID=$!
    
    # Run your benchmarks...
    
    # Stop collection
    kill -INT $MONITOR_PID
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import statistics


@dataclass
class GPUMetrics:
    """GPU metrics snapshot"""
    timestamp: float
    gpu_util_pct: float
    mem_util_pct: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    power_w: float  # GPU power for per-kernel energy calculation


@dataclass
class CPUMetrics:
    """CPU metrics snapshot"""
    timestamp: float
    cpu_util_pct: float  # Overall CPU utilization
    cpu_clock_mhz: float  # Average CPU frequency across all cores
    mem_used_mb: float
    mem_total_mb: float
    mem_util_pct: float


class MetricsCollector:
    """Collects GPU and CPU metrics at regular intervals"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.gpu_samples: List[GPUMetrics] = []
        self.cpu_samples: List[CPUMetrics] = []
        self.running = False
        self.last_cpu_stats = None
        
    def sample_gpu(self) -> Optional[GPUMetrics]:
        """Sample GPU metrics using nvidia-smi"""
        try:
            # Query: utilization.gpu, utilization.memory, clocks.sm, clocks.mem, power.draw
            result = subprocess.run(
                ['nvidia-smi', 
                 '--query-gpu=utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            
            if result.returncode != 0:
                return None
            
            # Parse output: "util_gpu, util_mem, sm_clock, mem_clock, power"
            # Note: if multiple GPUs, take first line
            lines = result.stdout.strip().split('\n')
            if len(lines) == 0:
                return None
            
            first_line = lines[0]
            values = [float(x.strip()) for x in first_line.split(',')]
            
            return GPUMetrics(
                timestamp=time.time(),
                gpu_util_pct=values[0],
                mem_util_pct=values[1],
                sm_clock_mhz=values[2],
                mem_clock_mhz=values[3],
                power_w=values[4]
            )
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, IndexError) as e:
            print(f"Warning: Failed to sample GPU: {e}", file=sys.stderr)
            return None
    
    def read_cpu_stats(self) -> Optional[tuple]:
        """Read CPU stats from /proc/stat"""
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()  # First line is aggregate CPU
                # Format: cpu  user nice system idle iowait irq softirq steal guest guest_nice
                parts = line.split()
                if parts[0] != 'cpu':
                    return None
                
                # Sum all time values
                values = [int(x) for x in parts[1:]]
                total_time = sum(values)
                idle_time = values[3]  # idle is 4th field
                
                return (total_time, idle_time, time.time())
        except (IOError, ValueError) as e:
            print(f"Warning: Failed to read /proc/stat: {e}", file=sys.stderr)
            return None
    
    def read_mem_stats(self) -> Optional[tuple]:
        """Read memory stats from /proc/meminfo"""
        try:
            mem_total = 0
            mem_available = 0
            
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_total = int(line.split()[1])  # kB
                    elif line.startswith('MemAvailable:'):
                        mem_available = int(line.split()[1])  # kB
            
            if mem_total == 0:
                return None
            
            mem_used = mem_total - mem_available
            return (mem_total / 1024, mem_used / 1024)  # Convert to MB
            
        except (IOError, ValueError) as e:
            print(f"Warning: Failed to read /proc/meminfo: {e}", file=sys.stderr)
            return None
    
    def read_cpu_clock(self) -> float:
        """Read average CPU clock speed from /proc/cpuinfo in MHz"""
        try:
            total_freq = 0.0
            count = 0
            
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('cpu MHz'):
                        freq = float(line.split(':')[1].strip())
                        total_freq += freq
                        count += 1
            
            if count == 0:
                return 0.0
            
            return total_freq / count
            
        except (IOError, ValueError) as e:
            print(f"Warning: Failed to read CPU frequency: {e}", file=sys.stderr)
            return 0.0
    
    def sample_cpu(self) -> Optional[CPUMetrics]:
        """Sample CPU metrics from /proc/stat and /proc/meminfo"""
        current_stats = self.read_cpu_stats()
        mem_stats = self.read_mem_stats()
        cpu_clock_mhz = self.read_cpu_clock()
        
        if current_stats is None or mem_stats is None:
            return None
        
        total_time, idle_time, timestamp = current_stats
        mem_total_mb, mem_used_mb = mem_stats
        
        # Calculate CPU utilization as difference from last sample
        cpu_util_pct = 0.0
        if self.last_cpu_stats is not None:
            last_total, last_idle, _ = self.last_cpu_stats
            total_diff = total_time - last_total
            idle_diff = idle_time - last_idle
            
            if total_diff > 0:
                cpu_util_pct = 100.0 * (1.0 - (idle_diff / total_diff))
        
        self.last_cpu_stats = current_stats
        
        return CPUMetrics(
            timestamp=timestamp,
            cpu_util_pct=cpu_util_pct,
            cpu_clock_mhz=cpu_clock_mhz,
            mem_used_mb=mem_used_mb,
            mem_total_mb=mem_total_mb,
            mem_util_pct=(mem_used_mb / mem_total_mb * 100.0) if mem_total_mb > 0 else 0.0
        )
    
    def sample_once(self):
        """Collect one sample of both GPU and CPU metrics"""
        gpu = self.sample_gpu()
        cpu = self.sample_cpu()
        
        if gpu is not None:
            self.gpu_samples.append(gpu)
        if cpu is not None:
            self.cpu_samples.append(cpu)
    
    def run(self):
        """Run continuous sampling until stopped"""
        self.running = True
        print(f"Starting metrics collection (interval={self.sample_interval}s)...", file=sys.stderr)
        
        # Initial CPU sample (for delta calculation)
        self.read_cpu_stats()
        time.sleep(0.1)
        
        while self.running:
            self.sample_once()
            time.sleep(self.sample_interval)
    
    def stop(self):
        """Stop sampling"""
        self.running = False
        print(f"\nStopping metrics collection. Collected {len(self.gpu_samples)} GPU samples, {len(self.cpu_samples)} CPU samples", file=sys.stderr)
    
    def compute_statistics(self) -> Dict:
        """Compute aggregated statistics from samples and calculate energy"""
        stats = {
            'num_gpu_samples': len(self.gpu_samples),
            'num_cpu_samples': len(self.cpu_samples),
            'sample_interval': self.sample_interval,
            'duration_s': 0.0,
            'gpu_energy_j': 0.0  # Total GPU energy via trapezoidal integration
        }
        
        if len(self.gpu_samples) > 0:
            start_time = self.gpu_samples[0].timestamp
            end_time = self.gpu_samples[-1].timestamp
            stats['duration_s'] = end_time - start_time
            
            # GPU statistics
            gpu_util = [s.gpu_util_pct for s in self.gpu_samples]
            mem_util = [s.mem_util_pct for s in self.gpu_samples]
            sm_clock = [s.sm_clock_mhz for s in self.gpu_samples]
            mem_clock = [s.mem_clock_mhz for s in self.gpu_samples]
            power = [s.power_w for s in self.gpu_samples]
            
            stats['gpu_utilization_pct'] = {
                'mean': statistics.mean(gpu_util),
                'median': statistics.median(gpu_util),
                'min': min(gpu_util),
                'max': max(gpu_util)
            }
            
            stats['gpu_memory_utilization_pct'] = {
                'mean': statistics.mean(mem_util),
                'median': statistics.median(mem_util),
                'min': min(mem_util),
                'max': max(mem_util)
            }
            
            stats['gpu_sm_clock_mhz'] = {
                'mean': statistics.mean(sm_clock),
                'median': statistics.median(sm_clock),
                'min': min(sm_clock),
                'max': max(sm_clock)
            }
            
            stats['gpu_mem_clock_mhz'] = {
                'mean': statistics.mean(mem_clock),
                'median': statistics.median(mem_clock),
                'min': min(mem_clock),
                'max': max(mem_clock)
            }
            
            stats['gpu_power_w'] = {
                'mean': statistics.mean(power),
                'median': statistics.median(power),
                'min': min(power),
                'max': max(power)
            }
            
            # Calculate GPU energy via trapezoidal integration
            # Energy = Σ(power_i × Δt) where Δt is time between samples
            if len(self.gpu_samples) > 1:
                timestamps = [s.timestamp for s in self.gpu_samples]
                energy_j = 0.0
                for i in range(1, len(timestamps)):
                    dt = timestamps[i] - timestamps[i-1]
                    avg_power = (power[i] + power[i-1]) / 2.0
                    energy_j += avg_power * dt
                stats['gpu_energy_j'] = energy_j
            elif len(self.gpu_samples) == 1:
                # Single sample: estimate energy = power * duration
                stats['gpu_energy_j'] = power[0] * stats['duration_s']
        
        if len(self.cpu_samples) > 1:  # Need at least 2 for utilization calculation
            # CPU statistics (skip first sample as it has util=0)
            cpu_util = [s.cpu_util_pct for s in self.cpu_samples[1:]]
            cpu_clock = [s.cpu_clock_mhz for s in self.cpu_samples]
            mem_util = [s.mem_util_pct for s in self.cpu_samples]
            mem_used = [s.mem_used_mb for s in self.cpu_samples]
            
            if len(cpu_util) > 0:
                stats['cpu_utilization_pct'] = {
                    'mean': statistics.mean(cpu_util),
                    'median': statistics.median(cpu_util),
                    'min': min(cpu_util),
                    'max': max(cpu_util)
                }
            
            stats['cpu_clock_mhz'] = {
                'mean': statistics.mean(cpu_clock),
                'median': statistics.median(cpu_clock),
                'min': min(cpu_clock),
                'max': max(cpu_clock)
            }
            
            stats['cpu_memory_utilization_pct'] = {
                'mean': statistics.mean(mem_util),
                'median': statistics.median(mem_util),
                'min': min(mem_util),
                'max': max(mem_util)
            }
            
            stats['cpu_memory_used_mb'] = {
                'mean': statistics.mean(mem_used),
                'median': statistics.median(mem_used),
                'min': min(mem_used),
                'max': max(mem_used)
            }
            
            stats['cpu_memory_total_mb'] = self.cpu_samples[0].mem_total_mb
        
        return stats
    
    def save_results(self, output_path: str):
        """Save aggregated statistics to JSON file"""
        stats = self.compute_statistics()
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved metrics to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Collect system metrics during kernel benchmarking')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file path')
    parser.add_argument('--interval', '-i', type=float, default=0.1, help='Sampling interval in seconds (default: 0.1)')
    
    args = parser.parse_args()
    
    collector = MetricsCollector(sample_interval=args.interval)
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, stopping...", file=sys.stderr)
        collector.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run collection
    try:
        collector.run()
    except KeyboardInterrupt:
        collector.stop()
    
    # Save results
    collector.save_results(args.output)


if __name__ == '__main__':
    main()
