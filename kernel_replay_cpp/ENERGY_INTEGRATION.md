# Power Meter Integration Guide

## Overview

The power meter logging system uses **WattsUp meters** connected via USB serial ports (`/dev/ttyUSB0`, `/dev/ttyUSB1`). The integration strategy is:

1. **Run power logger separately** (background process)
2. **Record timestamps** during benchmark execution
3. **Match timestamps** post-execution to calculate energy consumption

## Power Meter System

### Hardware Setup
- **2x WattsUp power meters** connected via USB serial
- Logs power in Watts at specified interval (default: 1 second)
- Output format: `time,id,pm1,pm2,sum`

### Power Logger Scripts

Located in: `/home/pace/sassy_metrics_tools/power_meter_logging/`

| Script | Purpose |
|--------|---------|
| `pm.py` | Python power logger (reads from serial ports) |
| `pm.sh` | Start/stop wrapper with nohup |
| `logpower.sh` | Bash alternative to pm.py |

## Integration Strategy

### Workflow

```
1. Start power logger (background)
   ./logpower.sh start
   
2. Run benchmark with timestamps
   ./kernel_benchmark compare --runs 1000
   (records start_time, end_time in results)
   
3. Stop power logger
   ./logpower.sh stop
   
4. Post-process: match timestamps
   python scripts/calculate_energy.py \
       --power-log power.csv \
       --benchmark-result results/full_model_timing.json \
       --output results/energy_report.json
```

### Timestamp Matching

Power log format:
```csv
time,id,pm1,pm2,sum
2026-02-04 10:15:23,0,125.3,118.7,244.0
2026-02-04 10:15:24,1,126.1,119.2,245.3
...
```

Benchmark records:
```json
{
  "start_timestamp": "2026-02-04 10:15:25.123456",
  "end_timestamp": "2026-02-04 10:15:35.654321",
  "duration_us": 10530865
}
```

Energy calculation:
```python
# 1. Parse power log and benchmark timestamps
# 2. Filter power samples between start_time and end_time
# 3. Integrate power over time: Energy = Σ(power_i × Δt)
```

## Implementation

### Phase 1: Synchronous Power Logging (Current - Placeholder)

The current `energy_hooks.hpp` has a placeholder `DummyProbe` that returns 0.

### Phase 2: Timestamp-Based Integration (Recommended)

Update the benchmark to record timestamps, then post-process with power logs.

#### 1. Update `full_model_benchmark.cpp`

Add timestamp recording:

```cpp
#include <chrono>
#include <iomanip>
#include <sstream>

std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// In benchmark_model():
std::string start_timestamp = get_iso_timestamp();

// ... run benchmark ...

std::string end_timestamp = get_iso_timestamp();

// Add to JSON output:
j["start_timestamp"] = start_timestamp;
j["end_timestamp"] = end_timestamp;
```

#### 2. Create Post-Processing Script

Create `scripts/calculate_energy.py`:

```python
#!/usr/bin/env python3
"""Calculate energy consumption from power log and benchmark timestamps."""

import argparse
import pandas as pd
import json
from datetime import datetime

def parse_timestamp(ts_str):
    """Parse ISO timestamp to datetime."""
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")

def calculate_energy(power_log_path, benchmark_result_path, output_path):
    """
    Match power log with benchmark timestamps and calculate energy.
    
    Energy = Σ(power_i × Δt) in Joules
    """
    # Load power log
    power_df = pd.read_csv(power_log_path)
    power_df['time'] = pd.to_datetime(power_df['time'])
    
    # Load benchmark results
    with open(benchmark_result_path) as f:
        benchmark = json.load(f)
    
    start_time = parse_timestamp(benchmark['start_timestamp'])
    end_time = parse_timestamp(benchmark['end_timestamp'])
    
    # Filter power samples within benchmark window
    mask = (power_df['time'] >= start_time) & (power_df['time'] <= end_time)
    samples = power_df[mask]
    
    if len(samples) == 0:
        print("Error: No power samples found in benchmark time window!")
        return
    
    # Calculate energy (trapezoidal integration)
    # Energy = Σ(power_i × Δt) where Δt is time between samples
    samples = samples.sort_values('time')
    samples['dt'] = samples['time'].diff().dt.total_seconds()
    samples['energy_j'] = samples['sum'] * samples['dt']  # Watts × seconds = Joules
    
    total_energy_j = samples['energy_j'].sum()
    avg_power_w = samples['sum'].mean()
    duration_s = (end_time - start_time).total_seconds()
    
    # Write output
    result = {
        "start_timestamp": benchmark['start_timestamp'],
        "end_timestamp": benchmark['end_timestamp'],
        "duration_seconds": duration_s,
        "num_power_samples": len(samples),
        "avg_power_watts": avg_power_w,
        "total_energy_joules": total_energy_j,
        "energy_wh": total_energy_j / 3600,  # Convert to Watt-hours
        "pm1_avg_watts": samples['pm1'].mean(),
        "pm2_avg_watts": samples['pm2'].mean(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Energy Calculation Complete:")
    print(f"  Duration: {duration_s:.3f} seconds")
    print(f"  Samples: {len(samples)}")
    print(f"  Average Power: {avg_power_w:.2f} W")
    print(f"  Total Energy: {total_energy_j:.2f} J ({total_energy_j / 3600:.6f} Wh)")
    print(f"  Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate energy from power log")
    parser.add_argument("--power-log", required=True, help="Power log CSV file")
    parser.add_argument("--benchmark-result", required=True, help="Benchmark result JSON")
    parser.add_argument("--output", required=True, help="Output energy report JSON")
    args = parser.parse_args()
    
    calculate_energy(args.power_log, args.benchmark_result, args.output)

if __name__ == "__main__":
    main()
```

#### 3. Update Workflow

Create `scripts/run_with_power_logging.sh`:

```bash
#!/bin/bash
# Automated workflow: power logging + benchmark + energy calculation

POWER_LOG_DIR="/home/pace/sassy_metrics_tools/power_meter_logging"
POWER_LOG="power_$(date +%Y%m%d_%H%M%S).csv"
RESULTS_DIR="results"

echo "=== Starting Power Logger ==="
cd "$POWER_LOG_DIR"
./logpower.sh start -o "$POWER_LOG"
sleep 2  # Give logger time to start

echo "=== Running Benchmark ==="
cd -  # Back to kernel_replay_cpp/build
./kernel_benchmark compare \
    --model ../../bloom_560m_traced.pt \
    --kernels ../data/kernel_signatures.json \
    --warmup 20 \
    --runs 1000 \
    --output-dir ../$RESULTS_DIR/

echo "=== Stopping Power Logger ==="
cd "$POWER_LOG_DIR"
./logpower.sh stop

echo "=== Calculating Energy ==="
cd -
python ../scripts/calculate_energy.py \
    --power-log "$POWER_LOG_DIR/$POWER_LOG" \
    --benchmark-result ../$RESULTS_DIR/full_model_timing.json \
    --output ../$RESULTS_DIR/energy_report.json

echo "=== Complete ==="
cat ../$RESULTS_DIR/energy_report.json
```

## Averaging Over Multiple Runs

### Already Implemented ✅

The benchmark already averages over multiple runs:

```bash
./kernel_benchmark compare --runs 1000  # Average over 1000 runs
```

**How it works:**
- Full model benchmark runs N times (specified by `--runs`)
- Each run is timed individually
- Reports: average, min, max across all runs
- Power logger runs continuously during all iterations

### Energy Per Run

```
Total Energy = ∫ P(t) dt  (integrated over all runs)
Energy Per Run = Total Energy / num_runs
```

The post-processing script can calculate this:

```python
energy_per_run_j = total_energy_j / benchmark['num_runs']
```

## Isolated Kernels Energy (Advanced)

For isolated kernel benchmarks, you can:

1. **Option A:** Run all kernels with power logging, calculate total energy
2. **Option B:** Run each kernel tier separately with power logging

```bash
# Log power for each tier separately
./kernel_benchmark isolated --kernels data/kernel_signatures.json
# (while power logger runs)
```

Then match timestamps for each tier to get tier-specific energy breakdown.

## Example Full Workflow

```bash
# 1. Start power logger
cd /home/pace/sassy_metrics_tools/power_meter_logging
./logpower.sh start -o power_bloom_benchmark.csv

# 2. Wait for logger to initialize
sleep 3

# 3. Run benchmark (will record timestamps)
cd /home/pace/piep_kernel_execution_graph/kernel_replay_cpp/build
./kernel_benchmark compare \
    --model ../../bloom_560m_traced.pt \
    --kernels ../data/kernel_signatures.json \
    --warmup 20 \
    --runs 1000

# 4. Stop power logger
cd /home/pace/sassy_metrics_tools/power_meter_logging
./logpower.sh stop

# 5. Calculate energy
cd /home/pace/piep_kernel_execution_graph/kernel_replay_cpp
python scripts/calculate_energy.py \
    --power-log /home/pace/sassy_metrics_tools/power_meter_logging/power_bloom_benchmark.csv \
    --benchmark-result results/full_model_timing.json \
    --output results/energy_report.json

# 6. View results
cat results/energy_report.json
```

## Expected Output

```json
{
  "start_timestamp": "2026-02-04 10:15:25.123",
  "end_timestamp": "2026-02-04 10:20:35.456",
  "duration_seconds": 310.333,
  "num_power_samples": 310,
  "avg_power_watts": 245.6,
  "total_energy_joules": 76218.4,
  "energy_wh": 21.172,
  "energy_per_run_j": 76.218,
  "pm1_avg_watts": 126.3,
  "pm2_avg_watts": 119.3
}
```

## Next Steps

1. Implement timestamp recording in `full_model_benchmark.cpp`
2. Create `scripts/calculate_energy.py` post-processing script
3. Create `scripts/run_with_power_logging.sh` automated workflow
4. Test with actual power meters
5. Add energy metrics to comparison report

## Alternative: Real-Time Integration (Future)

For real-time integration, you could:
- Start power logger subprocess from C++ code
- Use shared memory or named pipes for IPC
- Aggregate power samples in real-time

This is more complex but provides tighter integration. The timestamp-based approach is simpler and sufficient for most use cases.
