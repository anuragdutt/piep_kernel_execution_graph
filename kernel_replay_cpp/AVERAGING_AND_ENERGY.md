# Averaging and Energy Measurement Guide

## Averaging Over Multiple Runs ✅ Already Implemented

### How It Works

The benchmark **already averages** over multiple runs. This is controlled by the `--runs` parameter:

```bash
./kernel_benchmark compare --runs 1000
```

### Implementation Details

**Full Model Benchmark** (`full_model_benchmark.cpp`):
- Runs the model N times (default: 100, configurable with `--runs`)
- Each run is timed individually with CUDA events
- Calculates: **average**, **min**, **max** across all runs
- Returns average time per run in microseconds

```cpp
for (int i = 0; i < timed_runs; i++) {
    timer.start();
    auto output = model.forward({input_ids});
    timer.stop();
    times_us.push_back(timer.elapsed_us());
}

// Calculate statistics
double avg_time = sum(times_us) / times_us.size();
double min_time = min(times_us);
double max_time = max(times_us);
```

**Isolated Kernel Benchmarks**:
- Each unique kernel is benchmarked with:
  - **10 warmup iterations** (to warm up GPU)
  - **1 timed iteration** (to get single kernel time)
- The single kernel time is multiplied by invocation count from trace

### Default Parameters

| Parameter | Default | What It Controls |
|-----------|---------|------------------|
| `--warmup` | 10 | Warmup iterations before timing |
| `--runs` | 100 | Number of timed iterations (full model) |
| Isolated kernel warmup | 10 | Hardcoded in `benchmark_utils.hpp::benchmark_us()` |

### Recommended Settings

**Quick test (2-3 minutes):**
```bash
./kernel_benchmark compare --warmup 10 --runs 50
```

**Normal run (5-10 minutes):**
```bash
./kernel_benchmark compare --warmup 20 --runs 100
```

**High accuracy (30-60 minutes):**
```bash
./kernel_benchmark compare --warmup 50 --runs 1000
```

**Production benchmark (hours):**
```bash
./kernel_benchmark compare --warmup 100 --runs 10000
```

## Energy Measurement Integration ✅ Implemented

### Overview

Power measurement uses **timestamp matching**:

1. **Power logger runs continuously** in background (samples at 1 Hz)
2. **Benchmark records timestamps** (start/end of measurement period)
3. **Post-processing matches timestamps** to calculate energy

### Power Logger

Located: `/home/pace/sassy_metrics_tools/power_meter_logging/`

**Hardware:**
- 2x WattsUp power meters
- Connected via USB serial (`/dev/ttyUSB0`, `/dev/ttyUSB1`)
- Measures total system power (GPU + CPU + rest)

**Output format:**
```csv
time,id,pm1,pm2,sum
2026-02-04 10:15:23,0,125.3,118.7,244.0
2026-02-04 10:15:24,1,126.1,119.2,245.3
```

### Benchmark Timestamps ✅ Added

Updated `full_model_benchmark.cpp` to record:
- `start_timestamp`: ISO format with milliseconds
- `end_timestamp`: ISO format with milliseconds

Saved in `results/full_model_timing.json`:
```json
{
  "full_model_inference_us": 8234500,
  "num_runs": 1000,
  "start_timestamp": "2026-02-04 10:15:25.123",
  "end_timestamp": "2026-02-04 10:20:35.456"
}
```

### Energy Calculation ✅ Script Created

**Script:** `scripts/calculate_energy.py`

**How it works:**
1. Loads power log CSV
2. Loads benchmark results JSON
3. Filters power samples between start_timestamp and end_timestamp
4. Integrates power over time: **Energy = Σ(Power × Δt)**

**Output:** `results/energy_report.json`
```json
{
  "duration_seconds": 310.333,
  "num_power_samples": 310,
  "num_runs": 1000,
  "power_watts": {
    "average": 245.6,
    "peak": 267.3,
    "min": 228.1
  },
  "energy_joules": {
    "total": 76218.4,
    "per_run": 76.218
  },
  "energy_wh": {
    "total": 21.172,
    "per_run": 0.021172
  }
}
```

### Automated Workflow ✅ Script Created

**Script:** `scripts/run_with_power_logging.sh`

**Usage:**
```bash
cd build
../scripts/run_with_power_logging.sh \
    --model ../../bloom_560m_traced.pt \
    --kernels ../data/kernel_signatures.json \
    --runs 1000
```

**What it does:**
1. Starts power logger (`logpower.sh start`)
2. Waits 3 seconds for initialization
3. Runs benchmark with your arguments
4. Stops power logger (`logpower.sh stop`)
5. Calculates energy (`calculate_energy.py`)
6. Prints summary

**Output files:**
- `results/full_model_timing.json` - Benchmark timing
- `results/comparison_report.json` - Predicted vs actual
- `results/energy_report.json` - Energy consumption
- `power_meter_logging/power_bloom_YYYYMMDD_HHMMSS.csv` - Raw power log

## Complete Example

### 1. Build the project

```bash
cd kernel_replay_cpp
mkdir build && cd build
cmake .. && make -j8
```

### 2. Run with power logging (automated)

```bash
../scripts/run_with_power_logging.sh \
    --model ../../bloom_560m_traced.pt \
    --kernels ../data/kernel_signatures.json \
    --warmup 20 \
    --runs 1000
```

### 3. View results

```bash
# Timing results
cat ../results/full_model_timing.json | jq .

# Energy results
cat ../results/energy_report.json | jq .

# Comparison (predicted vs actual latency)
cat ../results/comparison_report.json | jq .
```

## Manual Workflow (If Needed)

If the automated script doesn't work, run manually:

```bash
# 1. Start power logger
cd /home/pace/sassy_metrics_tools/power_meter_logging
./logpower.sh start -o power_bloom.csv &
sleep 3

# 2. Run benchmark
cd /home/pace/piep_kernel_execution_graph/kernel_replay_cpp/build
./kernel_benchmark compare \
    --model ../../bloom_560m_traced.pt \
    --kernels ../data/kernel_signatures.json \
    --runs 1000

# 3. Stop power logger
cd /home/pace/sassy_metrics_tools/power_meter_logging
./logpower.sh stop

# 4. Calculate energy
cd /home/pace/piep_kernel_execution_graph/kernel_replay_cpp
python scripts/calculate_energy.py \
    --power-log /home/pace/sassy_metrics_tools/power_meter_logging/power_bloom.csv \
    --benchmark-result results/full_model_timing.json \
    --output results/energy_report.json
```

## Energy Per Kernel (Future Work)

To measure energy per kernel tier:

```bash
# Option 1: Run each tier separately with power logging
# (requires code modifications to benchmark each tier independently)

# Option 2: Use predicted energy
# If full model energy = 76.2 J and predicted latency is 95% accurate,
# then you can estimate tier contributions proportionally:
#   Tier 1 energy ≈ (tier1_time_us / total_time_us) × total_energy_j
```

## Accuracy Considerations

### Timing Accuracy
- **CUDA events:** Microsecond precision ✓
- **Averaging over 100-1000 runs:** Reduces variance ✓
- **Warmup iterations:** Avoids cold-start effects ✓

### Energy Accuracy
- **Power logger sampling:** 1 Hz (1 second intervals)
- **For short runs (<10s):** May have only 10 samples → less accurate
- **For long runs (>60s):** 60+ samples → good accuracy

**Recommendation:** For energy measurements, use `--runs 1000` to ensure:
- Run duration > 60 seconds
- More power samples for accurate integration
- Better averaging of timing variance

### Example Timeline

```
Time:  0s -------- 10s -------- 20s -------- ... -------- 60s
Power: 245W       246W         244W          ...         247W
       [Sample 1] [Sample 2]   [Sample 3]               [Sample 60]

Benchmark runs:
  Run 1: 0.5s
  Run 2: 0.5s
  ...
  Run 1000: 0.5s
  Total: ~500s (1000 runs × 0.5s each)

Power samples: ~500 samples @ 1 Hz
Energy: Σ(Power_i × 1s) = very accurate
```

## Summary

✅ **Averaging:** Already implemented via `--runs` parameter (default: 100)
✅ **Timestamps:** Added to `full_model_benchmark.cpp`
✅ **Energy calculation:** Implemented in `calculate_energy.py`
✅ **Automated workflow:** Implemented in `run_with_power_logging.sh`

**Ready to use!** Just run the automated script with your desired number of runs.
