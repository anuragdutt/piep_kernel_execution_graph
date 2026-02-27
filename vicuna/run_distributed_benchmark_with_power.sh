#!/bin/bash
# Run Vicuna distributed benchmark (TP=2) with unified power monitoring
#
# This script uses the UNIFIED vicuna_tp_profile.py script in benchmark mode
# to ensure profiling and benchmarking use IDENTICAL code paths.
#
# This script:
# 1. Starts unified power logger (2x WattsUp + GPUs + CPU)
# 2. Runs Vicuna-7B with torchrun (TP=2 on 2 GPUs) in benchmark mode
# 3. Stops the power logger
# 4. Calculates total energy consumption
#
# Usage:
#   ./run_distributed_benchmark_with_power.sh --warmup 10 --runs 100
#   ./run_distributed_benchmark_with_power.sh --warmup 5 --runs 50 --interval 0.5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POWER_LOG="$RESULTS_DIR/full_model_power_$TIMESTAMP.csv"
BENCHMARK_OUTPUT="$RESULTS_DIR/full_model_benchmark_$TIMESTAMP.json"

# Default parameters
POLL_INTERVAL="1.0"
WARMUP_RUNS=10
TIMED_RUNS=100
NPROC=2
GPUS="0,1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)     POLL_INTERVAL="$2"; shift 2 ;;
        --warmup)       WARMUP_RUNS="$2"; shift 2 ;;
        --runs)         TIMED_RUNS="$2"; shift 2 ;;
        --nproc)        NPROC="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        *)              echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== Vicuna-7B Distributed Benchmark with Power Monitoring ==="
echo "Timestamp: $TIMESTAMP"
echo "Configuration:"
echo "  Model: lmsys/vicuna-7b-v1.5"
echo "  Tensor Parallelism: TP=$NPROC"
echo "  GPUs: $GPUS"
echo "  Warmup runs: $WARMUP_RUNS"
echo "  Timed runs: $TIMED_RUNS"
echo "  Polling interval: ${POLL_INTERVAL}s"
echo "Output:"
echo "  Power log: $POWER_LOG"
echo "  Benchmark results: $BENCHMARK_OUTPUT"

mkdir -p "$RESULTS_DIR"

# Step 1: Start unified power logger
echo ""
echo -e "${GREEN}Step 1: Starting Unified Power Logger${NC}"
echo "  WattsUp monitors: /dev/ttyUSB0 (main), /dev/ttyUSB1 (auxiliary)"
echo "  GPUs: nvidia-smi monitoring"
echo "  CPU: Intel RAPL monitoring"

python3 "$SCRIPT_DIR/../kernel_replay_cpp/scripts/unified_power_logger.py" \
    -o "$POWER_LOG" \
    -i "$POLL_INTERVAL" &

LOGGER_PID=$!
echo "  Logger PID: $LOGGER_PID"

# Wait for logger to initialize
sleep 5

# Check if logger started successfully
if ! ps -p $LOGGER_PID > /dev/null 2>&1; then
    echo -e "${RED}Unified power logger failed to start!${NC}"
    exit 1
fi

# Step 2: Run distributed benchmark
echo ""
echo -e "${GREEN}Step 2: Running Vicuna Distributed Benchmark (unified script)${NC}"

set +e
CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NPROC \
    --master_port=29500 \
    "$SCRIPT_DIR/vicuna_tp_profile.py" \
    --mode benchmark \
    --model lmsys/vicuna-7b-v1.5 \
    --warmup $WARMUP_RUNS \
    --runs $TIMED_RUNS \
    --output "$BENCHMARK_OUTPUT" \
    --max-new-tokens 64

BENCH_EXIT=$?
set -e

# Step 3: Stop logger
echo ""
echo -e "${GREEN}Step 3: Stopping Power Logger${NC}"
kill $LOGGER_PID 2>/dev/null
wait $LOGGER_PID 2>/dev/null || true
sleep 2

# Verify power log exists and has data
if [ ! -f "$POWER_LOG" ]; then
    echo -e "${RED}Error: Power log not found: $POWER_LOG${NC}"
    exit 1
fi

LINE_COUNT=$(wc -l < "$POWER_LOG")
if [ "$LINE_COUNT" -lt 2 ]; then
    echo -e "${RED}Error: Power log appears empty (only $LINE_COUNT lines)${NC}"
    exit 1
fi

echo "  Power log: $LINE_COUNT samples recorded"

# Check benchmark results
if [ ! -f "$BENCHMARK_OUTPUT" ]; then
    echo -e "${RED}Error: Benchmark output not found: $BENCHMARK_OUTPUT${NC}"
    exit 1
fi

if [ $BENCH_EXIT -ne 0 ]; then
    echo -e "${YELLOW}Warning: Benchmark exited with code $BENCH_EXIT${NC}"
fi

# Step 4: Calculate energy consumption
echo ""
echo -e "${GREEN}Step 4: Calculating Energy Consumption${NC}"

python3 - <<EOF
import json
import pandas as pd
import sys

# Load benchmark results
with open("$BENCHMARK_OUTPUT", "r") as f:
    bench_data = json.load(f)

# Load power log (N/A when only one WattsUp meter reported in that poll)
power_df = pd.read_csv("$POWER_LOG", na_values=["N/A"])
power_df["timestamp"] = pd.to_datetime(power_df["timestamp"])
power_df["system_valid"] = power_df["system_total_watts"].notna()
power_df["system_total_watts"] = power_df["system_total_watts"].ffill().bfill()
power_df["gpu_total_watts"] = power_df["gpu_total_watts"].ffill().bfill()
power_df["cpu_watts"] = power_df["cpu_watts"].ffill().bfill()

# Get timing info from benchmark
start_time = bench_data["start_timestamp"]
end_time = bench_data["end_timestamp"]
duration_s = bench_data["total_duration_s"]
start_dt = pd.to_datetime(start_time)
end_dt = pd.to_datetime(end_time)

# Filter power measurements within benchmark window
filtered_power = power_df[
    (power_df["timestamp"] >= start_dt) &
    (power_df["timestamp"] <= end_dt)
].sort_values("timestamp").reset_index(drop=True)

if len(filtered_power) < 2:
    print(f"Warning: Only {len(filtered_power)} power samples during benchmark", file=sys.stderr)
    sys.exit(1)

# Integrate energy (same approach as analyze_energy_comparison): trapezoidal for GPU/CPU;
# for system use only valid samples and interpolate across N/A gaps.
def integrate_valid_aware(times, values, valid, t_start, t_end):
    valid_idx = [i for i in range(len(times)) if valid.iloc[i]]
    if not valid_idx:
        return float('nan')
    energy = 0.0
    energy += float(values.iloc[valid_idx[0]]) * (times.iloc[valid_idx[0]] - t_start).total_seconds()
    for k in range(len(valid_idx) - 1):
        i, j = valid_idx[k], valid_idx[k + 1]
        dt_s = (times.iloc[j] - times.iloc[i]).total_seconds()
        energy += (float(values.iloc[i]) + float(values.iloc[j])) / 2.0 * dt_s
    energy += float(values.iloc[valid_idx[-1]]) * (t_end - times.iloc[valid_idx[-1]]).total_seconds()
    return energy

t = filtered_power["timestamp"]
system_energy_j = integrate_valid_aware(t, filtered_power["system_total_watts"], filtered_power["system_valid"], start_dt, end_dt)
if pd.isna(system_energy_j) or (filtered_power["system_valid"].sum() == 0):
    print("Error: No valid system_total_watts in power log (need both WattsUp meters to report).", file=sys.stderr)
    sys.exit(1)

# GPU and CPU: trapezoidal over window
dt = t.diff().dt.total_seconds()
dt.iloc[0] = dt.iloc[1]
gpu_energy_j = (filtered_power["gpu_total_watts"] * dt).sum()
cpu_energy_j = (filtered_power["cpu_watts"] * dt).sum()

energy_system_wh = system_energy_j / 3600
energy_gpu_wh = gpu_energy_j / 3600
energy_cpu_wh = cpu_energy_j / 3600

# Averages for reporting (consistent with integrated energy over duration)
avg_system_power = system_energy_j / duration_s
avg_gpu_total_power = gpu_energy_j / duration_s
avg_cpu_power = cpu_energy_j / duration_s

# Print report
print("\n=== Energy Report ===")
print(f"Duration: {duration_s:.2f}s")
print(f"Power samples: {len(filtered_power)}")
print(f"\nAverage Power:")
print(f"  System: {avg_system_power:.2f}W")
print(f"  CPU: {avg_cpu_power:.2f}W")
print(f"  GPU Total: {avg_gpu_total_power:.2f}W")
print(f"\nEnergy Consumption:")
print(f"  System: {energy_system_wh:.4f} Wh ({energy_system_wh * 1000:.2f} mWh)")
print(f"  CPU: {energy_cpu_wh:.4f} Wh ({energy_cpu_wh * 1000:.2f} mWh)")
print(f"  GPU Total: {energy_gpu_wh:.4f} Wh ({energy_gpu_wh * 1000:.2f} mWh)")

# Save energy report
energy_report = {
    "timestamp": "$TIMESTAMP",
    "benchmark_file": "$BENCHMARK_OUTPUT",
    "power_log": "$POWER_LOG",
    "duration_s": duration_s,
    "power_samples": len(filtered_power),
    "average_power_w": {
        "system": float(avg_system_power),
        "cpu": float(avg_cpu_power),
        "gpu_total": float(avg_gpu_total_power)
    },
    "energy_wh": {
        "system": float(energy_system_wh),
        "cpu": float(energy_cpu_wh),
        "gpu_total": float(energy_gpu_wh)
    },
    "benchmark_stats": bench_data["stats"]
}

energy_report_file = "$RESULTS_DIR/full_model_energy_$TIMESTAMP.json"
with open(energy_report_file, "w") as f:
    json.dump(energy_report, f, indent=2)

print(f"\nEnergy report saved to: {energy_report_file}")
EOF

ENERGY_EXIT=$?

if [ $ENERGY_EXIT -ne 0 ]; then
    echo -e "${RED}Error calculating energy${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Complete ===${NC}"
echo "Power log: $POWER_LOG"
echo "Benchmark results: $BENCHMARK_OUTPUT"
echo "Energy report: $RESULTS_DIR/full_model_energy_$TIMESTAMP.json"
