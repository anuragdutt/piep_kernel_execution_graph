#!/bin/bash
# Run benchmark with GPU power logging via nvidia-smi
# Measures GPU power (not system). Use when WattsUp / external meter isn't reliable.
#
# Usage:
#   ./run_with_gpu_power.sh --model ../bloom_560m_traced.pt --runs 100
#   ./run_with_gpu_power.sh --gpu 0 --model ../bloom_560m_traced.pt --runs 100   # single GPU only
#
# Optional: CUDA_VISIBLE_DEVICES=0 ./run_with_gpu_power.sh --gpu 0 --model ... --runs 100

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU_POWER_LOG="$RESULTS_DIR/gpu_power_$TIMESTAMP.csv"

# Parse optional --gpu N (default 0: only log GPU 0 so energy = that GPU only)
GPU_ID="0"
BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ID="$2"; shift 2 ;;
        *)     BENCH_ARGS+=("$1"); shift ;;
    esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== BLOOM Kernel Benchmark with GPU Power (nvidia-smi) ==="
echo "Timestamp: $TIMESTAMP"
echo "GPU power log: $GPU_POWER_LOG"

mkdir -p "$RESULTS_DIR"

# Step 1: Start GPU power logger (nvidia-smi, 20 Hz), GPU $GPU_ID only
echo ""
echo -e "${GREEN}Step 1: Starting GPU Power Logger (nvidia-smi, GPU $GPU_ID only)${NC}"
# 0.05s = 20 Hz; -g so we only log the GPU the benchmark uses (default GPU 0)
python3 "$SCRIPT_DIR/gpu_power_logger.py" -o "$GPU_POWER_LOG" -i 0.05 -g "$GPU_ID" &
LOGGER_PID=$!
echo "  Logger PID: $LOGGER_PID (logging GPU $GPU_ID only)"
sleep 2

# Step 2: Run benchmark
echo ""
echo -e "${GREEN}Step 2: Running Benchmark${NC}"
echo "  Args: ${BENCH_ARGS[*]}"

set +e
cd "$SCRIPT_DIR/../build"

./kernel_benchmark compare \
    --output-dir "$RESULTS_DIR/" \
    --kernels ../data/kernel_signatures.json \
    --no-fusion \
    "${BENCH_ARGS[@]}"

BENCH_EXIT=$?
set -e

# Check results
if [ ! -f "$RESULTS_DIR/full_model_timing.json" ] || [ ! -f "$RESULTS_DIR/isolated_kernels_timing.json" ]; then
    echo -e "${RED}Benchmark failed - results not saved!${NC}"
    kill $LOGGER_PID 2>/dev/null
    exit 1
fi

if [ $BENCH_EXIT -ne 0 ]; then
    echo -e "${YELLOW}Note: Benchmark exited with code $BENCH_EXIT but results were saved${NC}"
fi

# Step 3: Stop logger
echo ""
echo -e "${GREEN}Step 3: Stopping GPU Power Logger${NC}"
kill $LOGGER_PID 2>/dev/null
wait $LOGGER_PID 2>/dev/null
sleep 1

# Step 4: Calculate energy
echo ""
echo -e "${GREEN}Step 4: Calculating Energy from GPU Power${NC}"
cd "$SCRIPT_DIR"

python3 calculate_per_kernel_energy.py \
    --power-log "$GPU_POWER_LOG" \
    --full-model-result "$RESULTS_DIR/full_model_timing.json" \
    --isolated-result "$RESULTS_DIR/isolated_kernels_timing.json" \
    --output "$RESULTS_DIR/gpu_energy_report.json" \
    --gpu "$GPU_ID"

echo ""
echo -e "${GREEN}=== Complete ===${NC}"
echo "GPU power log: $GPU_POWER_LOG"
echo "Energy report: $RESULTS_DIR/gpu_energy_report.json"
