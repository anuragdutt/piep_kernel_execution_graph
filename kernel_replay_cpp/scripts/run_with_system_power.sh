#!/bin/bash
# Run benchmark with unified power logging (WattsUp + GPU + CPU)
# 
# This script:
# 1. Starts unified power logger (2x WattsUp monitors + GPUs + CPU RAPL)
# 2. Runs the kernel benchmark (C++ isolated kernels)
# 3. Stops the power logger
# 4. Calculates energy consumption using per-kernel method
#
# Usage:
#   ./run_with_system_power.sh isolated --runs 1000
#   ./run_with_system_power.sh compare --model ../bloom_560m_traced.pt --runs 1000
#   ./run_with_system_power.sh isolated --runs 1000 --idle-power 108.0
#   ./run_with_system_power.sh isolated --interval 0.5 --runs 1000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POWER_LOG="$RESULTS_DIR/unified_power_$TIMESTAMP.csv"

# Default mode
MODE="isolated"

# Default sampling interval
POLL_INTERVAL="1.0"

# Idle power baseline (subtract this from measured power to get workload power)
# If not provided, will not subtract idle power
IDLE_POWER=""

# Parse mode (first positional argument)
if [[ $# -gt 0 ]] && [[ "$1" != --* ]]; then
    MODE="$1"
    shift
fi

# Parse optional --interval and --idle-power
BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)     POLL_INTERVAL="$2"; shift 2 ;;
        --idle-power)   IDLE_POWER="$2"; shift 2 ;;
        *)              BENCH_ARGS+=("$1"); shift ;;
    esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== Kernel Benchmark with Unified Power Monitoring ==="
echo "Mode: $MODE"
echo "Timestamp: $TIMESTAMP"
echo "Power log: $POWER_LOG"
if [[ -n "$IDLE_POWER" ]]; then
    echo "Idle power baseline: ${IDLE_POWER}W (will be subtracted)"
else
    echo "Idle power baseline: not provided (no idle subtraction)"
fi

mkdir -p "$RESULTS_DIR"

# Step 1: Start unified power logger
echo ""
echo -e "${GREEN}Step 1: Starting Unified Power Logger${NC}"
echo "  WattsUp monitors: /dev/ttyUSB0 (main), /dev/ttyUSB1 (auxiliary)"
echo "  GPUs: nvidia-smi monitoring"
echo "  CPU: Intel RAPL monitoring"
echo "  Polling interval: ${POLL_INTERVAL}s"

python3 "$SCRIPT_DIR/unified_power_logger.py" \
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

# Step 2: Run benchmark
echo ""
echo -e "${GREEN}Step 2: Running Benchmark (mode: $MODE)${NC}"
echo "  Args: ${BENCH_ARGS[*]}"

set +e
cd "$SCRIPT_DIR/../build"

./kernel_benchmark "$MODE" \
    --output-dir "$RESULTS_DIR/" \
    --kernels ../data/kernel_signatures.json \
    "${BENCH_ARGS[@]}"

BENCH_EXIT=$?
set -e

# Check results based on mode
if [[ "$MODE" == "isolated" ]]; then
    if [ ! -f "$RESULTS_DIR/isolated_kernels_timing.json" ]; then
        echo -e "${RED}Benchmark failed - isolated_kernels_timing.json not saved!${NC}"
        kill $LOGGER_PID 2>/dev/null
        exit 1
    fi
elif [[ "$MODE" == "full" ]]; then
    if [ ! -f "$RESULTS_DIR/full_model_timing.json" ]; then
        echo -e "${RED}Benchmark failed - full_model_timing.json not saved!${NC}"
        kill $LOGGER_PID 2>/dev/null
        exit 1
    fi
else
    # compare mode
    if [ ! -f "$RESULTS_DIR/full_model_timing.json" ] || [ ! -f "$RESULTS_DIR/isolated_kernels_timing.json" ]; then
        echo -e "${RED}Benchmark failed - results not saved!${NC}"
        kill $LOGGER_PID 2>/dev/null
        exit 1
    fi
fi

if [ $BENCH_EXIT -ne 0 ]; then
    echo -e "${YELLOW}Note: Benchmark exited with code $BENCH_EXIT but results were saved${NC}"
fi

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

# Step 4: Calculate energy
echo ""
echo -e "${GREEN}Step 4: Calculating Energy from Power Measurements${NC}"
cd "$SCRIPT_DIR"

# Build command with optional idle power argument
ENERGY_CMD="python3 calculate_per_kernel_energy.py \
    --power-log '$POWER_LOG' \
    --full-model-result '$RESULTS_DIR/full_model_timing.json' \
    --isolated-result '$RESULTS_DIR/isolated_kernels_timing.json' \
    --output '$RESULTS_DIR/unified_energy_report.json'"

if [[ -n "$IDLE_POWER" ]]; then
    ENERGY_CMD="$ENERGY_CMD --idle-power $IDLE_POWER"
fi

eval $ENERGY_CMD

echo ""
echo -e "${GREEN}=== Complete ===${NC}"
echo "Power log: $POWER_LOG"
echo "Energy report: $RESULTS_DIR/unified_energy_report.json"
