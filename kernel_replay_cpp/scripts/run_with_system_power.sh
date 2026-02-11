#!/bin/bash
# Run benchmark with SYSTEM power logging via remote WattsUp probe on etracker1
# 
# The WattsUp probes are swapped:
#   - etracker1's /dev/ttyUSB0 → measures etracker2's system power
#   - etracker2's /dev/ttyUSB0 → measures etracker1's system power
#
# This script:
# 1. Starts remote power logger on etracker1 (via SSH)
# 2. Runs the kernel benchmark on etracker2
# 3. Stops the remote power logger
# 4. Calculates energy consumption using per-kernel method
#
# Usage:
#   ./run_with_system_power.sh --model ../bloom_560m_traced.pt --runs 1000
#   ./run_with_system_power.sh --model ../bloom_560m_traced.pt --runs 1000 --idle-power 108.0
#   ./run_with_system_power.sh --interval 0.5 --model ...   # 2 Hz sampling (WattsUp is ~1Hz anyway)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SYSTEM_POWER_LOG="$RESULTS_DIR/system_power_$TIMESTAMP.csv"

# Default sampling interval (WattsUp meter is ~1 Hz, but we can try faster)
POLL_INTERVAL="1.0"

# Idle power baseline (subtract this from measured power to get workload power)
# If not provided, will not subtract idle power
IDLE_POWER=""

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

echo "=== BLOOM Kernel Benchmark with System Power (Remote WattsUp Probe) ==="
echo "Timestamp: $TIMESTAMP"
echo "System power log: $SYSTEM_POWER_LOG"
if [[ -n "$IDLE_POWER" ]]; then
    echo "Idle power baseline: ${IDLE_POWER}W (will be subtracted)"
else
    echo "Idle power baseline: not provided (no idle subtraction)"
fi

mkdir -p "$RESULTS_DIR"

# Step 1: Start remote power logger
echo ""
echo -e "${GREEN}Step 1: Starting Remote System Power Logger${NC}"
echo "  Remote host: etracker1 (130.245.127.111)"
echo "  Remote probe: /dev/ttyUSB0 → measures etracker2's system power"
echo "  Polling interval: ${POLL_INTERVAL}s"

/home/pace/piep_kernel_execution_graph/.venv/bin/python3 "$SCRIPT_DIR/remote_power_logger.py" \
    -o "$SYSTEM_POWER_LOG" \
    -i "$POLL_INTERVAL" \
    --fetch-interval 5.0 &

LOGGER_PID=$!
echo "  Logger PID: $LOGGER_PID"

# Wait for logger to initialize
sleep 5

# Check if logger started successfully
if ! ps -p $LOGGER_PID > /dev/null 2>&1; then
    echo -e "${RED}Remote power logger failed to start!${NC}"
    exit 1
fi

# Step 2: Run benchmark
echo ""
echo -e "${GREEN}Step 2: Running Benchmark${NC}"
echo "  Args: ${BENCH_ARGS[*]}"

set +e
cd "$SCRIPT_DIR/../build"

./kernel_benchmark compare \
    --output-dir "$RESULTS_DIR/" \
    --kernels ../data/kernel_signatures.json \
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
echo -e "${GREEN}Step 3: Stopping Remote Power Logger${NC}"
kill $LOGGER_PID 2>/dev/null
wait $LOGGER_PID 2>/dev/null || true
sleep 2

# Verify power log exists and has data
if [ ! -f "$SYSTEM_POWER_LOG" ]; then
    echo -e "${RED}Error: System power log not found: $SYSTEM_POWER_LOG${NC}"
    exit 1
fi

LINE_COUNT=$(wc -l < "$SYSTEM_POWER_LOG")
if [ "$LINE_COUNT" -lt 2 ]; then
    echo -e "${RED}Error: System power log appears empty (only $LINE_COUNT lines)${NC}"
    exit 1
fi

echo "  System power log: $LINE_COUNT samples recorded"

# Step 4: Calculate energy
echo ""
echo -e "${GREEN}Step 4: Calculating Energy from System Power${NC}"
cd "$SCRIPT_DIR"

# Build command with optional idle power argument
ENERGY_CMD="/home/pace/piep_kernel_execution_graph/.venv/bin/python3 calculate_per_kernel_energy.py \
    --power-log '$SYSTEM_POWER_LOG' \
    --full-model-result '$RESULTS_DIR/full_model_timing.json' \
    --isolated-result '$RESULTS_DIR/isolated_kernels_timing.json' \
    --output '$RESULTS_DIR/system_energy_report.json'"

if [[ -n "$IDLE_POWER" ]]; then
    ENERGY_CMD="$ENERGY_CMD --idle-power $IDLE_POWER"
fi

eval $ENERGY_CMD

echo ""
echo -e "${GREEN}=== Complete ===${NC}"
echo "System power log: $SYSTEM_POWER_LOG"
echo "Energy report: $RESULTS_DIR/system_energy_report.json"
echo ""
echo "To measure idle power for calibration, run:"
echo "  /home/pace/piep_kernel_execution_graph/.venv/bin/python3 $SCRIPT_DIR/measure_idle_power.py --duration 60"
