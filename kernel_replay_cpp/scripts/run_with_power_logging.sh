#!/bin/bash
# Automated workflow: power logging + benchmark + energy calculation
#
# This script:
# 1. Starts the power meter logger in background
# 2. Runs the kernel benchmark
# 3. Stops the power logger
# 4. Calculates energy consumption from timestamps
#
# Usage:
#   ./run_with_power_logging.sh [benchmark_args...]
#
# Example:
#   ./run_with_power_logging.sh --model ../../bloom_560m_traced.pt --runs 1000

# Note: Don't use 'set -e' because the benchmark may crash during cleanup
# but still produce valid results that we want to process

# Configuration
POWER_LOG_DIR="/home/pace/sassy_metrics_tools/power_meter_logging"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POWER_LOG="power_bloom_${TIMESTAMP}.csv"
RESULTS_DIR="../results"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== BLOOM Kernel Benchmark with Power Logging ===${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Power log: $POWER_LOG"
echo ""

# Check if power meter logging directory exists
if [ ! -d "$POWER_LOG_DIR" ]; then
    echo -e "${RED}Error: Power meter logging directory not found:${NC}"
    echo "  $POWER_LOG_DIR"
    echo ""
    echo "Please check the path or update POWER_LOG_DIR in this script."
    exit 1
fi

# Check if power meter script exists
if [ ! -f "$POWER_LOG_DIR/logpower.sh" ]; then
    echo -e "${RED}Error: logpower.sh not found in:${NC}"
    echo "  $POWER_LOG_DIR"
    exit 1
fi

# Check if benchmark executable exists
if [ ! -f "./kernel_benchmark" ]; then
    echo -e "${RED}Error: kernel_benchmark executable not found${NC}"
    echo "Please run this script from the build/ directory:"
    echo "  cd build && ../scripts/run_with_power_logging.sh"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${YELLOW}Step 1: Starting Power Logger (headless - single meter)${NC}"

# Use the headless power logger from the scripts directory
HEADLESS_LOGGER="$SCRIPT_DIR/power_logger_headless.py"
POWER_LOG_FULL="$POWER_LOG_DIR/$POWER_LOG"

echo "  Starting headless power logger with output: $POWER_LOG_FULL"
nohup python "$HEADLESS_LOGGER" -o "$POWER_LOG_FULL" > "$POWER_LOG_DIR/logger.out" 2>&1 &
LOGGER_PID=$!
echo $LOGGER_PID > "$POWER_LOG_DIR/pm_headless.pid"

# Wait for logger to initialize
echo "  Waiting for logger to initialize..."
sleep 3

# Check if logger started successfully
if ps -p $LOGGER_PID > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ Power logger started (PID: $LOGGER_PID)${NC}"
else
    echo -e "  ${RED}✗ Failed to start power logger${NC}"
    echo "  Check $POWER_LOG_DIR/logger.out for errors:"
    cat "$POWER_LOG_DIR/logger.out" 2>/dev/null || echo "  (no log file)"
    rm -f "$POWER_LOG_DIR/pm_headless.pid"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 2: Running Benchmark${NC}"
echo "  Arguments: $@"
echo ""

# Run benchmark with user-provided arguments
# Note: paths are relative to build/ directory
# Disable JIT fusion via environment variables (more reliable than C++ API)
export PYTORCH_JIT_USE_NNC_NOT_NVFUSER=0
export PYTORCH_JIT_ENABLE_SLOW_FALLBACK=0
export PYTORCH_JIT_DISABLE_WARNING_PRINTS=1

./kernel_benchmark compare \
    --output-dir "$RESULTS_DIR/" \
    --kernels ../data/kernel_signatures.json \
    --no-fusion \
    "$@"

BENCH_EXIT_CODE=$?

# Check if benchmark results were saved (even if crash at end)
if [ ! -f "$RESULTS_DIR/full_model_timing.json" ] || [ ! -f "$RESULTS_DIR/isolated_kernels_timing.json" ]; then
    echo -e "${RED}Benchmark failed - results not saved!${NC}"
    if [ -f "$POWER_LOG_DIR/pm_headless.pid" ]; then
        kill $(cat "$POWER_LOG_DIR/pm_headless.pid") 2>/dev/null
        rm -f "$POWER_LOG_DIR/pm_headless.pid"
    fi
    exit 1
fi

# Note: Benchmark may crash at end (libtorch cleanup issue) but results are saved
if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo -e "${YELLOW}Note: Benchmark crashed during cleanup but results were saved${NC}"
fi

echo ""
echo -e "${YELLOW}Step 3: Stopping Power Logger${NC}"
if [ -f "$POWER_LOG_DIR/pm_headless.pid" ]; then
    STOP_PID=$(cat "$POWER_LOG_DIR/pm_headless.pid")
    echo "  Stopping power logger (PID: $STOP_PID)"
    kill $STOP_PID 2>/dev/null
    rm -f "$POWER_LOG_DIR/pm_headless.pid"
fi

# Give logger time to finish writing
sleep 2

echo -e "  ${GREEN}✓ Power logger stopped${NC}"

# Check if power log was created and has data
POWER_LOG_FULL="$POWER_LOG_DIR/$POWER_LOG"
if [ ! -f "$POWER_LOG_FULL" ]; then
    echo -e "${RED}Error: Power log file not found: $POWER_LOG_FULL${NC}"
    exit 1
fi

LINE_COUNT=$(wc -l < "$POWER_LOG_FULL")
if [ "$LINE_COUNT" -lt 2 ]; then
    echo -e "${RED}Error: Power log appears empty (only $LINE_COUNT lines)${NC}"
    exit 1
fi

echo "  Power log: $LINE_COUNT samples recorded"

# Check if benchmark results exist
if [ ! -f "$RESULTS_DIR/full_model_timing.json" ]; then
    echo -e "${RED}Error: Full model benchmark results not found${NC}"
    echo "  Expected: $RESULTS_DIR/full_model_timing.json"
    exit 1
fi

if [ ! -f "$RESULTS_DIR/isolated_kernels_timing.json" ]; then
    echo -e "${RED}Error: Isolated kernels benchmark results not found${NC}"
    echo "  Expected: $RESULTS_DIR/isolated_kernels_timing.json"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 4: Calculating Energy${NC}"

# Check if calculate_energy.py exists
ENERGY_SCRIPT="$SCRIPT_DIR/calculate_energy.py"
if [ ! -f "$ENERGY_SCRIPT" ]; then
    echo -e "${RED}Error: calculate_energy.py not found: $ENERGY_SCRIPT${NC}"
    exit 1
fi

# Use the NEW per-kernel energy calculation script for accurate prediction
PER_KERNEL_ENERGY_SCRIPT="$SCRIPT_DIR/calculate_per_kernel_energy.py"
if [ -f "$PER_KERNEL_ENERGY_SCRIPT" ]; then
    echo "  Using per-kernel energy calculation (accurate method)..."
    python3 "$PER_KERNEL_ENERGY_SCRIPT" \
        --power-log "$POWER_LOG_FULL" \
        --full-model-result "$RESULTS_DIR/full_model_timing.json" \
        --isolated-result "$RESULTS_DIR/isolated_kernels_timing.json" \
        --output "$RESULTS_DIR/energy_report.json" || {
        echo -e "${RED}Per-kernel energy calculation failed!${NC}"
        exit 1
    }
else
    # Fallback to old method
    echo "  Using legacy energy calculation..."
    python3 "$ENERGY_SCRIPT" \
        --power-log "$POWER_LOG_FULL" \
        --full-model-result "$RESULTS_DIR/full_model_timing.json" \
        --isolated-result "$RESULTS_DIR/isolated_kernels_timing.json" \
        --output "$RESULTS_DIR/energy_report.json" || {
        echo -e "${RED}Energy calculation failed!${NC}"
        echo ""
        echo "This may happen if timestamps are not recorded in the benchmark."
        echo "Check ENERGY_INTEGRATION.md for instructions on updating full_model_benchmark.cpp"
        exit 1
    }
fi

echo ""
echo -e "${GREEN}=== Complete ===${NC}"
echo ""
echo "Results:"
echo "  - Full model timing:      $RESULTS_DIR/full_model_timing.json"
echo "  - Isolated kernel timing: $RESULTS_DIR/isolated_kernels_timing.json"
echo "  - Comparison report:      $RESULTS_DIR/comparison_report.json"
echo "  - Energy report:          $RESULTS_DIR/energy_report.json"
echo "  - Power log (raw):        $POWER_LOG_FULL"
echo ""
echo "View energy report:"
echo "  cat $RESULTS_DIR/energy_report.json"
echo ""
