#!/bin/bash
# Run Python kernel replay benchmark with unified power logging
#
# This script:
# 1. Starts unified_power_logger.py (2x WattsUp + GPUs + CPU RAPL)
# 2. Runs kernel_replay_benchmark.py (Tiers 1-3 on GPU 0, Tier 4 via torchrun)
# 3. Stops the power logger
# 4. Calls analyze_energy_comparison.py to compare vs full model energy
#
# Usage:
#   ./run_python_replay_with_power.sh
#   ./run_python_replay_with_power.sh --nccl           # include Tier 4 NCCL
#   ./run_python_replay_with_power.sh --interval 0.5   # faster power polling
#   ./run_python_replay_with_power.sh --tiers 1 2 3    # specific tiers only
#
# After the run, update ISOLATED_POWER path in scripts/analyze_energy_comparison.py
# to point at the new unified_power_*.csv produced here.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POWER_LOG="$REPO_ROOT/kernel_replay_cpp/results/unified_power_${TIMESTAMP}.csv"
OUTPUT_DIR="$REPO_ROOT/kernel_replay_cpp/results"

# Default settings
POLL_INTERVAL="1.0"
NCCL_FLAG=""
TIER_ARGS="--tiers 1 2 3 4"
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)     POLL_INTERVAL="$2"; shift 2 ;;
        --nccl)         NCCL_FLAG="--nccl"; TIER_ARGS="--tiers 1 2 3 4"; shift ;;
        --tiers)
            shift
            TIER_VALS=()
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                TIER_VALS+=("$1"); shift
            done
            TIER_ARGS="--tiers ${TIER_VALS[*]}"
            ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== Python Kernel Replay Benchmark with Power Monitoring ==="
echo "Timestamp:      $TIMESTAMP"
echo "Power log:      $POWER_LOG"
echo "Output dir:     $OUTPUT_DIR"
echo "Poll interval:  ${POLL_INTERVAL}s"
echo "Tiers:          $TIER_ARGS"
echo "NCCL mode:      ${NCCL_FLAG:-no}"
echo ""

mkdir -p "$RESULTS_DIR"
mkdir -p "$OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Start unified power logger
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${GREEN}Step 1: Starting Unified Power Logger${NC}"
echo "  WattsUp: /dev/ttyUSB0 (main), /dev/ttyUSB1 (auxiliary)"
echo "  GPUs: nvidia-smi monitoring"
echo "  CPU: Intel RAPL monitoring"
echo "  Polling interval: ${POLL_INTERVAL}s"

python3 "$REPO_ROOT/kernel_replay_cpp/scripts/unified_power_logger.py" \
    -o "$POWER_LOG" \
    -i "$POLL_INTERVAL" &

LOGGER_PID=$!
echo "  Logger PID: $LOGGER_PID"

# Wait for logger to initialize
sleep 5

if ! ps -p $LOGGER_PID > /dev/null 2>&1; then
    echo -e "${RED}Unified power logger failed to start!${NC}"
    exit 1
fi
echo "  Logger running."

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run Python kernel replay benchmark
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Step 2: Running Python Kernel Replay Benchmark${NC}"
echo "  Kernels:   $REPO_ROOT/kernel_replay_cpp/data/kernel_signatures.json"
echo "  Shape log: $SCRIPT_DIR/shape_log_final_rank0.jsonl"
echo "  Tiers:     $TIER_ARGS"

set +e

if [[ -n "$NCCL_FLAG" ]]; then
    # Tier 4 requires torchrun for NCCL distributed init
    echo "  Launcher:  torchrun --nproc_per_node=2"
    torchrun --nproc_per_node=2 \
        "$SCRIPT_DIR/kernel_replay_benchmark.py" \
        --kernels "$REPO_ROOT/kernel_replay_cpp/data/kernel_signatures.json" \
        --shape-log "$SCRIPT_DIR/shape_log_final_rank0.jsonl" \
        --output-dir "$OUTPUT_DIR" \
        $TIER_ARGS \
        $NCCL_FLAG \
        "${EXTRA_ARGS[@]}"
else
    echo "  Launcher:  python3 (single GPU)"
    python3 "$SCRIPT_DIR/kernel_replay_benchmark.py" \
        --kernels "$REPO_ROOT/kernel_replay_cpp/data/kernel_signatures.json" \
        --shape-log "$SCRIPT_DIR/shape_log_final_rank0.jsonl" \
        --output-dir "$OUTPUT_DIR" \
        $TIER_ARGS \
        "${EXTRA_ARGS[@]}"
fi

BENCH_EXIT=$?
set -e

# Verify results file was written
if [ ! -f "$OUTPUT_DIR/isolated_kernels_timing.json" ]; then
    echo -e "${RED}Benchmark failed - isolated_kernels_timing.json not saved!${NC}"
    kill $LOGGER_PID 2>/dev/null
    exit 1
fi

if [ $BENCH_EXIT -ne 0 ]; then
    echo -e "${YELLOW}Note: Benchmark exited with code $BENCH_EXIT but results were saved${NC}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Stop power logger
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Step 3: Stopping Power Logger${NC}"
kill $LOGGER_PID 2>/dev/null
wait $LOGGER_PID 2>/dev/null || true
sleep 2

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
echo "  Saved to:  $POWER_LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Update analyze_energy_comparison.py with new power log path and run
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Step 4: Updating ISOLATED_POWER path in analyze_energy_comparison.py${NC}"

ANALYZE_SCRIPT="$REPO_ROOT/scripts/analyze_energy_comparison.py"

# Patch the ISOLATED_POWER line in the analysis script to point at new CSV
python3 - <<PYEOF
import re

with open("$ANALYZE_SCRIPT") as f:
    content = f.read()

new_line = 'ISOLATED_POWER = (\n    BASE_DIR / "kernel_replay_cpp/results/unified_power_${TIMESTAMP}.csv"\n)'
content = re.sub(
    r'ISOLATED_POWER\s*=\s*\(\s*\n\s*BASE_DIR\s*/\s*"[^"]+"\s*\n\)',
    new_line,
    content
)

with open("$ANALYZE_SCRIPT", "w") as f:
    f.write(content)

print("  Updated ISOLATED_POWER to: unified_power_${TIMESTAMP}.csv")
PYEOF

echo ""
echo -e "${GREEN}Step 5: Running Energy Comparison Analysis${NC}"
python3 "$ANALYZE_SCRIPT"

echo ""
echo -e "${GREEN}=== Complete ===${NC}"
echo "Power log:       $POWER_LOG"
echo "Timing results:  $OUTPUT_DIR/isolated_kernels_timing.json"
echo "Comparison:      $REPO_ROOT/results/energy_comparison_report.json"
echo ""
echo "NOTE: To re-run analysis with idle power subtraction:"
echo "  python3 scripts/analyze_energy_comparison.py --idle-csv <idle_power.csv>"
