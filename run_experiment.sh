#!/bin/bash
# run_experiment.sh — End-to-end energy experiment for Vicuna-7B TP=2
#
# Full pipeline, run fresh for every invocation:
#
#   Stage 1 — Full-model inference (torchrun, TP=2, mode=both)
#             Produces a fresh Chrome trace AND benchmark latency/energy JSON.
#             Power is monitored for the entire run; energy is integrated over
#             the timed-runs window only.
#             Outputs:
#               results/<TS>/trace_rank{0,1}.json
#               results/<TS>/shape_log_rank{0,1}.jsonl
#               results/<TS>/full_model_benchmark.json
#               results/<TS>/full_model_power.csv
#               results/<TS>/full_model_energy.json
#
#   Stage 2 — Kernel extraction
#             Reads both rank traces → unique_kernels_compute.jsonl
#             Output: results/<TS>/unique_kernels_compute.jsonl
#
#   Stage 3 — Kernel classification
#             unique_kernels_compute.jsonl → kernel_signatures.json
#             Output: results/<TS>/kernel_signatures.json
#
#   Stage 4 — Kernel replay benchmark with power monitoring
#             kernel_signatures.json → isolated_kernels_timing.json
#             Power monitored throughout; timestamps in timing JSON allow
#             per-kernel energy extraction in Stage 5.
#             Outputs:
#               results/<TS>/isolated_kernels_timing.json
#               results/<TS>/replay_power.csv
#
#   Stage 5 — Energy comparison analysis
#             full_model_energy.json + isolated_kernels_timing.json + replay_power.csv
#             → results/<TS>/energy_comparison_report.json
#
# Usage:
#   ./run_experiment.sh                          # 64 decode tokens (default)
#   ./run_experiment.sh --decode-tokens 1        # prefill only (1 new token)
#   ./run_experiment.sh --decode-tokens 20       # 20 decode steps
#   ./run_experiment.sh --decode-tokens 64       # 64 decode steps (default)
#   ./run_experiment.sh --nccl                   # include Tier 4 NCCL in replay
#   ./run_experiment.sh --warmup 5 --runs 50     # full-model benchmark params
#   ./run_experiment.sh --interval 0.5           # power poll rate (default: 1.0s)
#   ./run_experiment.sh --skip-full-model        # re-use latest full-model files
#   ./run_experiment.sh --skip-replay            # skip kernel replay stage
#   ./run_experiment.sh --idle-json <path>       # subtract idle baseline in Stage 5
#   ./run_experiment.sh --run-dir results/myrun  # use a specific output directory

set -euo pipefail

# ── Directory layout ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VICUNA_DIR="$SCRIPT_DIR/vicuna"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
CLASSIFY_SCRIPT="$SCRIPT_DIR/scripts/classify_kernels.py"
POWER_LOGGER="$SCRIPT_DIR/scripts/unified_power_logger.py"

# ── Timestamp (used as default run directory name) ────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Defaults ──────────────────────────────────────────────────────────────────
DECODE_TOKENS=64          # --decode-tokens: number of new tokens to generate
POLL_INTERVAL="1.0"       # power logger poll rate (seconds)
WARMUP_RUNS=10            # vicuna_tp_profile.py --warmup
TIMED_RUNS=100            # vicuna_tp_profile.py --runs
NCCL_FLAG=""              # set to "--nccl" to enable Tier 4 NCCL replay
TIER_ARGS="1 2 3"         # replay tiers; Tier 4 added automatically with --nccl
SKIP_FULL_MODEL=0
SKIP_REPLAY=0
IDLE_ARGS=()
RUN_DIR=""                # if empty, auto-set to results/<TIMESTAMP>
PROMPT="Explain tensor parallelism in one paragraph."
MODEL="lmsys/vicuna-7b-v1.5"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --decode-tokens)   DECODE_TOKENS="$2";  shift 2 ;;
        --interval)        POLL_INTERVAL="$2";  shift 2 ;;
        --warmup)          WARMUP_RUNS="$2";    shift 2 ;;
        --runs)            TIMED_RUNS="$2";     shift 2 ;;
        --model)           MODEL="$2";          shift 2 ;;
        --prompt)          PROMPT="$2";         shift 2 ;;
        --nccl)            NCCL_FLAG="--nccl";  TIER_ARGS="1 2 3 4"; shift ;;
        --skip-full-model) SKIP_FULL_MODEL=1;   shift ;;
        --skip-replay)     SKIP_REPLAY=1;       shift ;;
        --idle-json)       IDLE_ARGS=("--idle-json" "$2"); shift 2 ;;
        --idle-csv)        IDLE_ARGS=("--idle-csv" "$2");  shift 2 ;;
        --run-dir)         RUN_DIR="$2";        shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--decode-tokens N] [--warmup N] [--runs N] [--nccl]" >&2
            echo "       [--skip-full-model] [--skip-replay] [--interval S]" >&2
            echo "       [--idle-json PATH] [--run-dir PATH]" >&2
            exit 1 ;;
    esac
done

# ── Run directory ─────────────────────────────────────────────────────────────
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$SCRIPT_DIR/results/${TIMESTAMP}_decode${DECODE_TOKENS}tok"
fi
mkdir -p "$RUN_DIR"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Helper: pick latest matching file ─────────────────────────────────────────
latest_file() {
    local pattern="$1"
    local match
    match=$(ls -1 $pattern 2>/dev/null | tail -1)
    echo "$match"
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Vicuna-7B TP=2 Energy Experiment — End-to-End               ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "Run directory:   $RUN_DIR"
echo "Timestamp:       $TIMESTAMP"
echo "Decode tokens:   $DECODE_TOKENS  (1 = prefill only)"
echo "Poll interval:   ${POLL_INTERVAL}s"
echo "Full-model:      warmup=$WARMUP_RUNS  timed_runs=$TIMED_RUNS"
echo "Replay tiers:    $TIER_ARGS  NCCL=${NCCL_FLAG:-no}"
echo "Skip full-model: $SKIP_FULL_MODEL"
echo "Skip replay:     $SKIP_REPLAY"
[[ ${#IDLE_ARGS[@]} -gt 0 ]] && echo "Idle baseline:   ${IDLE_ARGS[*]}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Full-model inference (profile + benchmark) with power monitoring
# ══════════════════════════════════════════════════════════════════════════════
# Outputs written INTO RUN_DIR (trace paths are passed to vicuna_tp_profile.py).
# vicuna_tp_profile.py in mode=both writes:
#   --trace      → trace_rank{0,1}.json  (the suffix _rank{rank} is appended)
#   --shape-log  → shape_log_rank{0,1}.jsonl
#   --output     → full_model_benchmark.json

TRACE_STEM="$RUN_DIR/trace.json"          # ranked_path() appends _rank{N}
SHAPE_LOG_STEM="$RUN_DIR/shape_log.jsonl" # ranked_path() appends _rank{N}
FULL_MODEL_BENCHMARK_JSON="$RUN_DIR/full_model_benchmark.json"
FULL_MODEL_POWER_CSV="$RUN_DIR/full_model_power.csv"
FULL_MODEL_ENERGY_JSON="$RUN_DIR/full_model_energy.json"

# Derived paths (after vicuna_tp_profile.py appends rank suffix)
TRACE_RANK0="$RUN_DIR/trace_rank0.json"
TRACE_RANK1="$RUN_DIR/trace_rank1.json"
SHAPE_LOG_RANK0="$RUN_DIR/shape_log_rank0.jsonl"

if [[ $SKIP_FULL_MODEL -eq 0 ]]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  STAGE 1: Full-model inference + profiling (TP=2)${NC}"
    echo -e "${BLUE}  decode_tokens=$DECODE_TOKENS  (= max_new_tokens for generate())${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Step 1a: Start power logger
    echo ""
    echo -e "${GREEN}[1/3] Starting power logger${NC}"
    python3 "$POWER_LOGGER" \
        -o "$FULL_MODEL_POWER_CSV" \
        -i "$POLL_INTERVAL" &
    FM_LOGGER_PID=$!
    echo "      Logger PID: $FM_LOGGER_PID"
    sleep 5

    if ! ps -p $FM_LOGGER_PID > /dev/null 2>&1; then
        echo -e "${RED}Power logger failed to start!${NC}"
        exit 1
    fi

    # Step 1b: Run Vicuna in mode=both (trace extraction + benchmark in one pass)
    echo ""
    echo -e "${GREEN}[2/3] Running Vicuna-7B (mode=both, warmup=$WARMUP_RUNS runs=$TIMED_RUNS decode_tokens=$DECODE_TOKENS)${NC}"
    set +e
    CUDA_VISIBLE_DEVICES=0,1 torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        "$VICUNA_DIR/vicuna_tp_profile.py" \
        --mode both \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --decode-tokens "$DECODE_TOKENS" \
        --warmup "$WARMUP_RUNS" \
        --runs "$TIMED_RUNS" \
        --trace "$TRACE_STEM" \
        --shape-log "$SHAPE_LOG_STEM" \
        --output "$FULL_MODEL_BENCHMARK_JSON"
    FM_BENCH_EXIT=$?
    set -e

    # Step 1c: Stop power logger
    echo ""
    echo -e "${GREEN}[3/3] Stopping power logger${NC}"
    kill $FM_LOGGER_PID 2>/dev/null
    wait $FM_LOGGER_PID 2>/dev/null || true
    sleep 2

    if [[ ! -f "$FULL_MODEL_POWER_CSV" ]]; then
        echo -e "${RED}Error: power log not written: $FULL_MODEL_POWER_CSV${NC}"; exit 1
    fi
    FM_SAMPLES=$(wc -l < "$FULL_MODEL_POWER_CSV")
    if [[ "$FM_SAMPLES" -lt 2 ]]; then
        echo -e "${RED}Error: power log empty ($FM_SAMPLES lines)${NC}"; exit 1
    fi
    echo "      Power samples: $FM_SAMPLES"

    if [[ ! -f "$FULL_MODEL_BENCHMARK_JSON" ]]; then
        echo -e "${RED}Error: benchmark JSON not written (exit $FM_BENCH_EXIT)${NC}"; exit 1
    fi
    if [[ ! -f "$TRACE_RANK0" || ! -f "$TRACE_RANK1" ]]; then
        echo -e "${RED}Error: trace files not found (expected $TRACE_RANK0, $TRACE_RANK1)${NC}"; exit 1
    fi
    [[ $FM_BENCH_EXIT -ne 0 ]] && echo -e "${YELLOW}Warning: vicuna_tp_profile.py exited with code $FM_BENCH_EXIT${NC}"

    # Step 1d: Integrate energy from power log (timed-runs window only)
    echo ""
    echo -e "${GREEN}Integrating full-model energy from power log...${NC}"
    python3 - <<PYEOF
import json, sys
import pandas as pd

with open("$FULL_MODEL_BENCHMARK_JSON") as f:
    bench = json.load(f)

df = pd.read_csv("$FULL_MODEL_POWER_CSV", na_values=["N/A"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["system_valid"] = df["system_total_watts"].notna()
df["system_total_watts"] = df["system_total_watts"].ffill().bfill()
df["gpu_total_watts"]    = df["gpu_total_watts"].ffill().bfill()
df["cpu_watts"]          = df["cpu_watts"].ffill().bfill()

start_dt = pd.to_datetime(bench["start_timestamp"])
end_dt   = pd.to_datetime(bench["end_timestamp"])
duration_s = bench["total_duration_s"]

w = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].sort_values("timestamp").reset_index(drop=True)
if len(w) < 2:
    print(f"ERROR: only {len(w)} power samples in full-model window", file=sys.stderr)
    sys.exit(1)

def integrate_valid_aware(df_win, t_start, t_end):
    valid_idx = [i for i in range(len(df_win)) if df_win["system_valid"].iloc[i]]
    if not valid_idx:
        return float("nan")
    e = 0.0
    t = df_win["timestamp"]
    P = df_win["system_total_watts"]
    e += float(P.iloc[valid_idx[0]]) * (t.iloc[valid_idx[0]] - t_start).total_seconds()
    for k in range(len(valid_idx) - 1):
        i, j = valid_idx[k], valid_idx[k+1]
        e += (float(P.iloc[i]) + float(P.iloc[j])) / 2.0 * (t.iloc[j] - t.iloc[i]).total_seconds()
    e += float(P.iloc[valid_idx[-1]]) * (t_end - t.iloc[valid_idx[-1]]).total_seconds()
    return e

sys_j = integrate_valid_aware(w, start_dt, end_dt)
if pd.isna(sys_j):
    print("ERROR: no valid system_total_watts (need both WattsUp meters)", file=sys.stderr)
    sys.exit(1)

dt = w["timestamp"].diff().dt.total_seconds()
dt.iloc[0] = dt.iloc[1]
gpu_j = (w["gpu_total_watts"] * dt).sum()
cpu_j = (w["cpu_watts"] * dt).sum()

result = {
    "timestamp": "$TIMESTAMP",
    "decode_tokens": $DECODE_TOKENS,
    "benchmark_file": "$FULL_MODEL_BENCHMARK_JSON",
    "power_log": "$FULL_MODEL_POWER_CSV",
    "duration_s": duration_s,
    "power_samples": len(w),
    "average_power_w": {
        "system":    float(sys_j / duration_s),
        "cpu":       float(cpu_j / duration_s),
        "gpu_total": float(gpu_j / duration_s),
    },
    "energy_wh": {
        "system":    float(sys_j / 3600),
        "cpu":       float(cpu_j / 3600),
        "gpu_total": float(gpu_j / 3600),
    },
    "benchmark_stats": bench["stats"],
}
with open("$FULL_MODEL_ENERGY_JSON", "w") as f:
    json.dump(result, f, indent=2)

print(f"  System:    {result['energy_wh']['system']:.4f} Wh  ({result['average_power_w']['system']:.1f} W avg)")
print(f"  GPU total: {result['energy_wh']['gpu_total']:.4f} Wh  ({result['average_power_w']['gpu_total']:.1f} W avg)")
print(f"  Saved:     $FULL_MODEL_ENERGY_JSON")
PYEOF

    echo ""
    echo -e "${CYAN}Stage 1 outputs:${NC}"
    echo "    Trace rank0:   $TRACE_RANK0"
    echo "    Trace rank1:   $TRACE_RANK1"
    echo "    Shape log:     $SHAPE_LOG_RANK0"
    echo "    Benchmark:     $FULL_MODEL_BENCHMARK_JSON"
    echo "    Energy:        $FULL_MODEL_ENERGY_JSON"
    echo "    Power CSV:     $FULL_MODEL_POWER_CSV"

else
    # --skip-full-model: discover latest existing run directory
    echo -e "${YELLOW}Skipping Stage 1 — looking for latest existing run directory${NC}"
    LATEST_RUN_DIR=$(ls -1d "$SCRIPT_DIR/results/"*_decode*tok 2>/dev/null | tail -1)
    if [[ -z "$LATEST_RUN_DIR" ]]; then
        echo -e "${RED}No existing run directories found under results/*_decode*tok${NC}"
        echo "Run without --skip-full-model first."
        exit 1
    fi
    RUN_DIR="$LATEST_RUN_DIR"
    TRACE_RANK0="$RUN_DIR/trace_rank0.json"
    TRACE_RANK1="$RUN_DIR/trace_rank1.json"
    SHAPE_LOG_RANK0="$RUN_DIR/shape_log_rank0.jsonl"
    FULL_MODEL_BENCHMARK_JSON="$RUN_DIR/full_model_benchmark.json"
    FULL_MODEL_ENERGY_JSON="$RUN_DIR/full_model_energy.json"
    FULL_MODEL_POWER_CSV="$RUN_DIR/full_model_power.csv"
    for f in "$TRACE_RANK0" "$TRACE_RANK1" "$FULL_MODEL_BENCHMARK_JSON" "$FULL_MODEL_ENERGY_JSON"; do
        if [[ ! -f "$f" ]]; then
            echo -e "${RED}Missing expected file: $f${NC}"; exit 1
        fi
    done
    echo "  Using run dir: $RUN_DIR"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Kernel extraction from fresh traces
# ══════════════════════════════════════════════════════════════════════════════
UNIQUE_KERNELS_JSONL="$RUN_DIR/unique_kernels_compute.jsonl"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  STAGE 2: Kernel extraction from traces${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python3 "$VICUNA_DIR/extract_vicuna_kernels.py" \
    --traces "$TRACE_RANK0" "$TRACE_RANK1" \
    --output-dir "$RUN_DIR"

if [[ ! -f "$UNIQUE_KERNELS_JSONL" ]]; then
    echo -e "${RED}Error: unique_kernels_compute.jsonl not produced${NC}"; exit 1
fi
echo -e "${CYAN}Stage 2 output:${NC} $UNIQUE_KERNELS_JSONL"

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Kernel classification
# ══════════════════════════════════════════════════════════════════════════════
KERNEL_SIGNATURES_JSON="$RUN_DIR/kernel_signatures.json"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  STAGE 3: Kernel classification (tier assignment)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# shape_mapping.json is optional; use it from RUN_DIR if the profile step produced one
SHAPE_MAPPING_ARG=""
if [[ -f "$RUN_DIR/shape_mapping.json" ]]; then
    SHAPE_MAPPING_ARG="--shapes $RUN_DIR/shape_mapping.json"
elif [[ -f "$VICUNA_DIR/shape_mapping.json" ]]; then
    SHAPE_MAPPING_ARG="--shapes $VICUNA_DIR/shape_mapping.json"
fi

python3 "$CLASSIFY_SCRIPT" \
    --input  "$UNIQUE_KERNELS_JSONL" \
    --output "$KERNEL_SIGNATURES_JSON" \
    $SHAPE_MAPPING_ARG

if [[ ! -f "$KERNEL_SIGNATURES_JSON" ]]; then
    echo -e "${RED}Error: kernel_signatures.json not produced${NC}"; exit 1
fi
echo -e "${CYAN}Stage 3 output:${NC} $KERNEL_SIGNATURES_JSON"

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Kernel replay benchmark with power monitoring
# ══════════════════════════════════════════════════════════════════════════════
ISOLATED_TIMING_JSON="$RUN_DIR/isolated_kernels_timing.json"
REPLAY_POWER_CSV="$RUN_DIR/replay_power.csv"

if [[ $SKIP_REPLAY -eq 0 ]]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  STAGE 4: Kernel replay benchmark with power monitoring${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Step 4a: Start power logger
    echo ""
    echo -e "${GREEN}[1/3] Starting power logger${NC}"
    python3 "$POWER_LOGGER" \
        -o "$REPLAY_POWER_CSV" \
        -i "$POLL_INTERVAL" &
    KR_LOGGER_PID=$!
    echo "      Logger PID: $KR_LOGGER_PID"
    sleep 5

    if ! ps -p $KR_LOGGER_PID > /dev/null 2>&1; then
        echo -e "${RED}Power logger failed to start!${NC}"
        exit 1
    fi

    # Step 4b: Run kernel replay
    echo ""
    echo -e "${GREEN}[2/3] Running kernel replay (tiers=$TIER_ARGS  NCCL=${NCCL_FLAG:-no})${NC}"

    # Shape log for Tier 3 shapes (rank0 log from Stage 1)
    SHAPE_LOG_ARG=""
    if [[ -f "$SHAPE_LOG_RANK0" ]]; then
        SHAPE_LOG_ARG="--shape-log $SHAPE_LOG_RANK0"
    fi

    set +e
    if [[ -n "$NCCL_FLAG" ]]; then
        echo "      Launcher: torchrun --nproc_per_node=2"
        torchrun --nproc_per_node=2 \
            "$VICUNA_DIR/kernel_replay_benchmark.py" \
            --kernels "$KERNEL_SIGNATURES_JSON" \
            $SHAPE_LOG_ARG \
            --output-dir "$RUN_DIR" \
            --tiers $TIER_ARGS \
            $NCCL_FLAG
    else
        echo "      Launcher: python3 (single GPU)"
        python3 "$VICUNA_DIR/kernel_replay_benchmark.py" \
            --kernels "$KERNEL_SIGNATURES_JSON" \
            $SHAPE_LOG_ARG \
            --output-dir "$RUN_DIR" \
            --tiers $TIER_ARGS
    fi
    KR_BENCH_EXIT=$?
    set -e

    # Step 4c: Stop power logger
    echo ""
    echo -e "${GREEN}[3/3] Stopping power logger${NC}"
    kill $KR_LOGGER_PID 2>/dev/null
    wait $KR_LOGGER_PID 2>/dev/null || true
    sleep 2

    if [[ ! -f "$REPLAY_POWER_CSV" ]]; then
        echo -e "${RED}Error: replay power log not written: $REPLAY_POWER_CSV${NC}"; exit 1
    fi
    KR_SAMPLES=$(wc -l < "$REPLAY_POWER_CSV")
    if [[ "$KR_SAMPLES" -lt 2 ]]; then
        echo -e "${RED}Error: replay power log empty ($KR_SAMPLES lines)${NC}"; exit 1
    fi
    echo "      Power samples: $KR_SAMPLES"

    # kernel_replay_benchmark.py writes to output-dir/isolated_kernels_timing.json
    if [[ ! -f "$ISOLATED_TIMING_JSON" ]]; then
        echo -e "${RED}Error: isolated_kernels_timing.json not written (exit $KR_BENCH_EXIT)${NC}"; exit 1
    fi
    [[ $KR_BENCH_EXIT -ne 0 ]] && echo -e "${YELLOW}Warning: replay exited with code $KR_BENCH_EXIT${NC}"

    echo ""
    echo -e "${CYAN}Stage 4 outputs:${NC}"
    echo "    Timing JSON: $ISOLATED_TIMING_JSON"
    echo "    Power CSV:   $REPLAY_POWER_CSV"

else
    # --skip-replay: expect the files already exist in RUN_DIR
    echo -e "${YELLOW}Skipping Stage 4 — expecting existing replay files in $RUN_DIR${NC}"
    if [[ ! -f "$ISOLATED_TIMING_JSON" ]]; then
        echo -e "${RED}No isolated_kernels_timing.json in $RUN_DIR${NC}"; exit 1
    fi
    if [[ ! -f "$REPLAY_POWER_CSV" ]]; then
        # Also accept legacy name from old runs
        REPLAY_POWER_CSV=$(latest_file "$RUN_DIR/unified_power_*.csv" 2>/dev/null || true)
        if [[ -z "$REPLAY_POWER_CSV" ]]; then
            echo -e "${RED}No replay_power.csv or unified_power_*.csv in $RUN_DIR${NC}"; exit 1
        fi
    fi
    echo "  Using timing: $ISOLATED_TIMING_JSON"
    echo "  Using power:  $REPLAY_POWER_CSV"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Energy comparison analysis
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  STAGE 5: Energy comparison analysis${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python3 "$SCRIPTS_DIR/analyze_energy_comparison.py" \
    --isolated-timing      "$ISOLATED_TIMING_JSON" \
    --isolated-power       "$REPLAY_POWER_CSV" \
    --full-model-energy    "$FULL_MODEL_ENERGY_JSON" \
    --full-model-benchmark "$FULL_MODEL_BENCHMARK_JSON" \
    --output-dir           "$RUN_DIR" \
    "${IDLE_ARGS[@]+"${IDLE_ARGS[@]}"}"

COMPARISON_REPORT="$RUN_DIR/energy_comparison_report.json"
echo -e "${CYAN}Stage 5 output:${NC} $COMPARISON_REPORT"

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        Experiment Complete                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Run directory: $RUN_DIR"
echo ""
echo "Key output files:"
if [[ $SKIP_FULL_MODEL -eq 0 ]]; then
    echo "  [Stage 1] Trace rank0:   $TRACE_RANK0"
    echo "  [Stage 1] Trace rank1:   $TRACE_RANK1"
    echo "  [Stage 1] Benchmark:     $FULL_MODEL_BENCHMARK_JSON"
    echo "  [Stage 1] Energy:        $FULL_MODEL_ENERGY_JSON"
    echo "  [Stage 1] Power CSV:     $FULL_MODEL_POWER_CSV"
fi
echo "  [Stage 2] Kernels JSONL: $UNIQUE_KERNELS_JSONL"
echo "  [Stage 3] Signatures:    $KERNEL_SIGNATURES_JSON"
if [[ $SKIP_REPLAY -eq 0 ]]; then
    echo "  [Stage 4] Timing JSON:   $ISOLATED_TIMING_JSON"
    echo "  [Stage 4] Power CSV:     $REPLAY_POWER_CSV"
fi
echo "  [Stage 5] Comparison:    $COMPARISON_REPORT"
echo ""
echo "To re-run analysis only (e.g. after capturing idle baseline):"
echo "  python3 scripts/analyze_energy_comparison.py \\"
echo "    --isolated-timing      $ISOLATED_TIMING_JSON \\"
echo "    --isolated-power       $REPLAY_POWER_CSV \\"
echo "    --full-model-energy    $FULL_MODEL_ENERGY_JSON \\"
echo "    --full-model-benchmark $FULL_MODEL_BENCHMARK_JSON \\"
echo "    --idle-json results/idle_power_stats.json \\"
echo "    --output-dir           $RUN_DIR"
