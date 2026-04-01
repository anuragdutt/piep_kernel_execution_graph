#!/bin/bash
# run_dataset_collection.sh — Collect kernel dataset for regression training
#
# For each model (Vicuna 7B, 13B, 33B):
#   1. Profile with TP=2, tensor cores OFF, 1 decode token, with_modules=True
#   2. Extract module-kernel mappings with I/O bytes
#   3. Classify kernels into tiers
#   4. Replay each kernel with NVML energy measurement
#   5. Build per-model CSV
#
# Final output: kernel_dataset.csv combining all models
#
# Usage:
#   ./run_dataset_collection.sh                         # all 3 models
#   ./run_dataset_collection.sh --models 7b             # just 7B
#   ./run_dataset_collection.sh --models 7b 13b         # 7B + 13B
#   ./run_dataset_collection.sh --skip-profile          # skip profiling, re-use traces
#   ./run_dataset_collection.sh --skip-replay           # skip replay, re-use timing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VICUNA_DIR="$SCRIPT_DIR/vicuna"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
CLASSIFY_SCRIPT="$SCRIPTS_DIR/classify_kernels.py"
DATASET_BUILDER="$SCRIPTS_DIR/build_kernel_dataset.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Defaults ──────────────────────────────────────────────────────────────────
MODELS_REQUESTED=()
SKIP_PROFILE=0
SKIP_REPLAY=0
APPEND=0
WARMUP_RUNS=2
TIMED_RUNS=100
PROMPT="Explain tensor parallelism in one paragraph."
BASE_OUTPUT_DIR="$SCRIPT_DIR/results/dataset_${TIMESTAMP}"

# ── Model configs ─────────────────────────────────────────────────────────────
declare -A MODEL_IDS
MODEL_IDS[7b]="lmsys/vicuna-7b-v1.5"
MODEL_IDS[13b]="lmsys/vicuna-13b-v1.5"
MODEL_IDS[33b]="lmsys/vicuna-33b-v1.3"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS_REQUESTED+=("$1")
                shift
            done
            ;;
        --skip-profile)   SKIP_PROFILE=1; shift ;;
        --skip-replay)    SKIP_REPLAY=1;  shift ;;
        --append)         APPEND=1;       shift ;;
        --warmup)         WARMUP_RUNS="$2"; shift 2 ;;
        --runs)           TIMED_RUNS="$2";  shift 2 ;;
        --output-dir)     BASE_OUTPUT_DIR="$2"; shift 2 ;;
        --prompt)         PROMPT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--models 7b 13b 33b] [--skip-profile] [--skip-replay] [--append]" >&2
            exit 1 ;;
    esac
done

# Default: all 3 models
if [[ ${#MODELS_REQUESTED[@]} -eq 0 ]]; then
    MODELS_REQUESTED=(7b 13b 33b)
fi

mkdir -p "$BASE_OUTPUT_DIR"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Banner ────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║       Kernel Dataset Collection — Bottom-Up Regression             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "Output directory:  $BASE_OUTPUT_DIR"
echo "Models:            ${MODELS_REQUESTED[*]}"
echo "Skip profile:      $SKIP_PROFILE"
echo "Skip replay:       $SKIP_REPLAY"
echo "Warmup:            $WARMUP_RUNS"
echo ""

# Track per-model outputs for final CSV assembly
MK_FILES=()
TIMING_FILES=()
MODEL_NAMES=()

# ══════════════════════════════════════════════════════════════════════════════
# Process each model
# ══════════════════════════════════════════════════════════════════════════════
for MODEL_KEY in "${MODELS_REQUESTED[@]}"; do
    MODEL_ID="${MODEL_IDS[$MODEL_KEY]}"
    if [[ -z "$MODEL_ID" ]]; then
        echo -e "${RED}Unknown model key: $MODEL_KEY (valid: 7b, 13b, 33b)${NC}"
        exit 1
    fi

    RUN_DIR="$BASE_OUTPUT_DIR/$MODEL_KEY"
    mkdir -p "$RUN_DIR"

    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Model: $MODEL_ID${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"

    TRACE_STEM="$RUN_DIR/trace.json"
    SHAPE_LOG_STEM="$RUN_DIR/shape_log.jsonl"
    MODULE_IO_STEM="$RUN_DIR/module_io_log.jsonl"
    TRACE_RANK0="$RUN_DIR/trace_rank0.json"
    TRACE_RANK1="$RUN_DIR/trace_rank1.json"
    MODULE_IO_RANK0="$RUN_DIR/module_io_log_rank0.jsonl"
    MODULE_IO_RANK1="$RUN_DIR/module_io_log_rank1.jsonl"
    UNIQUE_KERNELS_JSONL="$RUN_DIR/unique_kernels_compute.jsonl"
    MODULE_KERNELS_JSONL="$RUN_DIR/module_kernels.jsonl"
    KERNEL_SIGNATURES_JSON="$RUN_DIR/kernel_signatures.json"
    ISOLATED_TIMING_JSON="$RUN_DIR/isolated_kernels_timing.json"

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Profile with dataset mode
    # ══════════════════════════════════════════════════════════════════════
    if [[ $SKIP_PROFILE -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}[STEP 1] Profiling $MODEL_KEY (TP=2, dataset-mode, tensor cores OFF)${NC}"

        CUDA_VISIBLE_DEVICES=0,1 torchrun \
            --nproc_per_node=2 \
            --master_port=29500 \
            "$VICUNA_DIR/vicuna_tp_profile.py" \
            --mode profile \
            --model "$MODEL_ID" \
            --prompt "$PROMPT" \
            --dataset-mode \
            --warmup "$WARMUP_RUNS" \
            --trace "$TRACE_STEM" \
            --shape-log "$SHAPE_LOG_STEM" \
            --module-io-log "$MODULE_IO_STEM"

        if [[ ! -f "$TRACE_RANK0" || ! -f "$TRACE_RANK1" ]]; then
            echo -e "${RED}Error: trace files not found (expected $TRACE_RANK0, $TRACE_RANK1)${NC}"
            exit 1
        fi
        echo -e "${CYAN}  Traces saved: $TRACE_RANK0, $TRACE_RANK1${NC}"
    else
        echo -e "${YELLOW}[STEP 1] Skipped — re-using existing traces in $RUN_DIR${NC}"
        if [[ ! -f "$TRACE_RANK0" ]]; then
            echo -e "${RED}Error: no existing trace at $TRACE_RANK0${NC}"; exit 1
        fi
    fi

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Extract kernels (standard + module-aware)
    # ══════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${GREEN}[STEP 2] Extracting kernels (standard + module-aware)${NC}"

    # Build module-io-logs args (only pass files that exist)
    IO_LOG_ARGS=""
    if [[ -f "$MODULE_IO_RANK0" || -f "$MODULE_IO_RANK1" ]]; then
        IO_LOG_ARGS="--module-io-logs"
        [[ -f "$MODULE_IO_RANK0" ]] && IO_LOG_ARGS="$IO_LOG_ARGS $MODULE_IO_RANK0"
        [[ -f "$MODULE_IO_RANK1" ]] && IO_LOG_ARGS="$IO_LOG_ARGS $MODULE_IO_RANK1"
    fi

    python3 "$VICUNA_DIR/extract_vicuna_kernels.py" \
        --traces "$TRACE_RANK0" "$TRACE_RANK1" \
        --output-dir "$RUN_DIR" \
        --dataset-mode \
        $IO_LOG_ARGS

    if [[ ! -f "$MODULE_KERNELS_JSONL" ]]; then
        echo -e "${RED}Error: module_kernels.jsonl not produced${NC}"; exit 1
    fi
    echo -e "${CYAN}  Module kernels: $MODULE_KERNELS_JSONL${NC}"

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Classify kernels into tiers
    # ══════════════════════════════════════════════════════════════════════
    echo ""
    echo -e "${GREEN}[STEP 3] Classifying kernels${NC}"

    python3 "$CLASSIFY_SCRIPT" \
        --input "$UNIQUE_KERNELS_JSONL" \
        --output "$KERNEL_SIGNATURES_JSON"

    if [[ ! -f "$KERNEL_SIGNATURES_JSON" ]]; then
        echo -e "${RED}Error: kernel_signatures.json not produced${NC}"; exit 1
    fi
    echo -e "${CYAN}  Signatures: $KERNEL_SIGNATURES_JSON${NC}"

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Replay kernels with NVML energy measurement
    # ══════════════════════════════════════════════════════════════════════
    if [[ $SKIP_REPLAY -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}[STEP 4] Replaying kernels (NVML energy, tiers 1 2 3)${NC}"

        SHAPE_LOG_ARG=""
        if [[ -f "$RUN_DIR/shape_log_rank0.jsonl" ]]; then
            SHAPE_LOG_ARG="--shape-log $RUN_DIR/shape_log_rank0.jsonl"
        fi

        python3 "$VICUNA_DIR/kernel_replay_benchmark.py" \
            --kernels "$KERNEL_SIGNATURES_JSON" \
            $SHAPE_LOG_ARG \
            --output-dir "$RUN_DIR" \
            --tiers 1 2 3

        if [[ ! -f "$ISOLATED_TIMING_JSON" ]]; then
            echo -e "${RED}Error: isolated_kernels_timing.json not produced${NC}"; exit 1
        fi
        echo -e "${CYAN}  Timing: $ISOLATED_TIMING_JSON${NC}"
    else
        echo -e "${YELLOW}[STEP 4] Skipped — re-using existing timing in $RUN_DIR${NC}"
        if [[ ! -f "$ISOLATED_TIMING_JSON" ]]; then
            echo -e "${RED}Error: no existing timing at $ISOLATED_TIMING_JSON${NC}"; exit 1
        fi
    fi

    # Track for final assembly
    MK_FILES+=("$MODULE_KERNELS_JSONL")
    TIMING_FILES+=("$ISOLATED_TIMING_JSON")
    MODEL_NAMES+=("$MODEL_ID")

    echo ""
    echo -e "${CYAN}Completed $MODEL_KEY: $RUN_DIR${NC}"
done

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Build combined dataset CSV
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Building combined kernel dataset CSV${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

DATASET_CSV="$BASE_OUTPUT_DIR/kernel_dataset.csv"

APPEND_FLAG=""
if [[ "$APPEND" == "1" ]]; then
    APPEND_FLAG="--append"
fi

python3 "$DATASET_BUILDER" \
    --module-kernels "${MK_FILES[@]}" \
    --timing "${TIMING_FILES[@]}" \
    --model "${MODEL_NAMES[@]}" \
    --output "$DATASET_CSV" \
    $APPEND_FLAG

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    Dataset Collection Complete                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output directory: $BASE_OUTPUT_DIR"
echo ""
echo "Per-model outputs:"
for MODEL_KEY in "${MODELS_REQUESTED[@]}"; do
    echo "  $MODEL_KEY/:"
    echo "    trace_rank{0,1}.json      — Chrome traces"
    echo "    module_io_log_rank{0,1}.jsonl — Module I/O bytes"
    echo "    module_kernels.jsonl      — Module-kernel pairs"
    echo "    kernel_signatures.json    — Classified kernels"
    echo "    isolated_kernels_timing.json — Replay timing + NVML energy"
done
echo ""
echo "Combined dataset:"
echo "  $DATASET_CSV"
echo ""
echo "CSV columns: model, module_name, kernel_name, invocation_count,"
echo "             input_bytes, output_bytes, gpu_energy_j, latency_us, system_energy_j"
