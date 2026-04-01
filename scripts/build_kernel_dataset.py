#!/usr/bin/env python3
"""Build kernel dataset CSV from module_kernels.jsonl + isolated_kernels_timing.json.

Joins module-level kernel extraction data (module_name, kernel_name, invocation_count,
input_bytes, output_bytes) with replay benchmark results (gpu_energy, latency,
system_energy) to produce a single CSV for regression training.

Usage:
    python scripts/build_kernel_dataset.py \
        --module-kernels results/<run>/module_kernels.jsonl \
        --timing results/<run>/isolated_kernels_timing.json \
        --model vicuna-7b-v1.5 \
        --output results/<run>/kernel_dataset.csv

    # Combine multiple runs:
    python scripts/build_kernel_dataset.py \
        --module-kernels results/7b/module_kernels.jsonl results/13b/module_kernels.jsonl \
        --timing results/7b/isolated_kernels_timing.json results/13b/isolated_kernels_timing.json \
        --model vicuna-7b-v1.5 vicuna-13b-v1.5 \
        --output kernel_dataset.csv
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def load_module_kernels(path: str) -> List[Dict]:
    """Load module_kernels.jsonl."""
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def load_timing(path: str) -> Dict[str, Dict]:
    """Load isolated_kernels_timing.json and build lookup by kernel name + signature."""
    with open(path) as f:
        data = json.load(f)

    lookup: Dict[str, Dict] = {}
    for kernel in data.get("kernels", []):
        name = kernel.get("name", "")
        # Use kernel name as primary key. If multiple entries share a name
        # (different signatures), keep the first — the replay uses the same
        # timing for the same kernel name anyway.
        if name not in lookup:
            lookup[name] = kernel
    return lookup


def build_dataset(
    module_kernels: List[Dict],
    timing_lookup: Dict[str, Dict],
    model_name: str,
) -> List[Dict]:
    """Join module kernel data with timing/energy measurements.

    Each row = one (model, module, kernel) tuple.
    """
    rows: List[Dict] = []
    matched = 0
    unmatched = 0

    for mk in module_kernels:
        kernel_name = mk["kernel_name"]
        invocation_count = mk["invocation_count"]

        # Look up timing/energy from replay benchmark
        timing = timing_lookup.get(kernel_name)

        if timing is not None:
            latency_us = timing.get("single_time_us", 0.0)
            gpu_energy_mj = timing.get("gpu_energy_per_exec_mj", 0.0)
            system_energy_mj = timing.get("system_energy_per_exec_mj", 0.0)
            matched += 1
        else:
            latency_us = 0.0
            gpu_energy_mj = 0.0
            system_energy_mj = 0.0
            unmatched += 1

        # Convert mJ to J for the dataset
        gpu_energy_j = gpu_energy_mj / 1000.0
        system_energy_j = system_energy_mj / 1000.0

        rows.append({
            "model": model_name,
            "module_name": mk["module_name"],
            "kernel_name": kernel_name,
            "invocation_count": invocation_count,
            "input_bytes": mk.get("input_bytes", 0),
            "output_bytes": mk.get("output_bytes", 0),
            "gpu_energy_j": gpu_energy_j,
            "latency_us": latency_us,
            "system_energy_j": system_energy_j,
        })

    print(f"  Matched: {matched}/{matched + unmatched} kernels to timing data")
    if unmatched > 0:
        print(f"  Unmatched: {unmatched} kernels (timing=0, likely excluded APIs)")

    return rows


CSV_COLUMNS = [
    "model",
    "module_name",
    "kernel_name",
    "invocation_count",
    "input_bytes",
    "output_bytes",
    "gpu_energy_j",
    "latency_us",
    "system_energy_j",
]


def write_csv(rows: List[Dict], output_path: str) -> None:
    """Write dataset rows to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV written: {output_path} ({len(rows)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build kernel dataset CSV from module_kernels + timing."
    )
    parser.add_argument(
        "--module-kernels",
        nargs="+",
        required=True,
        help="module_kernels.jsonl file(s) from extract_vicuna_kernels.py --dataset-mode",
    )
    parser.add_argument(
        "--timing",
        nargs="+",
        required=True,
        help="isolated_kernels_timing.json file(s) from kernel_replay_benchmark.py",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="Model name(s), one per --module-kernels / --timing pair",
    )
    parser.add_argument(
        "--output",
        default="kernel_dataset.csv",
        help="Output CSV path (default: kernel_dataset.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting",
    )
    args = parser.parse_args()

    if len(args.module_kernels) != len(args.timing):
        print(
            f"ERROR: --module-kernels ({len(args.module_kernels)}) and "
            f"--timing ({len(args.timing)}) must have same number of files",
            file=sys.stderr,
        )
        return 1

    # Expand model names to match file count
    if len(args.model) == 1 and len(args.module_kernels) > 1:
        args.model = args.model * len(args.module_kernels)
    elif len(args.model) != len(args.module_kernels):
        print(
            f"ERROR: --model ({len(args.model)}) must match --module-kernels count "
            f"({len(args.module_kernels)}) or be a single value",
            file=sys.stderr,
        )
        return 1

    all_rows: List[Dict] = []

    for mk_path, timing_path, model_name in zip(
        args.module_kernels, args.timing, args.model
    ):
        print(f"\nProcessing model={model_name}")
        print(f"  Module kernels: {mk_path}")
        print(f"  Timing:         {timing_path}")

        module_kernels = load_module_kernels(mk_path)
        timing_lookup = load_timing(timing_path)
        rows = build_dataset(module_kernels, timing_lookup, model_name)
        all_rows.extend(rows)

    # Write output
    if args.append and os.path.exists(args.output):
        # Append mode: read existing, add new rows
        existing: List[Dict] = []
        with open(args.output, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
        all_rows = existing + all_rows
        print(f"\nAppending to existing ({len(existing)} rows) + new ({len(all_rows) - len(existing)} rows)")

    write_csv(all_rows, args.output)

    # Summary
    print(f"\n{'=' * 60}")
    print("Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Total rows:    {len(all_rows)}")
    models = set(r["model"] for r in all_rows)
    for model in sorted(models):
        model_rows = [r for r in all_rows if r["model"] == model]
        modules = set(r["module_name"] for r in model_rows)
        kernels = set(r["kernel_name"] for r in model_rows)
        print(f"  {model}: {len(model_rows)} rows, {len(modules)} modules, {len(kernels)} unique kernels")

    return 0


if __name__ == "__main__":
    sys.exit(main())
