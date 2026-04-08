#!/usr/bin/env python3
"""
merge_datasets.py — Merge per-model NCU metrics CSVs with kernel_dataset_roofline.csv.

Left-joins on (model, kernel_name) and writes consolidated_metrics.csv.

Usage:
  python scripts/merge_datasets.py \\
      --roofline  results/dataset_20260327_205511/kernel_dataset_roofline.csv \\
      --ncu       /tmp/ncu_7b.csv /tmp/ncu_13b.csv /tmp/ncu_33b.csv \\
      --output    results/dataset_20260327_205511/consolidated_metrics.csv
"""

import argparse
import sys

import pandas as pd


NEW_METRIC_COLS = [
    "sm_throughput_pct",
    "dram_throughput_pct",
    "achieved_occupancy_pct",
    "warp_activity",
    "warp_eligible",
    "issue_activity",
    "ipc",
    "inst_executed",
    "mem_inst",
    "dram_read_bytes",
    "dram_write_bytes",
    "l2_throughput_pct",
    "shared_bank_conflicts",
    "fma_utilization_pct",
    "avg_warp_latency",
    "stall_mem_dep_pct",
    "stall_long_sb_pct",
    "stall_short_sb_pct",
    "stall_math_pipe_pct",
    "stall_mio_throttle_pct",
    "stall_not_selected_pct",
    "gpu_sm_count",
    "gpu_clock_mhz",
    "gpu_mem_clock_mhz",
    "gpu_mem_bandwidth_gbs",
]

MERGE_KEYS = ["model", "kernel_name"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge NCU metrics with kernel_dataset_roofline.csv"
    )
    parser.add_argument(
        "--roofline", required=True,
        help="Path to kernel_dataset_roofline.csv (base dataset)"
    )
    parser.add_argument(
        "--ncu", nargs="+", required=True,
        help="Per-model NCU metrics CSV files (output of collect_ncu_metrics.py)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for consolidated_metrics.csv"
    )
    args = parser.parse_args()

    # Load base dataset
    base = pd.read_csv(args.roofline)
    print(f"Base dataset : {len(base)} rows, {len(base.columns)} cols")

    # Load and concatenate per-model NCU results
    ncu_frames = []
    for path in args.ncu:
        df = pd.read_csv(path)
        print(f"NCU metrics  : {len(df)} rows from {path}")
        ncu_frames.append(df)
    ncu = pd.concat(ncu_frames, ignore_index=True)

    # Validate no duplicate (model, kernel_name) keys in NCU data
    dup_mask = ncu.duplicated(subset=MERGE_KEYS, keep=False)
    if dup_mask.any():
        print(f"\nWARNING: {dup_mask.sum()} duplicate rows on {MERGE_KEYS} in NCU data:")
        print(ncu[dup_mask][MERGE_KEYS].to_string(index=False))
        print("Keeping first occurrence per key.\n")
        ncu = ncu.drop_duplicates(subset=MERGE_KEYS, keep="first")

    print(f"NCU combined : {len(ncu)} unique (model, kernel_name) keys")

    # Left join
    merged = base.merge(ncu, on=MERGE_KEYS, how="left")

    # Report coverage
    total = len(merged)
    populated = merged[NEW_METRIC_COLS[0]].notna().sum()
    missing = total - populated
    print(f"\nMerge results:")
    print(f"  Total rows        : {total}")
    print(f"  Rows with metrics : {populated}")
    print(f"  Rows without      : {missing}"
          + (" (NaN — skipped/failed kernels)" if missing else ""))

    # Report unmatched base rows
    unmatched_keys = set(zip(base["model"], base["kernel_name"])) - set(zip(ncu["model"], ncu["kernel_name"]))
    if unmatched_keys:
        print(f"\nWARNING: {len(unmatched_keys)} (model, kernel_name) combinations in base "
              "have no NCU entry:")
        for m, k in sorted(unmatched_keys)[:10]:
            print(f"  {m}  |  {k[:80]}")
        if len(unmatched_keys) > 10:
            print(f"  ... and {len(unmatched_keys) - 10} more")

    # Correlation of new metrics with gpu_energy_j (sanity check)
    numeric_new = [c for c in NEW_METRIC_COLS if merged[c].dtype in ("float64", "float32")]
    if "gpu_energy_j" in merged.columns and numeric_new:
        corrs = merged[["gpu_energy_j"] + numeric_new].corr()["gpu_energy_j"].drop("gpu_energy_j")
        print(f"\nCorrelation with gpu_energy_j (top 10 by |r|):")
        top = corrs.abs().nlargest(10)
        for feat in top.index:
            print(f"  {feat:<40s}  r = {corrs[feat]:+.4f}")

    # Save
    merged.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}  ({len(merged)} rows, {len(merged.columns)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
