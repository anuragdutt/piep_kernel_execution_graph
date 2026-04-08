#!/usr/bin/env python3
"""
Roofline Classify — Join roofline metrics with kernel_dataset.csv.

Reads roofline_metrics.json (from ncu_roofline_profiler.py) and the existing
kernel_dataset.csv, then produces kernel_dataset_roofline.csv with additional
columns for roofline classification.

Usage:
  python scripts/roofline_classify.py \
      --dataset results/dataset_.../kernel_dataset.csv \
      --roofline results/dataset_.../7b/roofline_metrics.json \
                 results/dataset_.../13b/roofline_metrics.json \
      --output results/dataset_.../kernel_dataset_roofline.csv
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def load_roofline_metrics(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load roofline metrics from one or more JSON files.
    
    Returns a dict keyed by kernel_name → roofline record.
    If multiple files provide data for the same kernel, later files overwrite.
    """
    combined: Dict[str, Dict[str, Any]] = {}
    
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        
        kernels = data.get("kernels", [])
        for k in kernels:
            name = k.get("kernel_name", "")
            if name:
                if name not in combined:
                    combined[name] = k
                else:
                    old_k = combined[name]
                    old_profiled = old_k.get("ncu_profiled", False)
                    new_profiled = k.get("ncu_profiled", False)
                    
                    if new_profiled and not old_profiled:
                        combined[name] = k
                    elif new_profiled and old_profiled:
                        if k.get("total_flops", 0) > old_k.get("total_flops", 0):
                            combined[name] = k
    
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join roofline metrics with kernel dataset"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to kernel_dataset.csv"
    )
    parser.add_argument(
        "--roofline", nargs="+", required=True,
        help="Path(s) to roofline_metrics.json (one per model)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: kernel_dataset_roofline.csv next to input)"
    )
    args = parser.parse_args()
    
    import csv
    
    # Load roofline metrics
    roofline = load_roofline_metrics(args.roofline)
    print(f"Loaded roofline data for {len(roofline)} unique kernels")
    
    # Read existing dataset
    with open(args.dataset, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    
    print(f"Loaded dataset with {len(rows)} rows, {len(fieldnames)} columns")
    
    # New columns to add
    new_columns = [
        "arithmetic_intensity",
        "achieved_gflops",
        "achieved_bw_gbs",
        "boundedness",
        "pct_peak_compute",
        "pct_peak_bw",
        "total_flops",
        "dram_bytes",
    ]
    
    out_fieldnames = fieldnames + [c for c in new_columns if c not in fieldnames]
    
    # Join
    matched = 0
    unmatched = 0
    enriched_rows = []
    
    for row in rows:
        kernel_name = row.get("kernel_name", "")
        
        if kernel_name in roofline:
            rf = roofline[kernel_name]
            row["arithmetic_intensity"] = rf.get("arithmetic_intensity", 0.0)
            row["achieved_gflops"] = rf.get("achieved_gflops", 0.0)
            row["achieved_bw_gbs"] = rf.get("achieved_bw_gbs", 0.0)
            row["boundedness"] = rf.get("boundedness", "unknown")
            row["pct_peak_compute"] = rf.get("pct_peak_compute", 0.0)
            row["pct_peak_bw"] = rf.get("pct_peak_bw", 0.0)
            row["total_flops"] = rf.get("total_flops", 0)
            row["dram_bytes"] = rf.get("dram_bytes", 0)
            matched += 1
        else:
            # No roofline data — set defaults
            row["arithmetic_intensity"] = 0.0
            row["achieved_gflops"] = 0.0
            row["achieved_bw_gbs"] = 0.0
            row["boundedness"] = "unknown"
            row["pct_peak_compute"] = 0.0
            row["pct_peak_bw"] = 0.0
            row["total_flops"] = 0
            row["dram_bytes"] = 0
            unmatched += 1
        
        enriched_rows.append(row)
    
    # Output path
    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.dataset)
        output_path = f"{base}_roofline{ext}"
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)
    
    print(f"\nJoin results:")
    print(f"  Matched:   {matched}/{len(rows)} rows")
    print(f"  Unmatched: {unmatched}/{len(rows)} rows")
    
    # Summary by boundedness
    from collections import Counter
    bound_counts = Counter(r["boundedness"] for r in enriched_rows)
    print(f"\nBoundedness distribution:")
    for b, c in sorted(bound_counts.items()):
        print(f"  {b:20s}: {c} rows")
    
    print(f"\nOutput: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
