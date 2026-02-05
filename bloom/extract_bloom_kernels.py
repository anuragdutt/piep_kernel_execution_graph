#!/usr/bin/env python3
"""Extract unique compute kernels from BLOOM trace.json (single GPU run).

Outputs:
  - bloom_kernel_launch_params.jsonl
  - bloom_kernel_launch_params.csv
  - bloom_kernel_counts.csv
  - bloom_unique_kernels_compute.jsonl
  - bloom_unique_kernels_compute.csv
  - bloom_unique_kernels_compute_summary.json
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

# Reuse functions from plot_kernel_gantt.py
def load_kernels(trace_path: str) -> List[Dict]:
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])

    def is_cuda_kernel(ev: Dict) -> bool:
        if ev.get("ph") != "X":
            return False
        cat = str(ev.get("cat", "")).lower()
        name = str(ev.get("name", "")).lower()
        if "cuda" in cat or "kernel" in cat:
            return True
        if "cuda" in name or "kernel" in name:
            return True
        args = ev.get("args", {})
        return "stream" in args and "device" in args

    kernels: List[Dict] = []
    for ev in events:
        if not is_cuda_kernel(ev):
            continue
        args = ev.get("args", {})
        kernels.append(
            {
                "name": ev.get("name", ""),
                "ts": float(ev.get("ts", 0.0)),
                "dur": float(ev.get("dur", 0.0)),
                "device": str(args.get("device", "")),
                "stream": str(args.get("stream", "")),
                "args": args,
            }
        )

    kernels.sort(key=lambda k: k["ts"])
    return kernels


def is_comm_kernel(name: str) -> bool:
    lname = name.lower()
    return "nccl" in lname or "allreduce" in lname


def classify_streams(kernels: Iterable[Dict], comm_threshold: float) -> Dict[str, str]:
    stream_totals: Dict[str, Dict[str, float]] = {}
    for k in kernels:
        stream = k["stream"]
        if stream == "":
            continue
        stream_totals.setdefault(stream, {"comm": 0.0, "total": 0.0})
        stream_totals[stream]["total"] += k["dur"]
        if is_comm_kernel(k["name"]):
            stream_totals[stream]["comm"] += k["dur"]

    stream_labels: Dict[str, str] = {}
    for stream, totals in stream_totals.items():
        comm_frac = totals["comm"] / (totals["total"] or 1.0)
        if comm_frac >= comm_threshold:
            stream_labels[stream] = "communication"
        else:
            stream_labels[stream] = "compute"

    return stream_labels


def kernel_category(kernel: Dict, stream_labels: Dict[str, str]) -> str:
    if kernel["stream"] == "":
        return "unknown"
    return stream_labels.get(kernel["stream"], "compute")


def write_kernel_launch_params(kernels: List[Dict], out_csv: str, out_jsonl: str) -> None:
    fields = [
        "name", "device", "stream",
        "grid_x", "grid_y", "grid_z",
        "block_x", "block_y", "block_z",
        "shared_memory", "registers_per_thread",
        "blocks_per_sm", "warps_per_sm", "est_achieved_occupancy",
        "correlation", "external_id", "queued", "cbid",
        "collective_name", "group_size", "in_msg_nelems", "out_msg_nelems",
        "process_group_name", "process_group_ranks", "dtype",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for k in kernels:
            args: Dict[str, Any] = k.get("args", {})
            grid = args.get("grid", [None, None, None])
            block = args.get("block", [None, None, None])
            row = [
                k.get("name", ""),
                k.get("device", ""),
                k.get("stream", ""),
                grid[0] if len(grid) > 0 else None,
                grid[1] if len(grid) > 1 else None,
                grid[2] if len(grid) > 2 else None,
                block[0] if len(block) > 0 else None,
                block[1] if len(block) > 1 else None,
                block[2] if len(block) > 2 else None,
                args.get("shared memory"),
                args.get("registers per thread"),
                args.get("blocks per SM"),
                args.get("warps per SM"),
                args.get("est. achieved occupancy %"),
                args.get("correlation"),
                args.get("External id"),
                args.get("queued"),
                args.get("cbid"),
                args.get("Collective name"),
                args.get("Group size"),
                args.get("In msg nelems"),
                args.get("Out msg nelems"),
                args.get("Process Group Name"),
                args.get("Process Group Ranks"),
                args.get("dtype"),
            ]
            writer.writerow(row)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for k in kernels:
            f.write(json.dumps(k, ensure_ascii=True) + "\n")


def write_kernel_counts(path: str, counts: Dict[Tuple[str, str], Counter]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "category", "kernel_name", "count"])
        for (device, category), counter in counts.items():
            for name, count in counter.most_common():
                writer.writerow([device, category, name, count])


# Extract unique kernels (from extract_unique_kernels.py)
def is_excluded(name: str, exclude_patterns: Iterable[str]) -> bool:
    lname = name.lower()
    for pat in exclude_patterns:
        if pat in lname:
            return True
    return False


def build_signature(name: str, args: Dict, sig_fields: Iterable[str]) -> Dict:
    """Build a normalized signature for uniqueness."""
    sig = {}
    for key in sig_fields:
        if key in args:
            sig[key] = args[key]
    
    lname = name.lower()
    is_memop = "memcpy" in lname or "memset" in lname or "memmove" in lname
    
    if is_memop:
        if "bytes" in args:
            sig["bytes"] = args["bytes"]
    else:
        sig["name"] = name
    
    return sig


def canonical_signature(sig: Dict) -> str:
    return json.dumps(sig, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def extract_unique_kernels(
    kernels: List[Dict],
    compute_kernel_names: set,
    exclude_patterns: List[str],
    sig_fields: List[str],
    device_filter: set,
) -> Tuple[Dict, Dict]:
    """Extract unique compute kernels (excluding NCCL, memory if not compute)."""
    unique: Dict[Tuple[str, str, str], Dict] = {}
    totals = {"seen": 0, "kept": 0, "excluded": 0, "device_filtered": 0}

    for k in kernels:
        totals["seen"] += 1
        name = k.get("name", "")
        device = k.get("device", "")
        args_dict = k.get("args", {}) or {}

        if device_filter and device not in device_filter:
            totals["device_filtered"] += 1
            continue

        if name not in compute_kernel_names:
            totals["excluded"] += 1
            continue

        if is_excluded(name, exclude_patterns):
            totals["excluded"] += 1
            continue

        totals["kept"] += 1
        sig = build_signature(name, args_dict, sig_fields)
        sig_json = canonical_signature(sig)
        key = (name, device, sig_json)
        if key not in unique:
            unique[key] = {
                "name": name,
                "device": device,
                "signature": sig,
                "args": args_dict,
                "count": 1,
            }
        else:
            unique[key]["count"] += 1

    return unique, totals


def main() -> int:
    trace_path = "trace.json"
    device_id = "0"
    comm_threshold = 0.5
    exclude_patterns = ["nccl", "allreduce"]
    sig_fields = ["device", "stream", "dtype", "grid", "block", "shared memory"]

    print(f"Loading kernels from {trace_path}...")
    kernels = load_kernels(trace_path)
    if not kernels:
        print("ERROR: No CUDA kernels found in trace.", file=sys.stderr)
        return 1
    
    print(f"Found {len(kernels)} CUDA kernel events.")

    # Classify streams
    stream_labels = classify_streams(kernels, comm_threshold)
    
    # Build category counts
    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for k in kernels:
        category = kernel_category(k, stream_labels)
        counts[(k["device"] or "unknown", category)][k["name"]] += 1

    # Write kernel launch params
    params_csv = f"bloom_kernel_launch_params_gpu{device_id}.csv"
    params_jsonl = f"bloom_kernel_launch_params_gpu{device_id}.jsonl"
    write_kernel_launch_params(kernels, params_csv, params_jsonl)
    print(f"Wrote {params_csv}")
    print(f"Wrote {params_jsonl}")

    # Write kernel counts
    counts_path = f"bloom_kernel_counts.csv"
    write_kernel_counts(counts_path, counts)
    print(f"Wrote {counts_path}")

    # Extract compute kernel names from counts CSV
    compute_kernel_names = set()
    with open(counts_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("category") == "compute":
                compute_kernel_names.add(row.get("kernel_name", ""))

    # Extract unique kernels
    print(f"\nExtracting unique compute kernels...")
    unique, totals = extract_unique_kernels(
        kernels,
        compute_kernel_names,
        exclude_patterns,
        sig_fields,
        device_filter={device_id},
    )

    # Write outputs
    out_jsonl = "bloom_unique_kernels_compute.jsonl"
    out_csv = "bloom_unique_kernels_compute.csv"

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in unique.values():
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name,device,count,signature_json\n")
        for (name, device, sig_json), item in unique.items():
            f.write(f"{json.dumps(name)},{device},{item['count']},{json.dumps(sig_json)}\n")

    summary_path = "bloom_unique_kernels_compute_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trace": trace_path,
                "device": device_id,
                "sig_fields": sig_fields,
                "totals": totals,
                "unique_count": len(unique),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"\nWrote {out_jsonl}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_path}")
    print(f"\nSummary:")
    print(json.dumps({"unique_count": len(unique), **totals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
