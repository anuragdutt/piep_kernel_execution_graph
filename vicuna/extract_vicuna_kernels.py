#!/usr/bin/env python3
"""Extract unique kernels (compute + communication) from Vicuna multi-rank traces.

This script processes Chrome trace JSON files from tensor-parallel Vicuna profiling,
classifies kernels into compute vs communication categories, and extracts unique
kernel signatures for C++ replay (including NCCL AllReduce and other collectives).

Outputs per rank:
  - kernel_launch_params_rank{rank}_gpu{gpu}.jsonl
  - kernel_launch_params_rank{rank}_gpu{gpu}.csv
  - kernels_rank{rank}_gpu{gpu}.csv (with category classification)

Combined outputs:
  - unique_kernels_compute.jsonl (all unique kernels with counts)
  - unique_kernels_compute.csv
  - unique_kernels_compute_summary.json
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def load_kernels(trace_path: str) -> List[Dict]:
    """Load CUDA kernel events from Chrome trace JSON."""
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
    """Check if kernel is a communication kernel (NCCL)."""
    lname = name.lower()
    return (
        "nccl" in lname
        or "allreduce" in lname
        or "allgather" in lname
        or "broadcast" in lname
    )


def classify_streams(
    kernels: Iterable[Dict], comm_threshold: float = 0.5
) -> Dict[str, str]:
    """Classify streams as 'compute' or 'communication' based on time spent in comm kernels."""
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
    """Get category of kernel based on stream classification."""
    if kernel["stream"] == "":
        return "unknown"
    return stream_labels.get(kernel["stream"], "compute")


def write_kernel_launch_params(
    kernels: List[Dict], out_csv: str, out_jsonl: str
) -> None:
    """Write kernel launch parameters to CSV and JSONL."""
    fields = [
        "name",
        "device",
        "stream",
        "grid_x",
        "grid_y",
        "grid_z",
        "block_x",
        "block_y",
        "block_z",
        "shared_memory",
        "registers_per_thread",
        "blocks_per_sm",
        "warps_per_sm",
        "est_achieved_occupancy",
        "correlation",
        "external_id",
        "queued",
        "cbid",
        "collective_name",
        "group_size",
        "in_msg_nelems",
        "out_msg_nelems",
        "process_group_name",
        "process_group_ranks",
        "dtype",
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


def write_kernel_counts(
    path: str, counts: Dict[Tuple[str, str], Counter], stream_labels: Dict[str, str]
) -> None:
    """Write kernel count statistics with category classification."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "category", "kernel_name", "count"])
        for (device, category), counter in counts.items():
            for name, count in counter.most_common():
                writer.writerow([device, category, name, count])


def extract_unique_kernels(
    kernels: List[Dict], stream_labels: Dict[str, str]
) -> Tuple[List[Dict], Dict]:
    """Extract unique kernels (compute + communication) with invocation counts."""
    compute_sig_fields = ["device", "stream", "dtype", "grid", "block", "shared memory"]
    comm_sig_fields = ["Collective name", "In msg nelems", "Out msg nelems", "dtype", "Group size"]

    unique: Dict[Tuple, Dict] = {}
    totals = {"total": 0, "compute": 0, "communication": 0, "unique": 0}

    for k in kernels:
        totals["total"] += 1
        name = k.get("name", "")
        category = kernel_category(k, stream_labels)
        is_comm = category == "communication" or is_comm_kernel(name)

        if is_comm:
            totals["communication"] += 1
        else:
            totals["compute"] += 1

        args_dict = k.get("args", {})
        device = k.get("device", "")

        if is_comm:
            sig = {"name": name}
            for key in comm_sig_fields:
                if key in args_dict:
                    sig[key] = args_dict[key]
        else:
            sig = {"name": name}
            for key in compute_sig_fields:
                if key in args_dict:
                    sig[key] = args_dict[key]

        sig_json = json.dumps(
            sig, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        # For communication kernels, uniqueness is (collective, nelems, group_size, dtype), not device.
        # Both ranks run the same collective; we merge counts so one unique kernel per config.
        if is_comm:
            key = (name, sig_json)
        else:
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

    totals["unique"] = len(unique)
    return list(unique.values()), totals


def process_rank(
    rank: int, trace_path: str, output_dir: str
) -> Tuple[List[Dict], Dict[str, str]]:
    """Process a single rank's trace file."""
    print(f"\n{'=' * 60}")
    print(f"Processing Rank {rank}: {trace_path}")
    print(f"{'=' * 60}")

    kernels = load_kernels(trace_path)
    print(f"  Loaded {len(kernels)} CUDA kernel events")

    stream_labels = classify_streams(kernels)
    compute_streams = sum(1 for cat in stream_labels.values() if cat == "compute")
    comm_streams = sum(1 for cat in stream_labels.values() if cat == "communication")
    print(f"  Streams: {compute_streams} compute, {comm_streams} communication")

    # Group kernels by device
    devices = sorted(set(k["device"] for k in kernels if k["device"]))
    for device in devices:
        device_kernels = [k for k in kernels if k["device"] == device]

        # Write launch params
        params_csv = os.path.join(
            output_dir, f"kernel_launch_params_rank{rank}_gpu{device}.csv"
        )
        params_jsonl = os.path.join(
            output_dir, f"kernel_launch_params_rank{rank}_gpu{device}.jsonl"
        )
        write_kernel_launch_params(device_kernels, params_csv, params_jsonl)
        print(f"  GPU {device}: Wrote {params_jsonl}")

        # Count kernels by category
        counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        for k in device_kernels:
            cat = kernel_category(k, stream_labels)
            counts[(device, cat)][k["name"]] += 1

        counts_csv = os.path.join(output_dir, f"kernels_rank{rank}_gpu{device}.csv")
        write_kernel_counts(counts_csv, counts, stream_labels)
        print(f"  GPU {device}: Wrote {counts_csv}")

    return kernels, stream_labels


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Vicuna kernels from multi-rank traces."
    )
    parser.add_argument(
        "--traces", nargs="+", required=True, help="Trace JSON files (one per rank)"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_kernels = []
    all_stream_labels = {}

    # Process each rank (trace_path order = rank 0, 1, ...)
    for rank, trace_path in enumerate(args.traces):
        if not os.path.exists(trace_path):
            print(f"ERROR: Trace file not found: {trace_path}", file=sys.stderr)
            return 1

        kernels, stream_labels = process_rank(rank, trace_path, args.output_dir)
        all_kernels.extend(kernels)  # merge both ranks for unique-kernel extraction
        # Merge stream labels with rank prefix to avoid collisions (used only for reference)
        for stream, label in stream_labels.items():
            all_stream_labels[f"rank{rank}_{stream}"] = label

    # Extract unique kernels across all ranks (compute + communication)
    # all_kernels = rank0 events + rank1 events; counts are summed across both ranks.
    # Comm kernels: merged by (name, collective, nelems, group_size) so one entry per config.
    # Compute kernels: kept per (name, device, sig) so GPU0 and GPU1 entries stay separate.
    print(f"\n{'=' * 60}")
    print("Extracting unique kernels (compute + communication)")
    print(f"{'=' * 60}")

    # Rebuild stream labels for merged kernels
    merged_stream_labels = classify_streams(all_kernels)
    unique_kernels, totals = extract_unique_kernels(all_kernels, merged_stream_labels)

    print(f"  Total kernels: {totals['total']}")
    print(f"  Compute: {totals['compute']}  Communication: {totals['communication']}")
    print(f"  Unique kernels: {totals['unique']}")

    # Write combined unique kernels
    out_jsonl = os.path.join(args.output_dir, "unique_kernels_compute.jsonl")
    out_csv = os.path.join(args.output_dir, "unique_kernels_compute.csv")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in unique_kernels:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name,device,count,signature_json\n")
        for item in unique_kernels:
            sig_json = json.dumps(item["signature"], sort_keys=True)
            f.write(
                f"{json.dumps(item['name'])},{item['device']},{item['count']},{json.dumps(sig_json)}\n"
            )

    # Write summary
    summary_path = os.path.join(args.output_dir, "unique_kernels_compute_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "traces": args.traces,
                "output_dir": args.output_dir,
                "totals": totals,
                "unique_count": len(unique_kernels),
                "top_10_kernels": sorted(
                    unique_kernels, key=lambda x: x["count"], reverse=True
                )[:10],
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"\n{'=' * 60}")
    print("Output files:")
    print(f"{'=' * 60}")
    print(f"  {out_jsonl}")
    print(f"  {out_csv}")
    print(f"  {summary_path}")

    # Print top 10 kernels by invocation count
    print(f"\n{'=' * 60}")
    print("Top 10 kernels by invocation count:")
    print(f"{'=' * 60}")
    for i, item in enumerate(
        sorted(unique_kernels, key=lambda x: x["count"], reverse=True)[:10], 1
    ):
        print(f"{i:2d}. {item['name'][:60]:60s} count={item['count']:6d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
