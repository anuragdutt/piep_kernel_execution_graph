#!/usr/bin/env python3
"""Generate kernel Gantt charts and kernel counts from PyTorch traces.

Example:
  python plot_kernel_gantt.py --trace /home/adutt/kernel_profiling/trace_rank{rank}.json --ranks 0,1

Outputs (per rank, per GPU):
  - kernels_gantt_rank{rank}_gpu{device}.png
  - kernels_rank{rank}_gpu{device}.csv
  - kernels_rank{rank}_kernel_counts.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ranked_path(path: str, rank: int) -> str:
    if "{rank}" in path:
        return path.format(rank=rank)
    root, ext = os.path.splitext(path)
    return f"{root}_rank{rank}{ext}"


def with_suffix(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{path}{suffix}"


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


def write_kernel_counts(path: str, counts: Dict[Tuple[str, str], Counter]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "category", "kernel_name", "count"])
        for (device, category), counter in counts.items():
            for name, count in counter.most_common():
                writer.writerow([device, category, name, count])


def write_kernel_launch_params(kernels: List[Dict], out_csv: str, out_jsonl: str) -> None:
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


def build_kernel_gantt(
    kernels: List[Dict],
    out_png: str,
    out_csv: str,
    max_kernels: int,
    stream_labels: Dict[str, str],
) -> None:
    if max_kernels > 0:
        kernels = kernels[:max_kernels]

    if not kernels:
        raise RuntimeError("No CUDA kernels found in trace. Try enabling CUDA profiling.")

    t0 = kernels[0]["ts"]
    for k in kernels:
        k["start_ms"] = (k["ts"] - t0) / 1000.0
        k["dur_ms"] = k["dur"] / 1000.0
        k["category"] = kernel_category(k, stream_labels)

    categories = ["compute", "communication", "unknown"]
    category_to_y = {c: i for i, c in enumerate(categories)}

    # CSV output
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "start_ms", "dur_ms", "device", "stream", "category"])
        for k in kernels:
            writer.writerow(
                [
                    k["name"],
                    f"{k['start_ms']:.6f}",
                    f"{k['dur_ms']:.6f}",
                    k["device"],
                    k["stream"],
                    k["category"],
                ]
            )

    # Plot Gantt chart: y-axis by category.
    fig_h = max(4, min(0.8 * len(categories), 8))
    fig_w = 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = {
        "compute": "#2b6cb0",
        "communication": "#c53030",
        "unknown": "#718096",
    }

    for k in kernels:
        y = category_to_y[k["category"]]
        ax.barh(y, k["dur_ms"], left=k["start_ms"], height=0.8, color=colors[k["category"]])

    ax.set_yticks(list(category_to_y.values()))
    ax.set_yticklabels([f"{c} stream" for c in categories])
    ax.set_xlabel("Time (ms)")
    ax.set_title("CUDA Kernel Execution Timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot kernel Gantt charts from traces.")
    parser.add_argument("--trace", default="trace_rank{rank}.json")
    parser.add_argument("--ranks", default="0,1")
    parser.add_argument("--max-kernels", type=int, default=0, help="0 = all kernels")
    parser.add_argument(
        "--comm-threshold",
        type=float,
        default=0.5,
        help="Fraction of comm kernel time to label a stream as communication.",
    )
    args = parser.parse_args()

    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip() != ""]
    if not ranks:
        print("ERROR: no ranks provided.", file=sys.stderr)
        return 1

    for rank in ranks:
        trace_path = ranked_path(args.trace, rank)
        if not os.path.exists(trace_path):
            print(f"WARNING: missing trace for rank {rank} at {trace_path}", file=sys.stderr)
            continue

        kernels = load_kernels(trace_path)
        if not kernels:
            print(f"WARNING: no kernels found for rank {rank}", file=sys.stderr)
            continue

        stream_labels = classify_streams(kernels, comm_threshold=args.comm_threshold)
        counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

        devices = sorted({k["device"] for k in kernels if k["device"] != ""})
        if not devices:
            devices = [""]

        outputs: List[str] = []
        for device_id in devices:
            device_kernels = [k for k in kernels if k["device"] == device_id] if device_id != "" else kernels
            if not device_kernels:
                continue

            for k in device_kernels:
                category = kernel_category(k, stream_labels)
                counts[(device_id or "unknown", category)][k["name"]] += 1

            suffix = f"_gpu{device_id}" if device_id != "" else "_gpuunknown"
            out_csv = with_suffix(f"kernels_rank{rank}.csv", suffix)
            out_png = with_suffix(f"kernels_gantt_rank{rank}.png", suffix)
            build_kernel_gantt(device_kernels, out_png, out_csv, args.max_kernels, stream_labels)
            outputs.append(out_png)

            params_csv = with_suffix(f"kernel_launch_params_rank{rank}.csv", suffix)
            params_jsonl = with_suffix(f"kernel_launch_params_rank{rank}.jsonl", suffix)
            write_kernel_launch_params(device_kernels, params_csv, params_jsonl)

        counts_path = f"kernels_rank{rank}_kernel_counts.csv"
        write_kernel_counts(counts_path, counts)
        if outputs:
            print(f"Wrote rank {rank}: {', '.join(outputs)}, {counts_path}")
        else:
            print(f"Wrote rank {rank}: {counts_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
