#!/usr/bin/env python3
"""
Extract AllReduce (and other NCCL collective) specs from kernel_launch_params JSONL files.
Outputs message sizes (elements + bytes), dtype, group size, ranks, and invocation counts
for use in C++ NCCL replay.

Usage:
  python scripts/extract_allreduce_specs.py vicuna/unique_kernels_compute_final.jsonl/kernel_launch_params_rank0_gpu0.jsonl [rank1.jsonl ...]
  python scripts/extract_allreduce_specs.py vicuna/unique_kernels_compute_final.jsonl/kernel_launch_params_rank*.jsonl
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# dtype -> element size in bytes
DTYPE_BYTES = {
    "Half": 2,
    "Float": 4,
    "Float32": 4,
    "BFloat16": 2,
    "Int64": 8,
    "Int32": 4,
    "Int8": 1,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_allreduce_specs.py <kernel_launch_params_rank0.jsonl> [rank1.jsonl ...]", file=sys.stderr)
        sys.exit(1)

    # Collect all collective events (allreduce, allgather, etc.) from args
    events = []
    for path in sys.argv[1:]:
        p = Path(path)
        if not p.exists():
            print(f"Warning: {p} not found", file=sys.stderr)
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = ev.get("name", "")
                args = ev.get("args") or {}
                if "nccl" not in name.lower() and "Collective name" not in args:
                    continue
                collective = args.get("Collective name", "")
                if not collective:
                    continue
                events.append({
                    "name": name,
                    "device": args.get("device"),
                    "collective": collective,
                    "in_nelems": args.get("In msg nelems"),
                    "out_nelems": args.get("Out msg nelems"),
                    "dtype": args.get("dtype", "Half"),
                    "group_size": args.get("Group size"),
                    "pg_ranks": args.get("Process Group Ranks"),
                    "dur": ev.get("dur"),
                })

    # Unique specs: (collective, in_nelems, out_nelems, dtype, group_size, pg_ranks) -> count
    spec_counts = defaultdict(int)
    for e in events:
        key = (
            e["collective"],
            e["in_nelems"],
            e["out_nelems"],
            e["dtype"],
            e["group_size"],
            e["pg_ranks"],
        )
        spec_counts[key] += 1

    # Report unique AllReduce (and other collectives) with message size in bytes
    print("Collective specs (unique) and total invocations:\n")
    for key, count in sorted(spec_counts.items(), key=lambda x: (-x[1], x[0])):
        collective, in_nelems, out_nelems, dtype, group_size, pg_ranks = key
        esize = DTYPE_BYTES.get(dtype, 2)
        in_bytes = (in_nelems or 0) * esize
        out_bytes = (out_nelems or 0) * esize
        print(f"  Collective: {collective}")
        print(f"    In msg nelems: {in_nelems}  -> {in_bytes} bytes")
        print(f"    Out msg nelems: {out_nelems} -> {out_bytes} bytes")
        print(f"    dtype: {dtype}  (element size {esize} B)")
        print(f"    Group size: {group_size}  Ranks: {pg_ranks}")
        print(f"    Invocations (across all provided files): {count}")
        print()

    # One-line summary for C++ replay
    total_allreduce = sum(c for k, c in spec_counts.items() if k[0] == "allreduce")
    if total_allreduce:
        # For TP=2, each collective appears on both ranks, so replay count = total_events / group_size
        for key, count in spec_counts.items():
            if key[0] != "allreduce":
                continue
            _, in_nelems, _, dtype, group_size, pg_ranks = key
            esize = DTYPE_BYTES.get(dtype, 2)
            replay_count = count // (group_size or 1)  # one AllReduce call per collective step
            print("--- C++ replay summary (AllReduce) ---")
            print(f"  nelems = {in_nelems};  nbytes = {in_nelems * esize};  ncclHalf;  ncclSum;  devices = {pg_ranks};  repeat count = {replay_count} (total events in files: {count})")


if __name__ == "__main__":
    main()
