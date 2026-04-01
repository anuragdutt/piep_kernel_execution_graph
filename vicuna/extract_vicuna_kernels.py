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
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def load_all_events(trace_path: str) -> Dict[str, List[Dict]]:
    """Load ALL event types from Chrome trace JSON, grouped by category.

    Returns dict with keys: 'kernel', 'cuda_runtime', 'cpu_op', 'python_function',
    'gpu_memcpy', 'user_annotation', etc.
    """
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for ev in events:
        cat = str(ev.get("cat", "")).lower()
        grouped[cat].append(ev)
    return grouped


def build_module_timeline(events_by_cat: Dict[str, List[Dict]]) -> List[Dict]:
    """Build sorted timeline of nn.Module events from with_modules=True trace.

    These are duration events (ph=X) in the 'python_function' category with names
    like 'nn.Module: LlamaForCausalLM' or similar. Their [ts, ts+dur] span tells
    us which CUDA kernels were launched during that module's forward pass.
    """
    module_events = []
    for ev in events_by_cat.get("python_function", []):
        name = ev.get("name", "")
        if name.startswith("nn.Module: "):
            module_name = name[len("nn.Module: "):]
            module_events.append({
                "module": module_name,
                "ts": float(ev.get("ts", 0)),
                "dur": float(ev.get("dur", 0)),
                "end_ts": float(ev.get("ts", 0)) + float(ev.get("dur", 0)),
                "args": ev.get("args", {}),
            })
    # Sort by ts ascending, then by duration descending (outer module first)
    module_events.sort(key=lambda e: (e["ts"], -e["dur"]))
    return module_events


def build_correlation_maps(
    events_by_cat: Dict[str, List[Dict]]
) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """Build lookup maps for CUDA kernel → cpu_op correlation.

    Returns:
        corr_to_runtime: correlation_id → cuda_runtime event
        extid_to_cpuop:  External_id → cpu_op event
    """
    corr_to_runtime: Dict[int, Dict] = {}
    for ev in events_by_cat.get("cuda_runtime", []):
        corr = ev.get("args", {}).get("correlation")
        if corr is not None:
            corr_to_runtime[corr] = ev

    extid_to_cpuop: Dict[int, Dict] = {}
    for ev in events_by_cat.get("cpu_op", []):
        ext_id = ev.get("args", {}).get("External id")
        if ext_id is not None:
            extid_to_cpuop[ext_id] = ev

    return corr_to_runtime, extid_to_cpuop


def _compute_io_bytes_from_cpuop(cpu_op: Optional[Dict]) -> Tuple[int, int]:
    """Compute input and output bytes from a cpu_op's Input Dims.

    Uses the operation name + input dims to estimate output dims.
    Assumes FP16 (2 bytes) unless we can detect otherwise.
    """
    if cpu_op is None:
        return 0, 0

    args = cpu_op.get("args", {})
    input_dims = args.get("Input Dims", [])
    op_name = cpu_op.get("name", "")
    dtype_bytes = 2  # FP16 default

    # Compute input bytes: sum of numel(dim) * dtype_bytes for each non-empty dim
    input_bytes = 0
    for dim in input_dims:
        if dim and isinstance(dim, list) and len(dim) > 0:
            numel = 1
            for d in dim:
                if isinstance(d, (int, float)) and d > 0:
                    numel *= int(d)
            input_bytes += numel * dtype_bytes

    # Estimate output bytes from operation semantics
    output_bytes = 0
    if op_name in ("aten::mm", "aten::matmul") and len(input_dims) >= 2:
        # [M, K] × [K, N] → [M, N]
        if input_dims[0] and input_dims[1] and len(input_dims[0]) >= 2 and len(input_dims[1]) >= 2:
            M = input_dims[0][0]
            N = input_dims[1][-1]
            output_bytes = int(M) * int(N) * dtype_bytes
    elif op_name == "aten::bmm" and len(input_dims) >= 2:
        # [B, M, K] × [B, K, N] → [B, M, N]
        if input_dims[0] and input_dims[1] and len(input_dims[0]) >= 3 and len(input_dims[1]) >= 3:
            B = input_dims[0][0]
            M = input_dims[0][1]
            N = input_dims[1][2]
            output_bytes = int(B) * int(M) * int(N) * dtype_bytes
    elif op_name == "aten::linear" and len(input_dims) >= 2:
        # [*, M, K] × [N, K] → [*, M, N]
        if input_dims[0] and input_dims[1] and len(input_dims[0]) >= 2 and len(input_dims[1]) >= 2:
            batch_dims = input_dims[0][:-1]
            N = input_dims[1][0]
            numel = 1
            for d in batch_dims:
                numel *= int(d)
            output_bytes = numel * int(N) * dtype_bytes
    elif input_dims and input_dims[0]:
        # Default: output shape ≈ first input shape (elementwise ops)
        numel = 1
        for d in input_dims[0]:
            if isinstance(d, (int, float)) and d > 0:
                numel *= int(d)
        output_bytes = numel * dtype_bytes

    return input_bytes, output_bytes


def find_module_for_kernel(
    kernel_ts: float, module_timeline: List[Dict]
) -> Optional[str]:
    """Find the innermost (most specific) module whose [ts, ts+dur] contains kernel_ts."""
    best = None
    best_dur = float("inf")
    for m in module_timeline:
        if m["ts"] <= kernel_ts <= m["end_ts"]:
            # Prefer the innermost (shortest duration) module
            if m["dur"] < best_dur:
                best = m["module"]
                best_dur = m["dur"]
    return best


def extract_module_kernels(
    trace_path: str,
    module_io_log_path: Optional[str] = None,
) -> List[Dict]:
    """Extract kernels with module attribution and I/O bytes.

    Strategy for module attribution (works WITHOUT with_modules trace events):
    1. Load module_io_log.jsonl → list of module hook invocations in chronological order
    2. Load cpu_op events from trace → list of all ATen ops with timestamps and External ids
    3. Build extid→module mapping: match cpu_ops to module hooks using timestamp ordering.
       Key insight: cpu_ops like aten::linear fire during a module's forward(), and our
       module hooks log the module_path for each invocation in the same order.
    4. For each CUDA kernel, follow correlation chain (kernel → cuda_runtime → cpu_op)
       and look up the module assignment for that cpu_op.

    Returns list of dicts: {module_name, kernel_name, invocation_count, input_bytes,
    output_bytes, signature, device}
    """
    events_by_cat = load_all_events(trace_path)
    corr_to_runtime, extid_to_cpuop = build_correlation_maps(events_by_cat)

    # Also try nn.Module events from trace (in case with_modules works)
    module_timeline = build_module_timeline(events_by_cat)

    # Load module I/O log from forward hooks
    module_io_entries: List[Dict] = []
    if module_io_log_path and os.path.exists(module_io_log_path):
        with open(module_io_log_path) as f:
            for line in f:
                module_io_entries.append(json.loads(line))

    # Build extid-to-module mapping using cpu_op containment within module hooks.
    #
    # Approach: Find all cpu_op events that have timestamps, sort by ts.
    # For each module_io entry (which fires for every module invocation in order),
    # find the cpu_op events that temporally fall within each module's [ts, ts+dur] span.
    #
    # Since we don't have explicit module ts/dur from the trace, we use nested cpu_ops:
    # 1. Find "top-level" ops for each module type (e.g., aten::linear for Linear modules)
    # 2. Match them to module_io entries by order
    # 3. All cpu_ops (and their CUDA kernels) within a top-level op's [ts, ts+dur]
    #    inherit that module's path.

    # Build sorted list of ALL cpu_ops with their External ids and timestamps
    cpu_ops_sorted = []
    for ev in events_by_cat.get("cpu_op", []):
        ext_id = ev.get("args", {}).get("External id")
        ts = ev.get("ts")
        dur = ev.get("dur", 0)
        if ext_id is not None and ts is not None:
            cpu_ops_sorted.append({
                "ext_id": ext_id,
                "ts": float(ts),
                "dur": float(dur or 0),
                "end_ts": float(ts) + float(dur or 0),
                "name": ev.get("name", ""),
                "args": ev.get("args", {}),
            })
    cpu_ops_sorted.sort(key=lambda x: (x["ts"], -x["dur"]))

    # Map External id → module path
    extid_to_module: Dict[int, str] = {}

    if module_io_entries:
        # Strategy: match cpu_ops to module hooks by finding for each module hook
        # invocation the cpu_ops that are contained within its timespan.
        #
        # Build a timeline of module spans by matching top-level ops to hook entries.
        # For each hook entry, we use the hook's module_class to find the matching
        # cpu_op pattern (e.g., Linear → aten::linear, Embedding → aten::embedding).

        MODULE_CLASS_TO_OPS = {
            "Linear": ["aten::linear"],
            "Embedding": ["aten::embedding"],
            "LlamaRMSNorm": ["aten::rsqrt", "aten::mul"],  # RMSNorm pattern
            "LayerNorm": ["aten::layer_norm", "aten::native_layer_norm"],
        }

        # Group module_io entries by their exact invocation order
        # Build spans: for each hook entry, find the cpu_op that corresponds to it
        module_spans: List[Dict] = []

        # Match aten::linear cpu_ops to Linear module hooks (most important for GEMM attribution)
        linear_entries = [e for e in module_io_entries if e["module_class"] == "Linear"]
        linear_cpu_ops = [op for op in cpu_ops_sorted if op["name"] == "aten::linear"]

        if len(linear_entries) == len(linear_cpu_ops):
            for io_entry, cpu_op in zip(linear_entries, linear_cpu_ops):
                module_spans.append({
                    "module_path": io_entry["module_path"],
                    "module_class": io_entry["module_class"],
                    "ts": cpu_op["ts"],
                    "end_ts": cpu_op["end_ts"],
                    "dur": cpu_op["dur"],
                    "input_bytes": io_entry.get("input_bytes", 0),
                    "output_bytes": io_entry.get("output_bytes", 0),
                })
        elif linear_cpu_ops:
            print(f"  WARNING: Linear hooks ({len(linear_entries)}) != aten::linear ops ({len(linear_cpu_ops)})")
            # Still try to match as many as possible
            n = min(len(linear_entries), len(linear_cpu_ops))
            for io_entry, cpu_op in zip(linear_entries[:n], linear_cpu_ops[:n]):
                module_spans.append({
                    "module_path": io_entry["module_path"],
                    "module_class": io_entry["module_class"],
                    "ts": cpu_op["ts"],
                    "end_ts": cpu_op["end_ts"],
                    "dur": cpu_op["dur"],
                    "input_bytes": io_entry.get("input_bytes", 0),
                    "output_bytes": io_entry.get("output_bytes", 0),
                })

        # Also match Embedding ops
        embed_entries = [e for e in module_io_entries if e["module_class"] == "Embedding"]
        embed_cpu_ops = [op for op in cpu_ops_sorted if op["name"] == "aten::embedding"]
        n = min(len(embed_entries), len(embed_cpu_ops))
        for io_entry, cpu_op in zip(embed_entries[:n], embed_cpu_ops[:n]):
            module_spans.append({
                "module_path": io_entry["module_path"],
                "module_class": io_entry["module_class"],
                "ts": cpu_op["ts"],
                "end_ts": cpu_op["end_ts"],
                "dur": cpu_op["dur"],
                "input_bytes": io_entry.get("input_bytes", 0),
                "output_bytes": io_entry.get("output_bytes", 0),
            })

        # Sort module spans by start time
        module_spans.sort(key=lambda x: (x["ts"], -x["dur"]))

        # Now assign every cpu_op to the innermost module span that contains it
        for op in cpu_ops_sorted:
            best_path = None
            best_dur = float("inf")
            op_ts = op["ts"]
            for span in module_spans:
                if span["ts"] <= op_ts <= span["end_ts"]:
                    if span["dur"] < best_dur:
                        best_path = span["module_path"]
                        best_dur = span["dur"]
            if best_path:
                extid_to_module[op["ext_id"]] = best_path

        print(f"  Module spans built: {len(module_spans)}")
        print(f"  CPU ops with module attribution: {len(extid_to_module)}/{len(cpu_ops_sorted)}")

    # Build module_path → I/O bytes lookup from hook data
    module_io_by_path: Dict[str, Dict] = {}
    for entry in module_io_entries:
        path = entry["module_path"]
        if path not in module_io_by_path:
            module_io_by_path[path] = {"input_bytes": 0, "output_bytes": 0, "count": 0}
        module_io_by_path[path]["input_bytes"] += entry.get("input_bytes", 0)
        module_io_by_path[path]["output_bytes"] += entry.get("output_bytes", 0)
        module_io_by_path[path]["count"] += 1

    # Process each CUDA kernel/memcpy event
    kernel_events = events_by_cat.get("kernel", [])
    gpu_memcpy_events = events_by_cat.get("gpu_memcpy", [])
    all_gpu_events = kernel_events + gpu_memcpy_events

    grouped: Dict[Tuple, Dict] = {}
    no_module_count = 0
    via_trace_module = 0
    via_hook_module = 0

    for ev in all_gpu_events:
        name = ev.get("name", "")
        args = ev.get("args", {})
        device = str(args.get("device", ""))
        kernel_ts = float(ev.get("ts", 0))

        # Try module attribution: first from trace module events, then from hook mapping
        module_name = None

        # Method 1: trace nn.Module events (from with_modules=True, if available)
        if module_timeline:
            module_name = find_module_for_kernel(kernel_ts, module_timeline)
            if module_name:
                via_trace_module += 1

        # Method 2: correlation chain → cpu_op → module (from forward hooks)
        if module_name is None:
            corr = args.get("correlation")
            if corr is not None and corr in corr_to_runtime:
                rt_ev = corr_to_runtime[corr]
                ext_id = rt_ev.get("args", {}).get("External id")
                if ext_id is not None and ext_id in extid_to_module:
                    module_name = extid_to_module[ext_id]
                    via_hook_module += 1

        if module_name is None:
            module_name = "__no_module__"
            no_module_count += 1

        # Get I/O bytes from cpu_op Input Dims
        corr = args.get("correlation")
        cpu_op = None
        if corr is not None and corr in corr_to_runtime:
            rt_ev = corr_to_runtime[corr]
            ext_id = rt_ev.get("args", {}).get("External id")
            if ext_id is not None and ext_id in extid_to_cpuop:
                cpu_op = extid_to_cpuop[ext_id]

        input_bytes, output_bytes = _compute_io_bytes_from_cpuop(cpu_op)

        # Build signature for uniqueness
        sig = {"name": name}
        for key in ["device", "stream", "grid", "block", "shared memory"]:
            if key in args:
                sig[key] = args[key]

        sig_json = json.dumps(sig, sort_keys=True, separators=(",", ":"))
        group_key = (module_name, name, device, sig_json)

        if group_key not in grouped:
            grouped[group_key] = {
                "module_name": module_name,
                "kernel_name": name,
                "device": device,
                "invocation_count": 1,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "signature": sig,
                "cpu_op_name": cpu_op.get("name", "") if cpu_op else "",
            }
        else:
            grouped[group_key]["invocation_count"] += 1
            grouped[group_key]["input_bytes"] += input_bytes
            grouped[group_key]["output_bytes"] += output_bytes

    # Override I/O bytes with hook data where available (more accurate)
    for key, entry in grouped.items():
        mod = entry["module_name"]
        if mod in module_io_by_path:
            hook_data = module_io_by_path[mod]
            if hook_data["count"] > 0 and hook_data["input_bytes"] > 0:
                avg_in = hook_data["input_bytes"] // hook_data["count"]
                avg_out = hook_data["output_bytes"] // hook_data["count"]
                entry["input_bytes"] = avg_in * entry["invocation_count"]
                entry["output_bytes"] = avg_out * entry["invocation_count"]
                entry["io_source"] = "hook"
            else:
                entry["io_source"] = "trace"
        else:
            entry["io_source"] = "trace"

    result = list(grouped.values())
    attributed = sum(1 for r in result if r["module_name"] != "__no_module__")
    print(f"  Module-kernel pairs: {len(result)} ({attributed} with module, {len(result) - attributed} unattributed)")
    print(f"  Attribution: {via_trace_module} via trace, {via_hook_module} via hooks, {no_module_count} unattributed")

    return result


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
    parser.add_argument(
        "--dataset-mode",
        action="store_true",
        help="Module-aware extraction: produce module_kernels.jsonl with per-(module,kernel) grouping",
    )
    parser.add_argument(
        "--module-io-logs",
        nargs="*",
        default=None,
        help="module_io_log_rank{N}.jsonl files from vicuna_tp_profile.py --dataset-mode",
    )
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

    # ── Dataset mode: module-aware extraction ──────────────────────────────
    if args.dataset_mode:
        print(f"\n{'=' * 60}")
        print("Module-aware kernel extraction (dataset mode)")
        print(f"{'=' * 60}")

        all_module_kernels: List[Dict] = []
        for rank, trace_path in enumerate(args.traces):
            print(f"\n  Processing rank {rank} for module-kernel mapping...")
            io_log_path = None
            if args.module_io_logs and rank < len(args.module_io_logs):
                io_log_path = args.module_io_logs[rank]
            elif args.module_io_logs is None:
                # Auto-discover from output-dir
                candidate = os.path.join(args.output_dir, f"module_io_log_rank{rank}.jsonl")
                if os.path.exists(candidate):
                    io_log_path = candidate

            mk = extract_module_kernels(trace_path, io_log_path)
            all_module_kernels.extend(mk)

        # Merge across ranks: group by (module_name, kernel_name, sig_json)
        merged: Dict[Tuple, Dict] = {}
        for entry in all_module_kernels:
            sig_json = json.dumps(entry["signature"], sort_keys=True, separators=(",", ":"))
            key = (entry["module_name"], entry["kernel_name"], sig_json)
            if key not in merged:
                merged[key] = dict(entry)  # copy
            else:
                merged[key]["invocation_count"] += entry["invocation_count"]
                merged[key]["input_bytes"] += entry["input_bytes"]
                merged[key]["output_bytes"] += entry["output_bytes"]

        module_kernels_list = list(merged.values())

        # Write module_kernels.jsonl
        mk_path = os.path.join(args.output_dir, "module_kernels.jsonl")
        with open(mk_path, "w", encoding="utf-8") as f:
            for item in module_kernels_list:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")

        print(f"\n  Module-kernel pairs (merged): {len(module_kernels_list)}")
        print(f"  Output: {mk_path}")

        # Print summary by module
        module_counts: Dict[str, int] = defaultdict(int)
        for item in module_kernels_list:
            module_counts[item["module_name"]] += item["invocation_count"]
        print(f"\n  Top 10 modules by total kernel invocations:")
        for mod, cnt in sorted(module_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {mod[:60]:60s} invocations={cnt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
