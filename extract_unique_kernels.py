#!/usr/bin/env python3
"""Extract unique compute kernels from PyTorch trace JSONL.

Unique kernel definition: (name, device, full args dict).
Uses compute-category CSVs to filter GPU work, excluding NCCL only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Tuple


DEFAULT_EXCLUDE = [
    "nccl",
    "allreduce",
]

# Fields that define a kernel's "input signature" for timing/energy.
# NOTE: Kernel name is included separately in build_signature() for compute kernels.
BASE_SIG_FIELDS = [
    "device",
    "stream",
    "dtype",
    "grid",
    "block",
    "shared memory",
]


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def is_excluded(name: str, exclude_patterns: Iterable[str]) -> bool:
    lname = name.lower()
    for pat in exclude_patterns:
        if pat in lname:
            return True
    return False


def build_signature(name: str, args: Dict, sig_fields: Iterable[str]) -> Dict:
    """Build a normalized signature for uniqueness.
    
    For compute kernels (non-memory-ops), the name is critical because
    different kernels with same launch config perform different ops.
    For memory ops, the name is generic (Memcpy/Memset) so we use bytes.
    """
    sig = {}
    for key in sig_fields:
        if key in args:
            sig[key] = args[key]
    
    lname = name.lower()
    is_memop = "memcpy" in lname or "memset" in lname or "memmove" in lname
    
    if is_memop:
        # Memory ops: signature is bytes transferred (name is generic like "Memcpy DtoD")
        if "bytes" in args:
            sig["bytes"] = args["bytes"]
    else:
        # Compute kernels: name is essential (different kernels = different ops)
        sig["name"] = name
    
    return sig


def canonical_signature(sig: Dict) -> str:
    return json.dumps(sig, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_compute_kernel_names(paths: Iterable[str], exclude_patterns: Iterable[str]) -> Dict[str, set]:
    """Return device->set(names) for category==compute, excluding mem/sync by name."""
    device_names: Dict[str, set] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            # Try both "kernel_name" (kernel_counts CSV) and "name" (raw CSV)
            name_field = "kernel_name" if "kernel_name" in reader.fieldnames else "name"
            
            for row in reader:
                if row.get("category") != "compute":
                    continue
                name = row.get(name_field, "")
                device = row.get("device", "")
                if is_excluded(name, exclude_patterns):
                    continue
                device_names.setdefault(device, set()).add(name)
    return device_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract unique compute kernels from JSONL.")
    parser.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated list of kernel_launch_params JSONL files.",
    )
    parser.add_argument(
        "--compute-csvs",
        required=True,
        help="Comma-separated list of kernels_rank*_gpu*.csv files (category column).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write unique kernel outputs.",
    )
    parser.add_argument(
        "--devices",
        default="0,1",
        help="Comma-separated device IDs to include.",
    )
    parser.add_argument(
        "--sig-fields",
        default=",".join(BASE_SIG_FIELDS),
        help="Comma-separated args keys to use for signature (grid/block as arrays).",
    )
    parser.add_argument(
        "--exclude",
        default=",".join(DEFAULT_EXCLUDE),
        help="Comma-separated name substrings to exclude (case-insensitive).",
    )
    args = parser.parse_args()

    inputs = parse_list(args.inputs)
    compute_csvs = parse_list(args.compute_csvs)
    output_dir = args.output_dir
    devices = set(parse_list(args.devices))
    exclude_patterns = [p.lower() for p in parse_list(args.exclude)]
    sig_fields = parse_list(args.sig_fields)

    os.makedirs(output_dir, exist_ok=True)
    compute_names_by_device = load_compute_kernel_names(compute_csvs, exclude_patterns)

    # key -> count, and sample metadata for output
    unique: Dict[Tuple[str, str, str], Dict] = {}
    totals = {"seen": 0, "kept": 0, "excluded": 0, "device_filtered": 0}

    for path in inputs:
        for event in load_jsonl(path):
            totals["seen"] += 1
            name = str(event.get("name", ""))
            args_dict = event.get("args", {}) or {}
            device = str(args_dict.get("device", event.get("device", "")))

            if devices and device not in devices:
                totals["device_filtered"] += 1
                continue

            if name not in compute_names_by_device.get(device, set()):
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

    # Write combined outputs
    out_jsonl = os.path.join(output_dir, "unique_kernels_compute.jsonl")
    out_csv = os.path.join(output_dir, "unique_kernels_compute.csv")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in unique.values():
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name,device,count,signature_json\n")
        for (name, device, sig_json), item in unique.items():
            f.write(f"{json.dumps(name)},{device},{item['count']},{json.dumps(sig_json)}\n")

    summary_path = os.path.join(output_dir, "unique_kernels_compute_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": inputs,
                "compute_csvs": compute_csvs,
                "devices": sorted(devices),
                "sig_fields": sig_fields,
                "totals": totals,
                "unique_count": len(unique),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_path}")
    print(json.dumps({"unique_count": len(unique), **totals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
