#!/usr/bin/env python3
"""
Plot Roofline — Generate a roofline chart from roofline_metrics.json.

Plots:
  - RTX A6000 roofline ceilings (FP16 compute + DRAM bandwidth)
  - Each kernel as a dot at (arithmetic_intensity, achieved_GFLOP/s)
  - Color-coded by tier, sized by invocation count

Usage:
  python scripts/plot_roofline.py \
      --roofline results/dataset_.../7b/roofline_metrics.json \
      --output results/dataset_.../7b/roofline_plot.png
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


# RTX A6000 specs (Empirical Boundaries)
PEAK_FP16_GFLOPS = 108800.0  # 108.8 TFLOPS = 108800 GFLOP/s
PEAK_BW_GBS = 590.3          # GB/s


def plot_roofline(
    kernels: List[Dict[str, Any]],
    output_path: str,
    title: str = "Kernel Roofline — RTX A6000 (TC disabled)",
) -> None:
    """Generate a roofline chart."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    
    # ── Roofline ceilings ─────────────────────────────────────────────────
    ai_range = np.logspace(-2, 5, 500)  # FLOP/byte
    
    # Bandwidth ceiling: Performance = BW × AI
    bw_ceiling = PEAK_BW_GBS * ai_range  # GFLOP/s
    
    # Compute ceiling: Performance = peak compute
    compute_ceiling = np.full_like(ai_range, PEAK_FP16_GFLOPS)
    
    # Roofline = min(BW ceiling, compute ceiling)
    roofline = np.minimum(bw_ceiling, compute_ceiling)
    
    # Ridge point
    ridge_ai = PEAK_FP16_GFLOPS / PEAK_BW_GBS
    
    ax.loglog(ai_range, roofline, "k-", linewidth=2.5, label="Roofline", zorder=5)
    ax.axvline(x=ridge_ai, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.annotate(
        f"Ridge point\nAI={ridge_ai:.1f}",
        xy=(ridge_ai, PEAK_FP16_GFLOPS),
        xytext=(ridge_ai * 3, PEAK_FP16_GFLOPS * 0.3),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray",
    )
    
    # Annotate ceilings
    ax.text(0.015, PEAK_BW_GBS * 0.015 * 0.7, 
            f"DRAM BW\n{PEAK_BW_GBS:.0f} GB/s",
            fontsize=8, color="blue", rotation=38, alpha=0.7)
    ax.text(ridge_ai * 20, PEAK_FP16_GFLOPS * 1.15,
            f"FP16 Peak: {PEAK_FP16_GFLOPS/1000:.1f} TFLOP/s",
            fontsize=9, color="red", alpha=0.7)
    
    # ── Plot kernels ──────────────────────────────────────────────────────
    tier_colors = {2: "#2196F3", 3: "#FF9800", 1: "#9E9E9E"}
    tier_labels = {2: "Tier 2 (cuBLAS GEMM)", 3: "Tier 3 (libtorch)", 1: "Tier 1 (memcpy)"}
    
    profiled = [k for k in kernels if k.get("ncu_profiled") and k.get("achieved_gflops", 0) > 0]
    
    for tier in [2, 3]:
        tier_kernels = [k for k in profiled if k.get("tier") == tier]
        if not tier_kernels:
            continue
        
        ais = [k["arithmetic_intensity"] for k in tier_kernels]
        perfs = [k["achieved_gflops"] for k in tier_kernels]
        counts = [max(1, k.get("invocation_count", 1)) for k in tier_kernels]
        
        # Size by invocation count (log scale)
        sizes = [30 + 20 * np.log1p(c) for c in counts]
        
        ax.scatter(
            ais, perfs,
            s=sizes,
            c=tier_colors.get(tier, "gray"),
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            label=f"{tier_labels.get(tier, f'Tier {tier}')} ({len(tier_kernels)})",
            zorder=10,
        )
        
        # Annotate notable kernels (top 3 by GFLOP/s per tier)
        sorted_kernels = sorted(tier_kernels, key=lambda k: k["achieved_gflops"], reverse=True)
        for k in sorted_kernels[:2]:
            short_name = k["kernel_name"].split("(")[0].split("<")[0][-30:]
            ax.annotate(
                short_name,
                xy=(k["arithmetic_intensity"], k["achieved_gflops"]),
                fontsize=6, alpha=0.6,
                xytext=(5, 5), textcoords="offset points",
            )
    
    # ── Shade regions ─────────────────────────────────────────────────────
    ax.fill_between(
        ai_range[ai_range < ridge_ai],
        0.01, PEAK_FP16_GFLOPS * 10,
        alpha=0.03, color="blue", label="_nolegend_"
    )
    ax.fill_between(
        ai_range[ai_range >= ridge_ai],
        0.01, PEAK_FP16_GFLOPS * 10,
        alpha=0.03, color="red", label="_nolegend_"
    )
    ax.text(0.02, 0.03, "Memory-bound", fontsize=11, color="blue", alpha=0.4,
            transform=ax.transAxes)
    ax.text(0.75, 0.03, "Compute-bound", fontsize=11, color="red", alpha=0.4,
            transform=ax.transAxes)
    
    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0.01, 1e5)
    ax.set_ylim(0.1, PEAK_FP16_GFLOPS * 3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.8)
    ax.grid(True, which="both", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Roofline plot saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate roofline chart")
    parser.add_argument(
        "--roofline", nargs="+", required=True,
        help="Path(s) to roofline_metrics.json"
    )
    parser.add_argument(
        "--output", default="roofline_plot.png",
        help="Output image path (default: roofline_plot.png)"
    )
    parser.add_argument(
        "--title", default=None,
        help="Chart title"
    )
    args = parser.parse_args()
    
    # Load all roofline data
    all_kernels = []
    for path in args.roofline:
        with open(path) as f:
            data = json.load(f)
        all_kernels.extend(data.get("kernels", []))
    
    print(f"Loaded {len(all_kernels)} kernel records from {len(args.roofline)} file(s)")
    
    title = args.title or "Kernel Roofline — RTX A6000 (Tensor Cores Disabled)"
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    plot_roofline(all_kernels, args.output, title=title)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
