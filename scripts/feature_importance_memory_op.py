#!/usr/bin/env python3
"""
feature_importance_memory_op.py — Feature importance analysis for the memory_op
boundedness group in consolidated_metrics.csv.

memory_op rows are tier-1 memcpy kernels: all 21 NCU hardware-counter metrics are NaN
for every row in this group.  This script handles that by excluding features that are
ALL NaN across the group before fitting — rather than dropping rows that contain any NaN.

Steps:
  1. Load consolidated_metrics.csv, filter to boundedness == "memory_op".
  2. Exclude any feature column that is entirely NaN (expected: all 21 NCU metrics).
  3. Drop rows that still have NaN in surviving feature columns or the target (expected: 0).
  4. Fit RandomForestRegressor + GridSearchCV (3-fold CV for the small dataset).
  5. Produce three feature importance plots:
       consolidated_metrics_fi_memory_op_gini.png
       consolidated_metrics_fi_memory_op_shap.png
       consolidated_metrics_fi_memory_op_permutation.png

Usage:
  python scripts/feature_importance_memory_op.py \\
      --csv results/dataset_20260327_205511/consolidated_metrics.csv
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ─────────────────────────────────────────────────────────────

TARGET = "system_energy_j"

# Full candidate feature list (same as feature_importance_analysis.py).
# Features that are ALL NaN for memory_op rows will be excluded automatically.
ALL_FEATURES = [
    "invocation_count",
    "input_bytes",
    "output_bytes",
    "gpu_energy_j",
    "latency_us",
    "arithmetic_intensity",
    "achieved_gflops",
    "achieved_bw_gbs",
    "pct_peak_compute",
    "pct_peak_bw",
    "total_flops",
    "dram_bytes",
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

# Grid search parameter space (identical to feature_importance_analysis.py)
PARAM_GRID = {
    "rf__n_estimators":      [100, 300],
    "rf__max_depth":         [None, 10, 20],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf":  [1, 2],
    "rf__max_features":      ["sqrt", 0.5],
}

TEST_SIZE        = 0.2
RANDOM_SEED      = 42
CV_FOLDS         = 3    # reduced from 5 because n=18
N_PERM_REPEATS   = 30
N_SHAP_BACKGROUND = 100

GROUP_COLOR      = "#4CAF50"
IMPORTANCE_COLORS = {
    "gini":        "#5C6BC0",
    "shap":        "#00897B",
    "permutation": "#E53935",
}
MIN_CLEAN_ROWS = 5


# ── Plot helper ───────────────────────────────────────────────────────────────

def _horizontal_bar_plot(
    importances: np.ndarray,
    feature_names: list,
    title: str,
    xlabel: str,
    color: str,
    out_path: str,
    errors: np.ndarray = None,
) -> None:
    """Draw a horizontal bar chart sorted by importance magnitude and save to PNG."""
    order = np.argsort(importances)
    vals  = importances[order]
    names = [feature_names[i] for i in order]
    errs  = errors[order] if errors is not None else None

    n = len(names)
    fig_h = max(6, 0.35 * n + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, vals, color=color, alpha=0.82, height=0.7)

    if errs is not None:
        ax.errorbar(
            vals, y_pos,
            xerr=errs,
            fmt="none", color="black", capsize=3, linewidth=1.2, alpha=0.7,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, vals):
        x_txt = val + max(abs(vals)) * 0.01
        ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7.5, color="#333333")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Feature importance analysis for the memory_op boundedness group. "
            "Excludes features that are entirely NaN for this group, then fits "
            "RandomForest + GridSearchCV and produces 3 importance plots."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to consolidated_metrics.csv")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--cv-folds",  type=int,   default=CV_FOLDS)
    parser.add_argument("--seed",      type=int,   default=RANDOM_SEED)
    args = parser.parse_args()

    test_size   = args.test_size
    cv_folds    = args.cv_folds
    random_seed = args.seed

    csv_path = os.path.abspath(args.csv)
    out_dir  = os.path.dirname(csv_path)
    prefix   = os.path.join(out_dir, "consolidated_metrics_fi_memory_op")

    if not os.path.isfile(csv_path):
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        return 1

    df_all = pd.read_csv(csv_path)
    print(f"Loaded {len(df_all)} rows from {csv_path}")

    # ── Filter to memory_op ───────────────────────────────────────────────────
    df = df_all[df_all["boundedness"] == "memory_op"].copy()
    print(f"memory_op rows: {len(df)}")

    if len(df) == 0:
        print("ERROR: no rows with boundedness == 'memory_op'.", file=sys.stderr)
        return 1

    if TARGET not in df.columns:
        print(f"ERROR: target column '{TARGET}' not in CSV.", file=sys.stderr)
        return 1

    # ── Candidate features present in CSV ────────────────────────────────────
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing_cols = [f for f in ALL_FEATURES if f not in df.columns]
    if missing_cols:
        print(f"NOTE: {len(missing_cols)} features not in CSV (skipping): {missing_cols}")

    # ── Column-level NaN exclusion ────────────────────────────────────────────
    excluded = [f for f in available if df[f].isna().all()]
    surviving = [f for f in available if f not in excluded]

    print(f"\nFeature NaN analysis over {len(df)} memory_op rows:")
    print(f"  Excluded (all NaN) : {len(excluded)}")
    for f in excluded:
        print(f"    {f}")
    print(f"  Surviving          : {len(surviving)}")
    for f in surviving:
        print(f"    {f}")

    if not surviving:
        print("ERROR: no usable features remain after NaN exclusion.", file=sys.stderr)
        return 1

    # ── Row-level NaN drop (on surviving features + target) ───────────────────
    use_cols = surviving + [TARGET]
    df_clean = df[use_cols].dropna()
    dropped  = len(df) - len(df_clean)
    print(f"\nRows after row-level NaN drop: {len(df_clean)} ({dropped} dropped)")

    if len(df_clean) < MIN_CLEAN_ROWS:
        print(f"ERROR: only {len(df_clean)} clean rows — minimum is {MIN_CLEAN_ROWS}. "
              "Cannot fit a model.", file=sys.stderr)
        return 1

    if len(df_clean) < 30:
        print(f"NOTE: small dataset ({len(df_clean)} rows). "
              f"Using {cv_folds}-fold CV and interpreting results with caution.")

    X = df_clean[surviving].values
    y = df_clean[TARGET].values
    feat_names = surviving

    print(f"\n{'='*64}")
    print(f"  Group: Memory Op  ({len(df_clean)} rows, {len(feat_names)} features)")
    print(f"  CV folds: {cv_folds}  |  Test size: {test_size}  |  Seed: {random_seed}")
    print(f"{'='*64}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    # ── Grid search ───────────────────────────────────────────────────────────
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestRegressor(random_state=random_seed, n_jobs=-1)),
    ])

    print(f"  Running GridSearchCV ({cv_folds}-fold, {len(X_train)} train samples)…")
    gs = GridSearchCV(
        pipe,
        PARAM_GRID,
        cv=cv_folds,
        scoring="r2",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    best_params = {k.replace("rf__", ""): v for k, v in gs.best_params_.items()
                   if k.startswith("rf__")}
    train_r2 = gs.best_score_
    test_r2  = gs.score(X_test, y_test)

    print(f"  Best params : {best_params}")
    print(f"  CV R²       : {train_r2:.4f}")
    print(f"  Test R²     : {test_r2:.4f}")

    best_rf: RandomForestRegressor = gs.best_estimator_.named_steps["rf"]
    scaler:  StandardScaler        = gs.best_estimator_.named_steps["scaler"]

    # ── 1. Gini / MDI importance ──────────────────────────────────────────────
    print("  Computing Gini importance…")
    gini_imp = best_rf.feature_importances_

    _horizontal_bar_plot(
        importances=gini_imp,
        feature_names=feat_names,
        title=f"Gini (MDI) Feature Importance\nMemory Op  |  Test R²={test_r2:.3f}",
        xlabel="Mean Decrease in Impurity (Gini coefficient)",
        color=IMPORTANCE_COLORS["gini"],
        out_path=f"{prefix}_gini.png",
    )

    # ── 2. SHAP values ────────────────────────────────────────────────────────
    print("  Computing SHAP values…")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    rng  = np.random.default_rng(random_seed)
    n_bg = min(N_SHAP_BACKGROUND, len(X_train_scaled))
    bg_idx    = rng.choice(len(X_train_scaled), size=n_bg, replace=False)
    background = X_train_scaled[bg_idx]

    explainer   = shap.TreeExplainer(best_rf, data=background,
                                     feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_scaled, check_additivity=False)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    _horizontal_bar_plot(
        importances=mean_abs_shap,
        feature_names=feat_names,
        title=f"SHAP Feature Importance (mean |SHAP|)\nMemory Op  |  Test R²={test_r2:.3f}",
        xlabel="Mean |SHAP value| (impact on model output magnitude)",
        color=IMPORTANCE_COLORS["shap"],
        out_path=f"{prefix}_shap.png",
    )

    # ── 3. Permutation importance ─────────────────────────────────────────────
    print(f"  Computing permutation importance ({N_PERM_REPEATS} repeats)…")
    perm = permutation_importance(
        best_rf,
        X_test_scaled,
        y_test,
        n_repeats=N_PERM_REPEATS,
        random_state=random_seed,
        n_jobs=-1,
        scoring="r2",
    )

    _horizontal_bar_plot(
        importances=perm.importances_mean,
        feature_names=feat_names,
        title=f"Permutation Feature Importance\nMemory Op  |  Test R²={test_r2:.3f}",
        xlabel="Mean decrease in R² when feature is permuted",
        color=IMPORTANCE_COLORS["permutation"],
        out_path=f"{prefix}_permutation.png",
        errors=perm.importances_std,
    )

    print(f"\n{'='*64}")
    print("Analysis complete.  3 plots written for Memory Op.")
    print(f"Output directory: {out_dir}")
    print("="*64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
