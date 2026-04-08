#!/usr/bin/env python3
"""
feature_importance_analysis.py — Random Forest regression + feature importance
for each boundedness group in consolidated_metrics.csv.

For each of the three boundedness groups (compute_bound, memory_bound, memory_op):
  - Filters rows to those with a complete feature set (no NaN in any feature or target).
  - Fits a RandomForestRegressor with GridSearchCV to find best hyperparameters.
  - Produces three horizontal-bar feature importance plots:
      1. Gini / MDI importance  (built-in RF .feature_importances_)
      2. SHAP values            (TreeExplainer, mean |SHAP| per feature)
      3. Permutation importance (sklearn permutation_importance on held-out test set)
  - Saves each plot as a PNG in the same directory as the input CSV.

Output files (9 total):
  consolidated_metrics_fi_<group>_gini.png
  consolidated_metrics_fi_<group>_shap.png
  consolidated_metrics_fi_<group>_permutation.png

Usage:
  python scripts/feature_importance_analysis.py \\
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

FEATURES = [
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

VALID_GROUPS = {"compute_bound", "memory_bound", "memory_op"}

# Grid search parameter space
PARAM_GRID = {
    "rf__n_estimators":      [100, 300],
    "rf__max_depth":         [None, 10, 20],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf":  [1, 2],
    "rf__max_features":      ["sqrt", 0.5],
}

TEST_SIZE   = 0.2
RANDOM_SEED = 42
N_PERM_REPEATS = 30   # repeats for permutation importance
CV_FOLDS    = 5
N_SHAP_BACKGROUND = 100   # background samples for SHAP TreeExplainer

# Plot style
BAR_COLOR_MAP = {
    "compute_bound": "#2196F3",
    "memory_bound":  "#FF5722",
    "memory_op":     "#4CAF50",
}
IMPORTANCE_COLORS = {
    "gini":        "#5C6BC0",
    "shap":        "#00897B",
    "permutation": "#E53935",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    order = np.argsort(importances)          # ascending → bottom of chart = least important
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

    # Annotate bar values
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


# ── Per-group analysis ────────────────────────────────────────────────────────

def analyse_group(
    group_name: str,
    df_group: pd.DataFrame,
    available_features: list,
    out_dir: str,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
    random_seed: int = RANDOM_SEED,
) -> None:
    """Fit RF, run grid search, and produce all three importance plots for one group."""

    label = group_name.replace("_", " ").title()
    color = BAR_COLOR_MAP.get(group_name, "#607D8B")
    prefix = os.path.join(out_dir, f"consolidated_metrics_fi_{group_name}")

    X = df_group[available_features].values
    y = df_group[TARGET].values
    feat_names = available_features

    print(f"\n{'='*64}")
    print(f"  Group: {label}  ({len(df_group)} rows, {len(feat_names)} features)")
    print(f"{'='*64}")

    if len(df_group) < 20:
        print(f"  SKIP: fewer than 20 clean rows — not enough data to fit a model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    # ── Grid search ──────────────────────────────────────────────────────────
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

    best_params = {k.replace("rf__", ""): v for k, v in gs.best_params_.items() if k.startswith("rf__")}
    train_r2    = gs.best_score_
    test_r2     = gs.score(X_test, y_test)

    print(f"  Best params : {best_params}")
    print(f"  CV R²       : {train_r2:.4f}")
    print(f"  Test R²     : {test_r2:.4f}")

    best_rf: RandomForestRegressor = gs.best_estimator_.named_steps["rf"]
    scaler:  StandardScaler        = gs.best_estimator_.named_steps["scaler"]

    # ── 1. Gini / MDI importance ─────────────────────────────────────────────
    print("  Computing Gini importance…")
    gini_imp = best_rf.feature_importances_

    _horizontal_bar_plot(
        importances=gini_imp,
        feature_names=feat_names,
        title=f"Gini (MDI) Feature Importance\n{label}  |  Test R²={test_r2:.3f}",
        xlabel="Mean Decrease in Impurity (Gini coefficient)",
        color=IMPORTANCE_COLORS["gini"],
        out_path=f"{prefix}_gini.png",
    )

    # ── 2. SHAP values ───────────────────────────────────────────────────────
    print("  Computing SHAP values…")
    # Scale the full training set the same way the pipeline does
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Use a random subsample as background to keep wall time manageable
    rng = np.random.default_rng(random_seed)
    n_bg = min(N_SHAP_BACKGROUND, len(X_train_scaled))
    bg_idx = rng.choice(len(X_train_scaled), size=n_bg, replace=False)
    background = X_train_scaled[bg_idx]

    explainer   = shap.TreeExplainer(best_rf, data=background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_scaled, check_additivity=False)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    _horizontal_bar_plot(
        importances=mean_abs_shap,
        feature_names=feat_names,
        title=f"SHAP Feature Importance (mean |SHAP|)\n{label}  |  Test R²={test_r2:.3f}",
        xlabel="Mean |SHAP value| (impact on model output magnitude)",
        color=IMPORTANCE_COLORS["shap"],
        out_path=f"{prefix}_shap.png",
    )

    # ── 3. Permutation importance ────────────────────────────────────────────
    print(f"  Computing permutation importance ({N_PERM_REPEATS} repeats)…")
    # Run on the scaled test set using the RF directly (post-scaling)
    perm = permutation_importance(
        best_rf,
        X_test_scaled,
        y_test,
        n_repeats=N_PERM_REPEATS,
        random_state=random_seed,
        n_jobs=-1,
        scoring="r2",
    )

    perm_mean = perm.importances_mean
    perm_std  = perm.importances_std

    _horizontal_bar_plot(
        importances=perm_mean,
        feature_names=feat_names,
        title=f"Permutation Feature Importance\n{label}  |  Test R²={test_r2:.3f}",
        xlabel="Mean decrease in R² when feature is permuted",
        color=IMPORTANCE_COLORS["permutation"],
        out_path=f"{prefix}_permutation.png",
        errors=perm_std,
    )

    print(f"  Done — 3 plots written for {label}.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Random Forest feature importance analysis per boundedness group "
            "in consolidated_metrics.csv.  Produces 9 PNG plots."
        )
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to consolidated_metrics.csv"
    )
    parser.add_argument(
        "--test-size", type=float, default=TEST_SIZE,
        help=f"Fraction of data for test split (default: {TEST_SIZE})"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=CV_FOLDS,
        help=f"Cross-validation folds for grid search (default: {CV_FOLDS})"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    args = parser.parse_args()

    test_size   = args.test_size
    cv_folds    = args.cv_folds
    random_seed = args.seed

    csv_path = os.path.abspath(args.csv)
    out_dir  = os.path.dirname(csv_path)

    if not os.path.isfile(csv_path):
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Restrict to valid boundedness values
    df = df[df["boundedness"].isin(VALID_GROUPS)].copy()
    print(f"Rows with valid boundedness {sorted(VALID_GROUPS)}: {len(df)}")

    # Check which feature columns actually exist in the CSV
    available_features = [f for f in FEATURES if f in df.columns]
    missing_features   = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print(f"WARNING: {len(missing_features)} requested features not in CSV, skipping them:")
        for f in missing_features:
            print(f"  {f}")

    if TARGET not in df.columns:
        print(f"ERROR: target column '{TARGET}' not in CSV", file=sys.stderr)
        return 1

    required_cols = available_features + [TARGET, "boundedness"]

    print(f"\nFeatures used : {len(available_features)}")
    print(f"Target        : {TARGET}")
    print(f"Output dir    : {out_dir}")

    # Per-group analysis
    groups_processed = 0
    for group in sorted(VALID_GROUPS):
        df_group_raw = df[df["boundedness"] == group][required_cols]
        df_group     = df_group_raw.dropna()

        dropped = len(df_group_raw) - len(df_group)
        print(f"\n[{group}]  {len(df_group_raw)} total rows → "
              f"{dropped} dropped (NaN) → {len(df_group)} clean rows")

        if len(df_group) < 20:
            print(f"  Skipping (fewer than 20 clean rows after NaN filter).")
            continue

        analyse_group(group, df_group, available_features, out_dir,
                      test_size=test_size, cv_folds=cv_folds, random_seed=random_seed)
        groups_processed += 1

    print(f"\n{'='*64}")
    print(f"Analysis complete.  {groups_processed} groups processed.")
    expected = groups_processed * 3
    print(f"Expected output files: {expected} PNGs in {out_dir}")
    print("="*64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
