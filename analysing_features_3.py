"""
diagnose_features.py
====================
Comprehensive diagnostics on the full feature matrix.
Run after `feature_engineering.py` to verify everything is sensible
and to produce summary tables for the methodology writeup.
"""

import pandas as pd
import numpy as np

from load_data import load_panel
from feature_engineering import (
    build_features_panel,
    FEATURE_GROUPS,
)


ENERGY = ["cl1s", "ho1s", "rb1s", "ng1s"]
ANCHOR = "cl1s"
SIGNAL_START = pd.Timestamp("2020-01-03")
SIGNAL_END = pd.Timestamp("2022-06-30")


def lookup_group(feature_name: str) -> str:
    """Return which feature group a column belongs to."""
    for grp, feats in FEATURE_GROUPS.items():
        if feature_name in feats:
            return grp
    return "UNKNOWN"


# ----------------------------------------------------------------------
# Section 1 — Full-sample feature statistics
# ----------------------------------------------------------------------

def per_feature_stats(feats: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One row per feature with summary stats over the full sample."""
    rows = []
    n_total = len(feats)
    for col in feats.columns:
        s = feats[col]
        n_valid = s.notna().sum()
        rows.append({
            "ticker": ticker,
            "feature": col,
            "group": lookup_group(col),
            "n_valid": n_valid,
            "pct_nan": 100 * (1 - n_valid / n_total),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p25": s.quantile(0.25),
            "p50": s.quantile(0.50),
            "p75": s.quantile(0.75),
            "max": s.max(),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Section 2 — Signal-period coverage
# ----------------------------------------------------------------------

def signal_period_coverage(feats_dict: dict) -> pd.DataFrame:
    """
    For each instrument, count how many rows in the signal period have
    *complete* feature vectors (no NaN), and identify problem features.
    """
    rows = []
    for tk in ENERGY:
        df_sig = feats_dict[tk].loc[SIGNAL_START:SIGNAL_END]
        n_rows = len(df_sig)
        n_complete = df_sig.dropna().shape[0]
        
        # Identify features with any NaN in the signal period
        nan_per_feature = df_sig.isna().sum()
        problematic = nan_per_feature[nan_per_feature > 0].sort_values(ascending=False)
        worst_offender = problematic.index[0] if len(problematic) else "(none)"
        worst_count = int(problematic.iloc[0]) if len(problematic) else 0
        
        rows.append({
            "ticker": tk,
            "signal_period_rows": n_rows,
            "complete_rows": n_complete,
            "pct_complete": 100 * n_complete / max(n_rows, 1),
            "features_with_nan": len(problematic),
            "worst_offender": worst_offender,
            "worst_offender_nans": worst_count,
        })
    return pd.DataFrame(rows)


def features_with_nan_in_signal_period(feats_dict: dict) -> pd.DataFrame:
    """For each problematic feature, show NaN counts across all 4 instruments."""
    rows = []
    all_features = list(feats_dict[ENERGY[0]].columns)
    for col in all_features:
        nans = {}
        for tk in ENERGY:
            df_sig = feats_dict[tk].loc[SIGNAL_START:SIGNAL_END]
            if col in df_sig.columns:
                nans[tk] = int(df_sig[col].isna().sum())
            else:
                nans[tk] = -1
        total = sum(v for v in nans.values() if v > 0)
        if total > 0:
            row = {"feature": col, "group": lookup_group(col), "total_nans": total}
            row.update(nans)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("total_nans", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Section 3 — Cross-instrument feature comparison
# ----------------------------------------------------------------------

def cross_instrument_summary(feats_dict: dict) -> pd.DataFrame:
    """
    For each feature, show its mean and std across all 4 instruments
    (restricted to signal period). Useful to spot which features vary
    a lot between instruments (good) vs are nearly identical (less useful).
    """
    rows = []
    all_features = list(feats_dict[ENERGY[0]].columns)
    for col in all_features:
        per_inst_means = {}
        per_inst_stds = {}
        for tk in ENERGY:
            df_sig = feats_dict[tk].loc[SIGNAL_START:SIGNAL_END]
            if col in df_sig.columns:
                per_inst_means[tk] = df_sig[col].mean()
                per_inst_stds[tk] = df_sig[col].std()
        means = pd.Series(per_inst_means)
        rows.append({
            "feature": col,
            "group": lookup_group(col),
            "mean_across_inst": means.mean(),
            "std_across_inst": means.std(),    # how much the mean varies between instruments
            "cl1s_mean": per_inst_means.get("cl1s", np.nan),
            "ho1s_mean": per_inst_means.get("ho1s", np.nan),
            "rb1s_mean": per_inst_means.get("rb1s", np.nan),
            "ng1s_mean": per_inst_means.get("ng1s", np.nan),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Section 4 — Feature correlation analysis (within-instrument)
# ----------------------------------------------------------------------

def top_correlated_pairs(feats: pd.DataFrame, ticker: str,
                         top_n: int = 20, threshold: float = 0.95) -> pd.DataFrame:
    """
    Find feature pairs with absolute correlation above `threshold`.
    Useful for identifying redundancy (relevant for Part 4 cluster analysis).
    """
    # Use signal period to avoid pre-2020 zero-fill artifacts
    df_sig = feats.loc[SIGNAL_START:SIGNAL_END]
    
    # Skip non-numeric or constant columns
    valid = [c for c in df_sig.columns
             if df_sig[c].notna().any() and df_sig[c].std() > 0]
    
    corr = df_sig[valid].corr().abs()
    # Upper triangle only (no self-correlations, no duplicates)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    corr_upper = corr.where(mask)
    
    pairs = (corr_upper.stack()
                       .sort_values(ascending=False)
                       .head(top_n)
                       .reset_index())
    pairs.columns = ["feature_1", "feature_2", "abs_corr"]
    pairs["group_1"] = pairs["feature_1"].apply(lookup_group)
    pairs["group_2"] = pairs["feature_2"].apply(lookup_group)
    pairs["same_group"] = pairs["group_1"] == pairs["group_2"]
    pairs["ticker"] = ticker
    
    return pairs[pairs["abs_corr"] >= threshold]


# ----------------------------------------------------------------------
# Section 5 — Per-group feature count and coverage
# ----------------------------------------------------------------------

def group_summary(feats: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Summary statistics aggregated to the group level."""
    df_sig = feats.loc[SIGNAL_START:SIGNAL_END]
    rows = []
    for grp, cols in FEATURE_GROUPS.items():
        cols_present = [c for c in cols if c in df_sig.columns]
        if not cols_present:
            continue
        sub = df_sig[cols_present]
        rows.append({
            "ticker": ticker,
            "group": grp,
            "n_features": len(cols_present),
            "rows_with_no_nan_in_group": sub.dropna().shape[0],
            "total_rows": len(sub),
            "pct_complete": 100 * sub.dropna().shape[0] / max(len(sub), 1),
            "mean_pct_nan_per_feature": 100 * sub.isna().mean().mean(),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)
    
    print("=" * 70)
    print("LOADING DATA AND BUILDING FEATURES")
    print("=" * 70)
    panel, primary_signals = load_panel(
        ohlcv_path="ohlcv_data.csv",
        signals_path="primary_signals.csv",
    )
    
    import time
    t0 = time.time()
    feats_dict = build_features_panel(
        panel=panel,
        primary_signals=primary_signals,
        asset_class_tickers=ENERGY,
        anchor_ticker=ANCHOR,
        train_end_date=pd.Timestamp("2019-12-31"),
        fit_regime_models=True,
    )
    print(f"\nFeatures built in {time.time() - t0:.1f}s")
    print()
    
    # ------------------------------------------------------------------
    # SECTION 1 — Per-feature stats for each instrument (saved to CSV)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SECTION 1 — Per-feature stats over full sample")
    print("=" * 70)
    all_stats = []
    for tk in ENERGY:
        stats = per_feature_stats(feats_dict[tk], tk)
        all_stats.append(stats)
    full_stats = pd.concat(all_stats, ignore_index=True)
    full_stats.to_csv("diag_per_feature_stats.csv", index=False)
    print(f"Saved {len(full_stats)} rows to diag_per_feature_stats.csv")
    print("\nSample (first 8 features for cl1s):")
    print(full_stats[full_stats["ticker"] == "cl1s"].head(8).round(4)
                                                    .to_string(index=False))
    
    # ------------------------------------------------------------------
    # SECTION 2 — Signal-period coverage
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 2 — Signal-period coverage")
    print("=" * 70)
    coverage = signal_period_coverage(feats_dict)
    print(coverage.round(2).to_string(index=False))
    
    print("\nFeatures with NaN values during signal period:")
    nan_features = features_with_nan_in_signal_period(feats_dict)
    if nan_features.empty:
        print("  (none — every feature is fully warmed up by the signal period)")
    else:
        print(nan_features.to_string(index=False))
        nan_features.to_csv("diag_nan_features.csv", index=False)
    
    # ------------------------------------------------------------------
    # SECTION 3 — Cross-instrument feature comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 3 — Cross-instrument mean comparison")
    print("=" * 70)
    cross = cross_instrument_summary(feats_dict)
    cross.to_csv("diag_cross_instrument.csv", index=False)
    print(f"Saved {len(cross)} rows to diag_cross_instrument.csv\n")
    
    print("Features whose means vary most across instruments (interesting):")
    cross_sorted = cross.dropna(subset=["std_across_inst"]).sort_values(
        "std_across_inst", ascending=False
    )
    print(cross_sorted.head(10)[
        ["feature", "group", "cl1s_mean", "ho1s_mean", "rb1s_mean", "ng1s_mean"]
    ].round(4).to_string(index=False))
    
    # ------------------------------------------------------------------
    # SECTION 4 — Highly correlated feature pairs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 4 — Highly correlated feature pairs (|corr| >= 0.95)")
    print("=" * 70)
    all_pairs = []
    for tk in ENERGY:
        pairs = top_correlated_pairs(feats_dict[tk], tk,
                                      top_n=30, threshold=0.95)
        all_pairs.append(pairs)
    full_pairs = pd.concat(all_pairs, ignore_index=True)
    full_pairs.to_csv("diag_correlated_pairs.csv", index=False)
    
    if full_pairs.empty:
        print("  (no pairs above threshold)")
    else:
        print(f"Found {len(full_pairs)} pair-instrument combinations")
        print("\nTop 15 most correlated pairs (averaged across instruments):")
        avg = (full_pairs.groupby(["feature_1", "feature_2", "group_1",
                                   "group_2", "same_group"])
                          ["abs_corr"].mean()
                          .reset_index()
                          .sort_values("abs_corr", ascending=False))
        print(avg.head(15).round(3).to_string(index=False))
    
    # ------------------------------------------------------------------
    # SECTION 5 — Per-group summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 5 — Per-group coverage during signal period")
    print("=" * 70)
    all_group = []
    for tk in ENERGY:
        all_group.append(group_summary(feats_dict[tk], tk))
    group_df = pd.concat(all_group, ignore_index=True)
    group_df.to_csv("diag_group_summary.csv", index=False)
    print(group_df.round(2).to_string(index=False))
    
    # ------------------------------------------------------------------
    # SECTION 6 — Final feature count and totals
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 6 — Final feature manifest")
    print("=" * 70)
    print(f"\nTotal features: {feats_dict['cl1s'].shape[1]}")
    print(f"Sum from FEATURE_GROUPS dict: {sum(len(v) for v in FEATURE_GROUPS.values())}")
    print()
    print("Features per group:")
    for grp, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in feats_dict["cl1s"].columns]
        missing = set(cols) - set(present)
        print(f"  {grp:25s}: {len(present):>3} present  "
              f"{'(' + str(len(missing)) + ' missing: ' + str(missing) + ')' if missing else ''}")
    
    extra = set(feats_dict["cl1s"].columns) - set(
        c for cols in FEATURE_GROUPS.values() for c in cols
    )
    if extra:
        print(f"\nColumns present but not in FEATURE_GROUPS dict (orphans):")
        for c in sorted(extra):
            print(f"  {c}")
    
    print("\n" + "=" * 70)
    print("DONE — diagnostic CSVs saved:")
    print("  diag_per_feature_stats.csv     — feature-level stats per instrument")
    print("  diag_nan_features.csv          — features with NaNs in signal period")
    print("  diag_cross_instrument.csv      — cross-instrument mean comparison")
    print("  diag_correlated_pairs.csv      — highly correlated feature pairs")
    print("  diag_group_summary.csv         — per-group coverage")
    print("=" * 70)