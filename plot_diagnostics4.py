"""
plot_diagnostics.py
===================
Generate diagnostic plots from the feature matrix.

Run AFTER feature_engineering.py — needs the feature data structures in memory,
or it can re-build them from load_panel + build_features_panel.

Produces ~6 PNG figures in the working directory:
    fig_01_coverage.png         — NaN coverage per group per instrument
    fig_02_correlation_heatmap.png   — full correlation matrix (color-coded)
    fig_03_cross_instrument.png — features sorted by cross-instrument variation
    fig_04_distributions.png    — distributions of selected key features
    fig_05_signal_period_evolution.png  — time series of regime / signal features
    fig_06_group_correlations.png  — within-group correlation structure
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from load_data import load_panel
from feature_engineering import (
    build_features_panel,
    FEATURE_GROUPS,
)


ENERGY = ["cl1s", "ho1s", "rb1s", "ng1s"]
ANCHOR = "cl1s"
SIGNAL_START = pd.Timestamp("2020-01-03")
SIGNAL_END = pd.Timestamp("2022-06-30")

# A consistent color per group
GROUP_COLORS = {
    "A_returns_momentum":    "#1f77b4",
    "B_volatility":          "#ff7f0e",
    "C_microstructure":      "#2ca02c",
    "D_meanrev_trend":       "#d62728",
    "E_latent_regime":       "#9467bd",
    "F_spectral_fractal":    "#8c564b",
    "G_cross_sectional":     "#e377c2",
    "H_signal_interaction":  "#7f7f7f",
    "I_seasonality":         "#bcbd22",
}


def lookup_group(name):
    for grp, feats in FEATURE_GROUPS.items():
        if name in feats:
            return grp
    return "UNKNOWN"


# ----------------------------------------------------------------------
# Figure 1 — NaN coverage heatmap (per group per instrument)
# ----------------------------------------------------------------------

def plot_coverage(feats_dict, savepath="fig_01_coverage.png"):
    """Heatmap of % NaN per feature × instrument during the signal period."""
    all_features = list(feats_dict[ENERGY[0]].columns)
    n_feats = len(all_features)
    
    coverage = np.zeros((n_feats, len(ENERGY)))
    for j, tk in enumerate(ENERGY):
        sub = feats_dict[tk].loc[SIGNAL_START:SIGNAL_END]
        for i, col in enumerate(all_features):
            coverage[i, j] = 100 * sub[col].isna().mean()
    
    fig, ax = plt.subplots(figsize=(8, 14))
    
    # Custom colormap: white for 0% nan, red as it gets worse
    cmap = LinearSegmentedColormap.from_list("nan_cmap", ["white", "#ffcccc", "#cc0000"])
    im = ax.imshow(coverage, aspect="auto", cmap=cmap, vmin=0, vmax=50)
    
    ax.set_xticks(range(len(ENERGY)))
    ax.set_xticklabels(ENERGY, fontsize=10)
    ax.set_yticks(range(n_feats))
    
    # Color y-tick labels by group
    yticklabels = []
    for col in all_features:
        grp = lookup_group(col)
        yticklabels.append(col)
    ax.set_yticklabels(yticklabels, fontsize=7)
    for tick, col in zip(ax.get_yticklabels(), all_features):
        tick.set_color(GROUP_COLORS.get(lookup_group(col), "black"))
    
    # Annotate cells
    for i in range(n_feats):
        for j in range(len(ENERGY)):
            val = coverage[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                       fontsize=6, color="black" if val < 25 else "white")
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("% NaN in signal period", fontsize=9)
    ax.set_title("NaN coverage by feature × instrument (signal period 2020-2022)",
                 fontsize=11, pad=15)
    
    # Legend for group colors
    handles = [plt.Rectangle((0,0), 1, 1, fc=c) for c in GROUP_COLORS.values()]
    ax.legend(handles, GROUP_COLORS.keys(),
              loc="center left", bbox_to_anchor=(1.25, 0.5), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {savepath}")


# ----------------------------------------------------------------------
# Figure 2 — Correlation heatmap (cl1s, signal period)
# ----------------------------------------------------------------------

def plot_correlation_heatmap(feats_dict, ticker="cl1s",
                              savepath="fig_02_correlation_heatmap.png"):
    """Full feature × feature correlation heatmap, grouped/ordered by category."""
    df = feats_dict[ticker].loc[SIGNAL_START:SIGNAL_END]
    
    # Order features by group
    ordered_cols = []
    group_boundaries = []   # column index where each new group starts
    for grp, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        group_boundaries.append((grp, len(ordered_cols), len(present)))
        ordered_cols.extend(present)
    
    # Keep only columns with non-zero variance
    valid_cols = [c for c in ordered_cols
                  if df[c].notna().any() and df[c].std() > 0]
    
    corr = df[valid_cols].corr()
    
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    
    ax.set_xticks(range(len(valid_cols)))
    ax.set_xticklabels(valid_cols, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(valid_cols)))
    ax.set_yticklabels(valid_cols, fontsize=6.5)
    
    # Color tick labels by group
    for tick, col in zip(ax.get_xticklabels(), valid_cols):
        tick.set_color(GROUP_COLORS.get(lookup_group(col), "black"))
    for tick, col in zip(ax.get_yticklabels(), valid_cols):
        tick.set_color(GROUP_COLORS.get(lookup_group(col), "black"))
    
    # Draw group boundary lines
    cum = 0
    boundary_positions = []
    for grp, _, n in group_boundaries:
        present_in_grp = sum(1 for c in FEATURE_GROUPS[grp] if c in valid_cols)
        if present_in_grp == 0:
            continue
        cum += present_in_grp
        boundary_positions.append(cum)
    
    for b in boundary_positions[:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=0.6, alpha=0.4)
        ax.axvline(b - 0.5, color="black", linewidth=0.6, alpha=0.4)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson correlation", fontsize=10)
    ax.set_title(f"Feature correlation matrix — {ticker} (signal period)\n"
                 f"Tick label colours indicate feature group; black lines = group boundaries",
                 fontsize=11, pad=15)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {savepath}")


# ----------------------------------------------------------------------
# Figure 3 — Cross-instrument variation
# ----------------------------------------------------------------------

def plot_cross_instrument(feats_dict, savepath="fig_03_cross_instrument.png"):
    """Bar chart of features sorted by cross-instrument std-of-mean."""
    all_features = list(feats_dict[ENERGY[0]].columns)
    rows = []
    for col in all_features:
        means = []
        for tk in ENERGY:
            sub = feats_dict[tk].loc[SIGNAL_START:SIGNAL_END]
            means.append(sub[col].mean())
        rows.append({
            "feature": col,
            "group": lookup_group(col),
            "std_across_inst": np.nanstd(means),
            "values": means,
        })
    df = pd.DataFrame(rows).sort_values("std_across_inst", ascending=True)
    
    # Drop features that are constants/NaN across instruments
    df = df[df["std_across_inst"] > 0].reset_index(drop=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 12),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    
    colors = [GROUP_COLORS.get(g, "gray") for g in df["group"]]
    
    # Left panel: bar chart of std-of-means
    axes[0].barh(range(len(df)), df["std_across_inst"], color=colors)
    axes[0].set_yticks(range(len(df)))
    axes[0].set_yticklabels(df["feature"], fontsize=6.5)
    for tick, grp in zip(axes[0].get_yticklabels(), df["group"]):
        tick.set_color(GROUP_COLORS.get(grp, "black"))
    axes[0].set_xlabel("Std of mean across 4 instruments", fontsize=9)
    axes[0].set_title("Features sorted by cross-instrument variation\n(higher = more discriminative)",
                       fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="x")
    
    # Right panel: spider-like plot — each row shows per-instrument mean as colored dots
    for i, row in df.iterrows():
        vals = row["values"]
        v = np.array(vals, dtype=float)
        # Normalize to [0, 1] for visual comparability within each row
        finite = np.isfinite(v)
        if finite.sum() >= 2 and (np.nanmax(v) - np.nanmin(v)) > 0:
            v_norm = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))
        else:
            v_norm = np.zeros_like(v)
        for j, tk in enumerate(ENERGY):
            if not np.isfinite(v_norm[j]):
                continue
            axes[1].scatter(j, i, s=50 + float(v_norm[j]) * 150, alpha=0.7,
                           color=GROUP_COLORS.get(row["group"], "gray"),
                           edgecolors="black", linewidths=0.3)
    
    axes[1].set_xticks(range(len(ENERGY)))
    axes[1].set_xticklabels(ENERGY, fontsize=9)
    axes[1].set_yticks(range(len(df)))
    axes[1].set_yticklabels([], fontsize=0)
    axes[1].set_title("Relative magnitude per instrument\n(dot size = normalized mean within row)",
                       fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="x")
    axes[1].set_xlim(-0.5, len(ENERGY) - 0.5)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {savepath}")


# ----------------------------------------------------------------------
# Figure 4 — Distribution histograms of key features
# ----------------------------------------------------------------------

def plot_distributions(feats_dict, savepath="fig_04_distributions.png"):
    """Histograms of selected features, one row per feature, 4 columns per instrument."""
    key_features = [
        ("ret_5d", "A_returns_momentum"),
        ("vol_yang_zhang_20d", "B_volatility"),
        ("amihud_20d", "C_microstructure"),
        ("rsi_14", "D_meanrev_trend"),
        ("hmm_p_state0", "E_latent_regime"),
        ("hurst_90d", "F_spectral_fractal"),
        ("corr_basket_60d", "G_cross_sectional"),
        ("signal_persistence", "H_signal_interaction"),
        ("heating_season", "I_seasonality"),
    ]
    
    fig, axes = plt.subplots(len(key_features), len(ENERGY),
                             figsize=(13, 2.0 * len(key_features)))
    
    for i, (feat, grp) in enumerate(key_features):
        for j, tk in enumerate(ENERGY):
            ax = axes[i, j]
            sub = feats_dict[tk].loc[SIGNAL_START:SIGNAL_END]
            if feat not in sub.columns:
                ax.text(0.5, 0.5, "not present", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            
            vals = sub[feat].dropna()
            if len(vals) == 0:
                ax.text(0.5, 0.5, "all NaN", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            
            ax.hist(vals, bins=30, color=GROUP_COLORS.get(grp, "gray"),
                    alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.axvline(vals.mean(), color="red", linewidth=1, linestyle="--", alpha=0.7)
            
            if i == 0:
                ax.set_title(tk, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(feat, fontsize=8, rotation=0, ha="right",
                             va="center", labelpad=10)
            ax.tick_params(labelsize=6)
    
    fig.suptitle("Distributions of selected features × instruments (signal period)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {savepath}")


# ----------------------------------------------------------------------
# Figure 5 — Time series of key features through the signal period
# ----------------------------------------------------------------------

def plot_signal_period_evolution(feats_dict, savepath="fig_05_signal_period_evolution.png"):
    """Time-series plots: see how feature values evolve through 2020-2022."""
    panel_features = [
        ("hmm_p_state0", "Probability of stress regime"),
        ("vol_yang_zhang_20d", "Yang-Zhang volatility"),
        ("primary_signal", "Primary signal (-1/0/+1)"),
        ("signal_persistence", "Signal persistence (days)"),
        ("corr_basket_60d", "Correlation with peer basket"),
        ("hurst_90d", "Hurst exponent"),
    ]
    
    fig, axes = plt.subplots(len(panel_features), 1, figsize=(12, 12),
                             sharex=True)
    
    for i, (feat, title) in enumerate(panel_features):
        ax = axes[i]
        for tk in ENERGY:
            sub = feats_dict[tk].loc[SIGNAL_START:SIGNAL_END]
            if feat in sub.columns:
                ax.plot(sub.index, sub[feat], label=tk,
                       linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9, ncol=4)
        ax.tick_params(labelsize=8)
    
    axes[-1].set_xlabel("Date", fontsize=10)
    fig.suptitle("Time-series evolution of key features through the signal period",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {savepath}")


# ----------------------------------------------------------------------
# Figure 6 — Within-group correlation submatrices
# ----------------------------------------------------------------------

def plot_within_group_correlations(feats_dict, ticker="cl1s",
                                    savepath="fig_06_group_correlations.png"):
    """Small heatmaps showing correlation structure WITHIN each group."""
    df = feats_dict[ticker].loc[SIGNAL_START:SIGNAL_END]
    
    # Filter groups that have at least 2 valid features
    plot_groups = []
    for grp, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df.columns
                   and df[c].notna().any() and df[c].std() > 0]
        if len(present) >= 2:
            plot_groups.append((grp, present))
    
    n_groups = len(plot_groups)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, (grp, cols) in enumerate(plot_groups):
        ax = axes[i]
        sub_corr = df[cols].corr()
        im = ax.imshow(sub_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title(f"{grp}  ({len(cols)} features)",
                     fontsize=9, color=GROUP_COLORS.get(grp, "black"),
                     fontweight="bold")
        
        # Annotate cells
        for r in range(len(cols)):
            for c in range(len(cols)):
                ax.text(c, r, f"{sub_corr.iloc[r, c]:.2f}",
                       ha="center", va="center", fontsize=6,
                       color="white" if abs(sub_corr.iloc[r, c]) > 0.5 else "black")
    
    # Hide unused axes
    for j in range(n_groups, len(axes)):
        axes[j].axis("off")
    
    fig.suptitle(f"Within-group feature correlations — {ticker} (signal period)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {savepath}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data and building features...")
    panel, primary_signals = load_panel(
        ohlcv_path="ohlcv_data.csv",
        signals_path="primary_signals.csv",
    )
    
    feats_dict = build_features_panel(
        panel=panel,
        primary_signals=primary_signals,
        asset_class_tickers=ENERGY,
        anchor_ticker=ANCHOR,
        train_end_date=pd.Timestamp("2019-12-31"),
        fit_regime_models=True,
    )
    
    print("\nGenerating plots:")
    plot_coverage(feats_dict)
    plot_correlation_heatmap(feats_dict, ticker="cl1s")
    plot_cross_instrument(feats_dict)
    plot_distributions(feats_dict)
    plot_signal_period_evolution(feats_dict)
    plot_within_group_correlations(feats_dict, ticker="cl1s")
    
    print("\nDone. PNG files saved in working directory.")