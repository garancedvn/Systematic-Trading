"""
build_base_features.py  --  RUN ONCE (heavy, HMM-independent).

Produces two cached parquet files consumed by the HMM-iteration loop:
  * base_features_full.parquet : full-history feature_df (pre-merge) for HMM training
  * labeled_base.parquet       : signal-merged + triple-barrier-labeled subset (NO hmm_* cols)
  * split_meta.json            : ENERGY_INSTRUMENTS + global_train_end_date

Everything here is invariant to the HMM choice, so it is computed a single time.
Re-run only if the raw CSVs, the base features, or the triple-barrier params change.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import talib
import yfinance as yf
import statsmodels.api as sm1
import statsmodels.tsa.stattools as ts
from sklearn.preprocessing import StandardScaler


# --- [notebook cell 5] ------------------------------------------------
# ============================================================
# Phase 1 — Setup and Data Loading
# ============================================================

import os
import numpy as np
from numpy import lib
import pandas as pd

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.dates as mdates

import statsmodels.api as sm1
import talib
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

import shap

import itertools

import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from itertools import product
import statsmodels.tsa.stattools as ts

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from itertools import product
import lightgbm as lgb

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import optuna
from sklearn.base import clone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import copy
import warnings

from itertools import combinations
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

from sklearn.metrics import (    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    classification_report,
    average_precision_score,
    brier_score_loss
)

RANDOM_STATE = 42

# Display options
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)

# --- [notebook cell 6] ------------------------------------------------
# ------------------------------------------------------------
# Project configuration
# ------------------------------------------------------------

DATA_DIR = "."  # Change this path if your CSV files are in another folder

OHLCV_FILE = os.path.join(DATA_DIR, "ohlcv_data.csv")
SIGNALS_FILE = os.path.join(DATA_DIR, "primary_signals.csv")

ENERGY_INSTRUMENTS = ["cl1s", "ho1s", "rb1s", "ng1s"]

INSTRUMENT_NAMES = {
    "cl1s": "WTI Crude Oil",
    "ho1s": "Heating Oil",
    "rb1s": "RBOB Gasoline",
    "ng1s": "Natural Gas",
}

print("Energy universe:")
for ticker, name in INSTRUMENT_NAMES.items():
    print(f"- {ticker.upper()}: {name}")

# --- [notebook cell 7] ------------------------------------------------
# ============================================================
# Load Raw Data
# ============================================================

ohlcv_raw = pd.read_csv(OHLCV_FILE)
signals_raw = pd.read_csv(SIGNALS_FILE)

print("OHLCV shape:", ohlcv_raw.shape)
print("Signals shape:", signals_raw.shape)

print(ohlcv_raw.head())
print(signals_raw.head())

# --- [notebook cell 9] ------------------------------------------------
# ============================================================
# Standardize Dates and Instrument Names
# ============================================================

ohlcv = ohlcv_raw.copy()
signals = signals_raw.copy()

# Convert date columns
ohlcv["date"] = pd.to_datetime(ohlcv["date"])
signals["date"] = pd.to_datetime(signals["date"])

# Standardize instrument names
ohlcv["instrument"] = ohlcv["instrument"].str.lower()

# Keep only Energy instruments
ohlcv_energy = ohlcv[ohlcv["instrument"].isin(ENERGY_INSTRUMENTS)].copy()

# Keep date + Energy signal columns
signals_energy = signals[["date"] + ENERGY_INSTRUMENTS].copy()

# Sort
ohlcv_energy = ohlcv_energy.sort_values(["instrument", "date"]).reset_index(drop=True)
signals_energy = signals_energy.sort_values("date").reset_index(drop=True)

print("Energy OHLCV shape:", ohlcv_energy.shape)
print("Energy signals shape:", signals_energy.shape)

print(ohlcv_energy.head())
print(signals_energy.head())

# --- [notebook cell 11] ------------------------------------------------
# ============================================================
# Basic Data Integrity Checks
# ============================================================

print("OHLCV date range by instrument:")
print(
    ohlcv_energy
    .groupby("instrument")["date"]
    .agg(["min", "max", "count"])
)

print("\nSignals date range:")
print(
    signals_energy["date"].agg(["min", "max", "count"])
)

print("\nDuplicate OHLCV rows by (date, instrument):")
n_dup_ohlcv = ohlcv_energy.duplicated(subset=["date", "instrument"]).sum()
print(n_dup_ohlcv)

print("\nDuplicate signal dates:")
n_dup_signals = signals_energy.duplicated(subset=["date"]).sum()
print(n_dup_signals)

print("\nMissing values in OHLCV:")
print(ohlcv_energy.isna().sum())

print("\nMissing values in signals:")
print(signals_energy.isna().sum())

# --- [notebook cell 13] ------------------------------------------------
# ============================================================
# Signal Distribution
# ============================================================

signal_distribution = {}

for inst in ENERGY_INSTRUMENTS:
    
    counts = (
        signals_energy[inst]
        .value_counts()
        .sort_index()
    )
    
    signal_distribution[inst] = counts

    print(f"\n{inst.upper()} signal distribution:")
    print(counts)

    print("\nPercentages:")
    print((counts / counts.sum() * 100).round(2))

# --- [notebook cell 15] ------------------------------------------------
# ============================================================
# OHLC Consistency Checks
# ============================================================

# High should be >= low
invalid_high_low = (ohlcv_energy["high"] < ohlcv_energy["low"]).sum()

# Open should lie inside [low, high]
invalid_open = (
    (ohlcv_energy["open"] < ohlcv_energy["low"]) |
    (ohlcv_energy["open"] > ohlcv_energy["high"])
).sum()

# Close should lie inside [low, high]
invalid_close = (
    (ohlcv_energy["close"] < ohlcv_energy["low"]) |
    (ohlcv_energy["close"] > ohlcv_energy["high"])
).sum()

# Negative or zero prices
non_positive_prices = (
    (ohlcv_energy[["open", "high", "low", "close"]] <= 0)
    .sum()
    .sum()
)

print("Invalid high/low rows:", invalid_high_low)
print("Invalid open rows:", invalid_open)
print("Invalid close rows:", invalid_close)
print("Non-positive prices:", non_positive_prices)

# --- [notebook cell 17] ------------------------------------------------
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=[
        f"{inst.upper()} — {INSTRUMENT_NAMES[inst]}"
        for inst in ENERGY_INSTRUMENTS
    ]
)

for row, inst in enumerate(ENERGY_INSTRUMENTS, start=1):
    df_inst = ohlcv_energy[ohlcv_energy["instrument"] == inst]
    
    fig.add_trace(
        go.Scatter(
            x=df_inst["date"],
            y=df_inst["close"],
            mode="lines",
            name=inst.upper(),
            line=dict(width=1.5)
        ),
        row=row,
        col=1
    )
    
    fig.update_yaxes(title_text="Close Price", row=row, col=1)

fig.update_layout(
    height=900,
    width=950,
    title_text="Historical Close Prices for Energy Futures",
    showlegend=False
)

fig.show()

# --- [notebook cell 19] ------------------------------------------------
# ============================================================
# Daily Log Returns
# ============================================================

ohlcv_energy["log_return"] = (
    ohlcv_energy
    .groupby("instrument")["close"]
    .transform(lambda x: np.log(x / x.shift(1)))
)

print(
    ohlcv_energy[
        ["date", "instrument", "close", "log_return"]
    ].head(10)
)

# --- [notebook cell 20] ------------------------------------------------
# ============================================================
# Plot Log Returns
# ============================================================
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=[
        f"{inst.upper()} — {INSTRUMENT_NAMES[inst]}"
        for inst in ENERGY_INSTRUMENTS
    ]
)

for row, inst in enumerate(ENERGY_INSTRUMENTS, start=1):
    df_inst = ohlcv_energy[ohlcv_energy["instrument"] == inst]

    fig.add_trace(
        go.Scatter(
            x=df_inst["date"],
            y=df_inst["log_return"],
            mode="lines",
            name=inst.upper(),
            line=dict(width=1.5)
        ),
        row=row,
        col=1
    )

    fig.add_hline(
        y=0,
        line=dict(color="black", width=1),
        row=row,
        col=1
    )

    fig.update_yaxes(title_text="Log Return", row=row, col=1)

fig.update_layout(
    height=900,
    width=950,
    title_text="Daily Log Returns for Energy Futures",
    showlegend=False
)

fig.show()

# --- [notebook cell 22] ------------------------------------------------
# ============================================================
# Reshape Signals to Long Format
# ============================================================

signals_long = signals_energy.melt(
    id_vars="date",
    value_vars=ENERGY_INSTRUMENTS,
    var_name="instrument",
    value_name="primary_signal"
)

signals_long = (
    signals_long
    .sort_values(["instrument", "date"])
    .reset_index(drop=True)
)

print("Signals long shape:", signals_long.shape)

print(signals_long.head(10))

# ======================================================================
# FEATURE ENGINEERING (base, HMM-independent)
# ======================================================================

# --- [notebook cell 25] ------------------------------------------------
# ============================================================
# Phase 2 — Feature Engineering Setup
# ============================================================

features_df = ohlcv_energy.copy()

# Make sure data is sorted before computing rolling/grouped features
features_df = (
    features_df
    .sort_values(["instrument", "date"])
    .reset_index(drop=True)
)

print("Feature engineering base shape:", features_df.shape)
print(features_df.head())

# --- [notebook cell 26] ------------------------------------------------
# ============================================================
# Core Return and Momentum Features
# ============================================================

MOMENTUM_WINDOWS = [5, 10, 20, 60]

for window in MOMENTUM_WINDOWS:
    
    features_df[f"momentum_{window}d"] = (
        features_df
        .groupby("instrument")["close"]
        .transform(lambda x: np.log(x / x.shift(window)))
    )

# Rolling mean return features
RETURN_MEAN_WINDOWS = [5, 20, 60]

for window in RETURN_MEAN_WINDOWS:
    
    features_df[f"mean_return_{window}d"] = (
        features_df
        .groupby("instrument")["log_return"]
        .transform(lambda x: x.rolling(window=window).mean())
    )

print(
    features_df[
        ["date", "instrument", "close", "log_return",
         "momentum_5d", "momentum_20d", "momentum_60d",
         "mean_return_5d", "mean_return_20d"]
    ].head(25)
)

# --- [notebook cell 28] ------------------------------------------------
def rolling_zscore(s, w):
    mean = s.rolling(w).mean()
    std = s.rolling(w).std()
    zscore = (s - mean) / std
    return zscore

features_df["ret_20d_zscore"] = rolling_zscore(features_df["mean_return_20d"], 252)
print(features_df[["date", "instrument", "close", "mean_return_20d", "ret_20d_zscore"]])

# --- [notebook cell 30] ------------------------------------------------
# ============================================================
# Rolling Realized Volatility
# ============================================================

VOL_WINDOWS = [5, 20, 60]

for window in VOL_WINDOWS:

    features_df[f"realized_vol_{window}d"] = (
        features_df
        .groupby("instrument")["log_return"]
        .transform(lambda x: x.rolling(window=window).std())
    )

print(
    features_df[
        [
            "date",
            "instrument",
            "log_return",
            "realized_vol_5d",
            "realized_vol_20d",
            "realized_vol_60d"
        ]
    ].head() #change to head(30) if you want to see more rows
)

# --- [notebook cell 31] ------------------------------------------------
# ============================================================
# EWMA Volatility
# ============================================================

EWMA_SPANS = [10, 20]

for span in EWMA_SPANS:

    features_df[f"ewma_vol_{span}d"] = (
        features_df
        .groupby("instrument")["log_return"]
        .transform(
            lambda x: x.ewm(span=span, adjust=False).std()
        )
    )

print(
    features_df[
        [
            "date",
            "instrument",
            "ewma_vol_10d",
            "ewma_vol_20d"
        ]
    ].head() #change to head(30) if you want to see more rows
)

# --- [notebook cell 32] ------------------------------------------------
# ============================================================
# Volatility Regime Features
# ============================================================

features_df["vol_ratio_20_60"] = (
    features_df["realized_vol_20d"]
    /
    features_df["realized_vol_60d"]
)

print(
    features_df[
        [
            "date",
            "instrument",
            "realized_vol_20d",
            "realized_vol_60d",
            "vol_ratio_20_60"
        ]
    ].head() #change to head(30) if you want to see more rows
)

# --- [notebook cell 34] ------------------------------------------------
# ============================================================
# More efficient volatility calculations then the standard one
# ============================================================

def garman_klass_vol(o, h, l, c, window: int = 20) -> pd.Series:
    hl = 0.5 * np.log(h / l) ** 2
    co = (2 * np.log(2) - 1) * np.log(c / o) ** 2
    daily_var = (hl - co).clip(lower=0)             # ← clip before rolling
    return np.sqrt(daily_var.rolling(window).mean())

def yang_zhang_vol(open, high, low, close, window=20):
    close_prev = close.shift(1)
    overnight = (np.log(open/close_prev)) ** 2
    open_to_close = (np.log(close/open)) ** 2

    sigma_overnight = overnight.rolling(window).mean()
    sigma_open_to_close = open_to_close.rolling(window).mean()

    rs = np.log(high / open) * np.log(high / close) + np.log(low / open) * np.log(low / close)
    sigma_rs = rs.rolling(window).mean()

    k= 0.34 / (1.34 + (window + 1) / (window - 1))
    var = sigma_overnight + k * sigma_open_to_close + (1-k) * sigma_rs
    return np.sqrt(var.clip(lower=0))

open = features_df["open"]
high = features_df["high"]
low = features_df["low"]
close = features_df["close"]

features_df["garman_klass_vol"] = garman_klass_vol(open, high, low, close)
features_df["yang_zhang_vol"] = yang_zhang_vol(open, high, low, close)

print(
    features_df[
        [
            "date",
            "instrument", 
            "garman_klass_vol",
            "yang_zhang_vol"
        ]
    ].iloc[20:] # skip first 20 rows because rolling window starts at 20
)

# --- [notebook cell 36] ------------------------------------------------
# ============================================================
# Higher Moments and Range Position Features
# ============================================================

# ------------------------------------------------------------
# Rolling skewness of log returns (20-day and 60-day)
# ------------------------------------------------------------

features_df["skew_20d"] = (
    features_df
    .groupby("instrument")["log_return"]
    .transform(lambda x: x.rolling(20).skew())
)

features_df["skew_60d"] = (
    features_df
    .groupby("instrument")["log_return"]
    .transform(lambda x: x.rolling(60).skew())
)

# ------------------------------------------------------------
# Rolling kurtosis of log returns (20-day)
# ------------------------------------------------------------

features_df["kurt_20d"] = (
    features_df
    .groupby("instrument")["log_return"]
    .transform(lambda x: x.rolling(20).kurt())
)

# ------------------------------------------------------------
# Downside volatility (std of negative returns only, 20-day window)
# ------------------------------------------------------------

def _downside_std(x):
    neg = x[x < 0]
    return np.std(neg) if len(neg) > 1 else np.nan

features_df["downside_vol_20d"] = (
    features_df
    .groupby("instrument")["log_return"]
    .transform(
        lambda x: x.rolling(20).apply(_downside_std, raw=True)
    )
)

# ------------------------------------------------------------
# Price range position over 60-day window
# 0.0 = at 60d low, 1.0 = at 60d high
# ------------------------------------------------------------

hh_60 = (
    features_df
    .groupby("instrument")["high"]
    .transform(lambda x: x.rolling(60).max())
)
ll_60 = (
    features_df
    .groupby("instrument")["low"]
    .transform(lambda x: x.rolling(60).min())
)

features_df["price_range_position_60d"] = (
    (features_df["close"] - ll_60) / (hh_60 - ll_60).replace(0, np.nan)
)

# ------------------------------------------------------------
# 5-day change in range position (velocity)
# ------------------------------------------------------------

features_df["range_position_5d_chg"] = (
    features_df
    .groupby("instrument")["price_range_position_60d"]
    .transform(lambda x: x.diff(5))
)

# ------------------------------------------------------------
# Return / vol-change correlation (60-day, leverage effect)
# ------------------------------------------------------------

# Use realized_vol_20d (already built in section 2.2) as the vol measure
features_df["vol_change_20d"] = (
    features_df
    .groupby("instrument")["realized_vol_20d"]
    .transform(lambda x: x.diff())
)

features_df["return_vol_correl_60d"] = (
    features_df
    .groupby("instrument", group_keys=False)
    .apply(
        lambda g: g["log_return"].rolling(60).corr(g["vol_change_20d"])
    )
)

# Drop the temporary vol_change column
features_df = features_df.drop(columns=["vol_change_20d"])

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print(
    features_df[
        [
            "date",
            "instrument",
            "skew_20d",
            "kurt_20d",
            "downside_vol_20d",
            "price_range_position_60d",
            "range_position_5d_chg",
            "return_vol_correl_60d",
        ]
    ]
    .dropna()
    .head(10)
)

# Quick sanity check on the new features
print("\nSummary statistics for new features (CL only):")
print(
    features_df[features_df["instrument"] == "cl1s"][
        [
            "skew_20d", "skew_60d", "kurt_20d",
            "downside_vol_20d",
            "price_range_position_60d", "range_position_5d_chg",
            "return_vol_correl_60d",
        ]
    ]
    .describe()
    .round(4)
)

# --- [notebook cell 37] ------------------------------------------------
def parkinson_vol(h, l, window=20):
    return np.sqrt((np.log(h / l) ** 2 / (4 * np.log(2))).rolling(window).mean())

def rogers_satchell_vol(o, h, l, c, window=20):
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(rs.clip(lower=0).rolling(window).mean())

features_df['vol_parkinson_20d'] = features_df.groupby("instrument", group_keys=False).apply(
    lambda g: parkinson_vol(g['high'], g['low']))
features_df['vol_rogers_satchell_20d'] = features_df.groupby("instrument", group_keys=False).apply(
    lambda g: rogers_satchell_vol(g['open'], g['high'], g['low'], g['close']))

# --- [notebook cell 39] ------------------------------------------------
# ============================================================
# Volume Features
# ============================================================

# Log volume change
features_df["log_volume_change"] = (
    features_df
    .groupby("instrument")["volume"]
    .transform(lambda x: np.log(x / x.shift(1)))
)

# Rolling volume averages
VOLUME_WINDOWS = [5, 20]

for window in VOLUME_WINDOWS:

    features_df[f"volume_mean_{window}d"] = (
        features_df
        .groupby("instrument")["volume"]
        .transform(lambda x: x.rolling(window).mean())
    )

    features_df[f"volume_std_{window}d"] = (
        features_df
        .groupby("instrument")["volume"]
        .transform(lambda x: x.rolling(window).std())
    )

print(
    features_df[
        [
            "date",
            "instrument",
            "volume",
            "log_volume_change",
            "volume_mean_20d",
            "volume_std_20d"
        ]
    ].head() #change to head(30) if you want to see more rows
)

# --- [notebook cell 40] ------------------------------------------------
# ============================================================
# Volume Z-Score Features
# ============================================================

features_df["volume_zscore_20d"] = (
    (
        features_df["volume"]
        - features_df["volume_mean_20d"]
    )
    /
    features_df["volume_std_20d"]
)

print(
    features_df[
        [
            "date",
            "instrument",
            "volume",
            "volume_mean_20d",
            "volume_std_20d",
            "volume_zscore_20d"
        ]
    ].head() #change to head(30) if you want to see more rows
)

# --- [notebook cell 41] ------------------------------------------------
# ============================================================
# Open Interest Features
# ============================================================

# Log change in open interest
features_df["log_oi_change"] = (
    features_df
    .groupby("instrument")["open_interest"]
    .transform(lambda x: np.log(x / x.shift(1)))
)

# Open interest momentum
OI_WINDOWS = [5, 20]

for window in OI_WINDOWS:

    features_df[f"oi_momentum_{window}d"] = (
        features_df
        .groupby("instrument")["open_interest"]
        .transform(lambda x: np.log(x / x.shift(window)))
    )

print(
    features_df[
        [
            "date",
            "instrument",
            "open_interest",
            "log_oi_change",
            "oi_momentum_5d",
            "oi_momentum_20d"
        ]
    ].head() #change to head(30) if you want to see more rows
)

# --- [notebook cell 43] ------------------------------------------------
# ============================================================
# Microstructure and Liquidity Features
# ============================================================

# ------------------------------------------------------------
# Helper: dollar volume series (used by multiple features)
# ------------------------------------------------------------

features_df["dollar_volume"] = (
    features_df["close"] * features_df["volume"]
)

# ------------------------------------------------------------
# Amihud illiquidity (20-day rolling mean of |return| / dollar volume)
# ------------------------------------------------------------

features_df["amihud_daily"] = (
    features_df["log_return"].abs()
    / features_df["dollar_volume"].replace(0, np.nan)
)

features_df["amihud_20d"] = (
    features_df
    .groupby("instrument")["amihud_daily"]
    .transform(lambda x: x.rolling(20).mean())
)

# ------------------------------------------------------------
# Roll's effective spread (20-day window)
#   spread ≈ 2 * sqrt( -Cov(Δp_t, Δp_{t-1}) )
# Only defined where the autocovariance is negative (bid-ask bounce).
# Clipped to zero otherwise.
# ------------------------------------------------------------

features_df["price_change"] = (
    features_df
    .groupby("instrument")["close"]
    .transform(lambda x: x.diff())
)

# Lagged price change for autocovariance computation
features_df["price_change_lag1"] = (
    features_df
    .groupby("instrument")["price_change"]
    .transform(lambda x: x.shift(1))
)

# Rolling mean of the product (this is the autocovariance of price changes)
features_df["cov_pricechanges_20d"] = (
    features_df
    .groupby("instrument", group_keys=False)
    .apply(
        lambda g: (g["price_change"] * g["price_change_lag1"]).rolling(20).mean()
    )
)

features_df["roll_spread_20d"] = (
    2 * np.sqrt((-features_df["cov_pricechanges_20d"]).clip(lower=0))
)

# ------------------------------------------------------------
# Log dollar volume (liquidity-level feature)
# ------------------------------------------------------------

features_df["dollar_volume_log"] = (
    np.log(features_df["dollar_volume"].replace(0, np.nan))
)

# ------------------------------------------------------------
# Kyle's lambda: price impact per unit of √volume (20-day window)
#   lambda = Cov(|return|, sqrt(volume)) / Var(sqrt(volume))
# ------------------------------------------------------------

features_df["abs_return"] = features_df["log_return"].abs()
features_df["sqrt_volume"] = np.sqrt(features_df["volume"].replace(0, np.nan))

def _kyle_lambda(g, w=20):
    abs_r = g["abs_return"]
    sqrt_v = g["sqrt_volume"]
    cov = (abs_r * sqrt_v).rolling(w).mean() \
          - abs_r.rolling(w).mean() * sqrt_v.rolling(w).mean()
    var = sqrt_v.rolling(w).var()
    return cov / var.replace(0, np.nan)

features_df["kyle_lambda_20d"] = (
    features_df
    .groupby("instrument", group_keys=False)
    .apply(_kyle_lambda)
)

# ------------------------------------------------------------
# Clean up temporary columns
# ------------------------------------------------------------

features_df = features_df.drop(
    columns=[
        "dollar_volume",
        "amihud_daily",
        "price_change",
        "price_change_lag1",
        "cov_pricechanges_20d",
        "abs_return",
        "sqrt_volume",
    ]
)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print(
    features_df[
        [
            "date",
            "instrument",
            "amihud_20d",
            "roll_spread_20d",
            "dollar_volume_log",
            "kyle_lambda_20d",
        ]
    ]
    .dropna()
    .head(10)
)

print("\nSummary statistics (CL only):")
print(
    features_df[features_df["instrument"] == "cl1s"][
        [
            "amihud_20d",
            "roll_spread_20d",
            "dollar_volume_log",
            "kyle_lambda_20d",
        ]
    ]
    .describe()
    .round(6)
)

# --- [notebook cell 44] ------------------------------------------------
features_df['hl_spread']      = (features_df['high'] - features_df['low']) / features_df['close']
features_df['oc_spread']      = (features_df['close'] - features_df['open']) / features_df['open']
features_df['close_position'] = (features_df['close'] - features_df['low']) / (features_df['high'] - features_df['low'])

# --- [notebook cell 46] ------------------------------------------------
# ============================================================
# RSI Indicator
# ============================================================

def compute_rsi(series, window=14):
    """
    Compute the Relative Strength Index (RSI).
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


features_df["rsi_14d"] = (
    features_df
    .groupby("instrument")["close"]
    .transform(lambda x: compute_rsi(x, window=14))
)

print(
    features_df[
        ["date", "instrument", "close", "rsi_14d"]
    ].head(70)
)

# --- [notebook cell 47] ------------------------------------------------
# ============================================================
# Bollinger-Style Price Z-Score
# ============================================================

BOLLINGER_WINDOWS = [20, 60]

for window in BOLLINGER_WINDOWS:

    rolling_mean = (
        features_df
        .groupby("instrument")["close"]
        .transform(lambda x: x.rolling(window=window).mean())
    )

    rolling_std = (
        features_df
        .groupby("instrument")["close"]
        .transform(lambda x: x.rolling(window=window).std())
    )

    features_df[f"price_zscore_{window}d"] = (
        (features_df["close"] - rolling_mean) / rolling_std
    )

print(
    features_df[
        [
            "date",
            "instrument",
            "close",
            "price_zscore_20d",
            "price_zscore_60d"
        ]
    ].head(70)
)

# --- [notebook cell 48] ------------------------------------------------
# ============================================================
# MACD Indicator
# ============================================================

def compute_macd(series, short_span=12, long_span=26, signal_span=9):
    """
    Compute MACD, MACD signal line, and MACD histogram.
    """
    ema_short = series.ewm(span=short_span, adjust=False).mean()
    ema_long = series.ewm(span=long_span, adjust=False).mean()

    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_span, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd, macd_signal, macd_hist


macd_results = (
    features_df
    .groupby("instrument")["close"]
    .apply(lambda x: pd.DataFrame({
        "macd": compute_macd(x)[0],
        "macd_signal": compute_macd(x)[1],
        "macd_hist": compute_macd(x)[2],
    }, index=x.index))
)

macd_results = macd_results.reset_index(level=0, drop=True)

features_df[["macd", "macd_signal", "macd_hist"]] = macd_results[
    ["macd", "macd_signal", "macd_hist"]
]

print(
    features_df[
        [
            "date",
            "instrument",
            "close",
            "macd",
            "macd_signal",
            "macd_hist"
        ]
    ].head(70)
)

# --- [notebook cell 50] ------------------------------------------------
#ATX 14 - average directional index, measures strength of trend not direction

open = features_df["open"]
high = features_df["high"]
low = features_df["low"]
close = features_df["close"]

tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

atr = tr.ewm(alpha=1/14, adjust=False).mean()
up_move = high - high.shift(1)
down_move = low.shift(1) - low
plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
plus_di = 100 * pd.Series(plus_dm, index=features_df.index).ewm(alpha=1/14, adjust=False).mean() / atr
minus_di = 100 * pd.Series(minus_dm, index=features_df.index).ewm(alpha=1/14, adjust=False).mean() / atr
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

features_df["atx_14"] = dx.ewm(alpha=1/14, adjust=False).mean()

ma_200 = close.rolling(200).mean()
std_60 = close.rolling(60).std()
features_df["distance_from_200d_ma"] = (close - ma_200) / std_60.replace(0, np.nan)

print(features_df[
    [
        "date", 
        "instrument",
        "high",
        "low",
        "close",
        "atx_14",
        "distance_from_200d_ma"
    ]
].iloc[200:])

# --- [notebook cell 51] ------------------------------------------------
def add_talib_block(g):
    h, l, c, v = g['high'].values, g['low'].values, g['close'].values, g['volume'].values.astype(float)
    g['willr_14'] = talib.WILLR(h, l, c, timeperiod=14)
    g['stoch_k'], g['stoch_d'] = talib.STOCH(h, l, c)
    g['obv'] = talib.OBV(c, v)
    g['mfi_14'] = talib.MFI(h, l, c, v, timeperiod=14)
    g['atr_14'] = talib.ATR(h, l, c, timeperiod=14)
    u, m, lo = talib.BBANDS(c, timeperiod=20)
    g['bb_width'] = (u - lo) / m
    g['bb_position'] = (c - lo) / (u - lo)
    g['vol_oi_ratio'] = g['volume'] / g['open_interest'].replace(0, np.nan)
    return g

features_df = features_df.groupby("instrument", group_keys=False).apply(add_talib_block)

# --- [notebook cell 53] ------------------------------------------------
# ============================================================
# Rolling Return Autocorrelation
# ============================================================

AUTOCORR_WINDOWS = [20, 60]

for window in AUTOCORR_WINDOWS:

    features_df[f"autocorr_return_{window}d"] = (
        features_df
        .groupby("instrument")["log_return"]
        .transform(
            lambda x: x.rolling(window).corr(x.shift(1))
        )
    )

print(
    features_df[
        [
            "date",
            "instrument",
            "log_return",
            "autocorr_return_20d",
            "autocorr_return_60d"
        ]
    ].head(80)
)

# --- [notebook cell 54] ------------------------------------------------
# ============================================================
# Absolute Return Autocorrelation
# ============================================================

features_df["abs_log_return"] = (
    features_df["log_return"].abs()
)

for window in AUTOCORR_WINDOWS:

    features_df[f"autocorr_abs_return_{window}d"] = (
        features_df
        .groupby("instrument")["abs_log_return"]
        .transform(
            lambda x: x.rolling(window).corr(x.shift(1))
        )
    )

print(
    features_df[
        [
            "date",
            "instrument",
            "abs_log_return",
            "autocorr_abs_return_20d",
            "autocorr_abs_return_60d"
        ]
    ].head(80)
)

# --- [notebook cell 55] ------------------------------------------------
# ============================================================
# Trend Persistence Features
# ============================================================

# Positive-return indicator
features_df["positive_return"] = (
    (features_df["log_return"] > 0).astype(int)
)

PERSISTENCE_WINDOWS = [10, 20]

for window in PERSISTENCE_WINDOWS:

    features_df[f"positive_return_ratio_{window}d"] = (
        features_df
        .groupby("instrument")["positive_return"]
        .transform(
            lambda x: x.rolling(window).mean()
        )
    )

print(
    features_df[
        [
            "date",
            "instrument",
            "positive_return",
            "positive_return_ratio_10d",
            "positive_return_ratio_20d"
        ]
    ].head(70)
)

# --- [notebook cell 57] ------------------------------------------------
def rolling_apply_array(s, w, fn):
    vals = s.values 
    n = len(s)
    out_vals = np.full(n, np.nan)
    for i in range(w - 1, n):
        window = vals[i - w + 1 : i + 1]
        if np.all(np.isfinite(window)):
            try:
                out_vals[i] = fn(window)
            except Exception:
                out_vals[i] = np.nan
    return pd.Series(out_vals, index=s.index)

def dominant_cycle_period(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 8 or np.std(x) == 0:
        return np.nan
    
    t = np.arange(len(x))
    coeffs = np.polyfit(t, x, 1)
    detrended = x - np.polyval(coeffs, t)
    
    spec = np.abs(np.fft.rfft(detrended))
    # Zero out: DC + the lowest 2 bins (which absorb residual trend curvature)
    spec[:3] = 0
    
    if spec.sum() == 0:
        return np.nan
    
    k = np.argmax(spec)
    return len(x) / max(k, 1)

def spectral_entropy(x):
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    spec = spec[1:]  # drop DC component
    total = spec.sum()
    
    if total <= 0:
        return np.nan
    
    p = spec / total
    p = p[p > 0]
    return -(p * np.log(p)).sum() / np.log(len(p))

def hurst(x: np.ndarray) -> float:
    """
    Hurst exponent via the rescaled range (R/S) method.
    Original Hurst-Mandelbrot estimator.
    H ≈ 0.5: random walk
    H > 0.5: persistent (trending)
    H < 0.5: anti-persistent (mean-reverting)
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 20 or np.std(x) == 0:
        return np.nan
    
    # Work on increments (returns), not levels
    returns = np.diff(x)
    if len(returns) < 10 or np.std(returns) == 0:
        return np.nan
    
    # Build R/S for several scales
    scales = []
    rs_values = []
    for n in [10, 20, 30, 40, 50, 60, 80]:
        if n > len(returns):
            break
        # Split into chunks of size n
        n_chunks = len(returns) // n
        if n_chunks < 2:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = returns[i * n : (i + 1) * n]
            mean_c = chunk.mean()
            z = np.cumsum(chunk - mean_c)
            R = z.max() - z.min()         # range of cumulative deviations
            S = np.std(chunk, ddof=1)     # std of chunk
            if S > 0 and np.isfinite(R / S):
                rs_chunk.append(R / S)
        if len(rs_chunk) >= 1:
            scales.append(n)
            rs_values.append(np.mean(rs_chunk))
    
    if len(scales) < 3:
        return np.nan
    
    # log(R/S) ~ H * log(n) for fBm
    slope, _ = np.polyfit(np.log(scales), np.log(rs_values), 1)
    return slope

def dfa(x):
    if len(x) < 16:
        return np.nan
    
    y = np.cumsum(x - x.mean())
    
    scales = [s for s in [4, 8, 16] if s < len(x) // 2]
    if len(scales) < 2:
        return np.nan
    
    f = []
    for s in scales:
        n_segments = len(y) // s
        rms_list = []
        for i in range(n_segments):
            seg = y[i * s : (i + 1) * s]
            t = np.arange(s)
            poly = np.polyfit(t, seg, 1)
            trend = np.polyval(poly, t)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        f.append(np.mean(rms_list) if rms_list else np.nan)
    
    f = np.array(f)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        return np.nan
    
    slope, _ = np.polyfit(np.log(scales), np.log(f), 1)
    return slope

def approx_entropy(x: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """
    Approximate entropy (Pincus 1991). Lower = more predictable.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return np.nan
    
    sd = np.std(x)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    r = r_mult * sd
    
    def _phi(m_):
        # Build all sub-sequences of length m_
        N = n - m_ + 1
        if N <= 0:
            return np.nan
        patterns = np.zeros((N, m_))
        for i in range(N):
            patterns[i] = x[i : i + m_]
        
        # For each pattern, count how many others are within r in Chebyshev distance
        counts = np.zeros(N)
        for i in range(N):
            diff = np.abs(patterns - patterns[i])
            max_diff = diff.max(axis=1)
            counts[i] = np.sum(max_diff <= r) / N
        
        # Avoid log(0)
        counts = np.where(counts > 0, counts, 1e-12)
        return np.mean(np.log(counts))
    
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    if not (np.isfinite(phi_m) and np.isfinite(phi_m1)):
        return np.nan
    return phi_m - phi_m1

# --- [notebook cell 58] ------------------------------------------------
LOG_CLOSE_COL = "log_close"
features_df[LOG_CLOSE_COL] = np.log(features_df["close"])

# Hurst — applied to log prices, 90-day window
features_df["hurst_90d"] = (
    features_df
    .groupby("instrument")[LOG_CLOSE_COL]
    .transform(lambda x: x.rolling(90).apply(hurst, raw=True))
)

# DFA alpha — applied to log returns, 90-day window
features_df["dfa_alpha_90d"] = (
    features_df
    .groupby("instrument")["log_return"]
    .transform(lambda x: x.rolling(90).apply(dfa, raw=True))
)

# Dominant cycle period — applied to log prices, 60-day window
features_df["dominant_cycle_period"] = (
    features_df
    .groupby("instrument")[LOG_CLOSE_COL]
    .transform(lambda x: x.rolling(60).apply(dominant_cycle_period, raw=True))
)

# Spectral entropy — applied to log returns, 60-day window
features_df["spectral_entropy"] = (
    features_df
    .groupby("instrument")["log_return"]
    .transform(lambda x: x.rolling(60).apply(spectral_entropy, raw=True))
)

# Approximate entropy — applied to log returns, 20-day window
features_df["approx_entropy_20d"] = (
    features_df
    .groupby("instrument")["log_return"]
    .transform(lambda x: x.rolling(20).apply(approx_entropy, raw=True))
)

# Clean up the temporary log_close column
features_df = features_df.drop(columns=[LOG_CLOSE_COL])

print(
    features_df[
        [
            "date", "instrument",
            "hurst_90d", "dfa_alpha_90d",
            "dominant_cycle_period", "spectral_entropy",
            "approx_entropy_20d",
        ]
    ].dropna().head()
)

# --- [notebook cell 59] ------------------------------------------------
def lempel_ziv_complexity(seq):
    i, c, u, v, v_max, n = 0, 1, 1, 1, 1, len(seq)
    if n == 0: return 0
    while u + v <= n:
        if seq[i + v - 1] == seq[u + v - 1]:
            v += 1
        else:
            v_max = max(v_max, v); i += 1
            if i == u:
                c += 1; u += v_max; v = v_max = 1; i = 0
            else:
                v = 1
    return c + (v != 1)

def frac_diff_ffd(series, d, thres=1e-5):
    w = [1.0]; k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres: break
        w.append(w_k); k += 1
    w = np.array(w[::-1]); width = len(w) - 1
    out = pd.Series(index=series.index, dtype=float)
    for i in range(width, len(series)):
        out.iloc[i] = np.dot(w, series.iloc[i - width:i + 1].values)
    return out

def sadf_rolling(close, step=5):
    sadf_vals = np.full(len(close), np.nan)
    for i in range(25, len(close), step):
        window = close.iloc[max(0, i - 252):i].values
        try:
            sadf_vals[i] = ts.adfuller(window, maxlag=1, autolag=None)[0]
        except Exception:
            pass
    out = pd.Series(sadf_vals, index=close.index)
    return out.ffill()

def shannon_entropy_vectorized(series, window, bins=10):
    arr = series.values
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        x = arr[i - window + 1:i + 1]
        x = x[~np.isnan(x)]
        if len(x) == 0: continue
        hist, _ = np.histogram(x, bins=bins)
        p = hist[hist > 0].astype(float); p /= p.sum()
        result[i] = -np.sum(p * np.log2(p))
    return pd.Series(result, index=series.index)

def lz_rolling_fast(series, window):
    binary = (series.values > 0).astype(int)
    result = np.full(len(binary), np.nan)
    for i in range(window - 1, len(binary)):
        seq = ''.join(map(str, binary[i - window + 1:i + 1]))
        result[i] = lempel_ziv_complexity(seq)
    return pd.Series(result, index=series.index)


features_df['close_fracdiff'] = features_df.groupby("instrument")['close'].transform(
    lambda x: frac_diff_ffd(x, d=0.4))
features_df['sadf']= sadf_rolling(features_df['close'], step=5)  
features_df['shannon_entropy_hist_20d'] = features_df.groupby("instrument")['log_return'].transform(
    lambda x: shannon_entropy_vectorized(x, 20))
features_df['shannon_entropy_hist_60d'] = features_df.groupby("instrument")['log_return'].transform(
    lambda x: shannon_entropy_vectorized(x, 60))
features_df['lz_complexity_20d'] = features_df.groupby("instrument")['log_return'].transform(
    lambda x: lz_rolling_fast(x, 20))
features_df['lz_complexity_60d'] = features_df.groupby("instrument")['log_return'].transform(
    lambda x: lz_rolling_fast(x, 60))

# --- [notebook cell 61] ------------------------------------------------
# ============================================================
# Cross-Sectional Momentum Rank
# ============================================================

features_df["momentum_rank_20d"] = (
    features_df
    .groupby("date")["momentum_20d"]
    .rank(pct=True)
)

features_df["momentum_rank_60d"] = (
    features_df
    .groupby("date")["momentum_60d"]
    .rank(pct=True)
)

print(
    features_df[
        [
            "date",
            "instrument",
            "momentum_20d",
            "momentum_rank_20d",
            "momentum_60d",
            "momentum_rank_60d"
        ]
    ]
    .sort_values(["date", "momentum_rank_20d"])
    .head(70)
)

# --- [notebook cell 62] ------------------------------------------------
# ============================================================
# Relative Volatility Features
# ============================================================

# Sector average volatility by date
sector_avg_vol_20d = (
    features_df
    .groupby("date")["realized_vol_20d"]
    .transform("mean")
)

sector_avg_vol_60d = (
    features_df
    .groupby("date")["realized_vol_60d"]
    .transform("mean")
)

# Relative volatility
features_df["relative_vol_20d"] = (
    features_df["realized_vol_20d"]
    / sector_avg_vol_20d
)

features_df["relative_vol_60d"] = (
    features_df["realized_vol_60d"]
    / sector_avg_vol_60d
)

print(
    features_df[
        [
            "date",
            "instrument",
            "realized_vol_20d",
            "relative_vol_20d",
            "realized_vol_60d",
            "relative_vol_60d"
        ]
    ].head(70)
)

# --- [notebook cell 63] ------------------------------------------------
# ============================================================
# Sector Dispersion Features
# ============================================================

# Cross-sectional std of momentum across assets
features_df["sector_momentum_dispersion_20d"] = (
    features_df
    .groupby("date")["momentum_20d"]
    .transform("std")
)

features_df["sector_momentum_dispersion_60d"] = (
    features_df
    .groupby("date")["momentum_60d"]
    .transform("std")
)

# Cross-sectional std of volatility across assets
features_df["sector_vol_dispersion_20d"] = (
    features_df
    .groupby("date")["realized_vol_20d"]
    .transform("std")
)

print(
    features_df[
        [
            "date",
            "instrument",
            "sector_momentum_dispersion_20d",
            "sector_momentum_dispersion_60d",
            "sector_vol_dispersion_20d"
        ]
    ].head(70)
)

# --- [notebook cell 64] ------------------------------------------------
# ============================================================
# Relative Momentum vs Sector Mean
# ============================================================

sector_mean_momentum_20d = (
    features_df
    .groupby("date")["momentum_20d"]
    .transform("mean")
)

features_df["relative_momentum_20d"] = (
    features_df["momentum_20d"]
    - sector_mean_momentum_20d
)

print(
    features_df[
        [
            "date",
            "instrument",
            "momentum_20d",
            "relative_momentum_20d"
        ]
    ]
    .sort_values(["date", "relative_momentum_20d"])
    .head(70)
)

# --- [notebook cell 66] ------------------------------------------------
# ============================================================
# Seasonality Features
# ============================================================

def _seasonal_ramp(months, days_in_month, start_month, peak_month, end_month):
    """
    Smooth tent function over a window of months [start_month, end_month],
    peaking at peak_month. Returns 0 outside the window, 1 at the peak.
    Handles year-wrap (e.g., heating: Oct -> Mar) via modulo arithmetic.
    """
    month_frac = (months - 1) + (days_in_month - 1) / 31.0
    
    diff = month_frac - (peak_month - 1)
    diff = (diff + 6) % 12 - 6   # wrap to [-6, 6]
    
    dist_to_start = (peak_month - start_month) % 12
    if dist_to_start == 0:
        dist_to_start = 1
    dist_to_end = (end_month - peak_month) % 12
    if dist_to_end == 0:
        dist_to_end = 1
    
    ramp = np.where(
        diff < 0,
        1 + diff / dist_to_start,
        1 - diff / dist_to_end,
    )
    return np.clip(ramp, 0, 1)


# ------------------------------------------------------------
# Day-of-year cyclic encoding
# ------------------------------------------------------------

doy = features_df["date"].dt.dayofyear

features_df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
features_df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

# ------------------------------------------------------------
# Physical-demand seasonal ramps
# ------------------------------------------------------------

months = features_df["date"].dt.month.values
days_in_month = features_df["date"].dt.days_in_month.values

features_df["heating_season"] = _seasonal_ramp(
    months, days_in_month,
    start_month=10, peak_month=1, end_month=4,
)

features_df["driving_season_progress"] = _seasonal_ramp(
    months, days_in_month,
    start_month=4, peak_month=7, end_month=8,
)

features_df["hurricane_season_indicator"] = _seasonal_ramp(
    months, days_in_month,
    start_month=6, peak_month=9, end_month=11,
)

# ------------------------------------------------------------
# Quarter progress (fraction through current quarter)
# ------------------------------------------------------------

quarter_starts = features_df["date"].dt.to_period("Q").dt.start_time
quarter_ends = features_df["date"].dt.to_period("Q").dt.end_time

quarter_length = (quarter_ends - quarter_starts).dt.days.clip(lower=1)
elapsed = (features_df["date"] - quarter_starts).dt.days

features_df["quarter_progress"] = elapsed / quarter_length

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print(
    features_df[
        [
            "date",
            "instrument",
            "day_of_year_sin",
            "day_of_year_cos",
            "heating_season",
            "driving_season_progress",
            "hurricane_season_indicator",
            "quarter_progress",
        ]
    ]
    .head(10)
)

# Quick sanity check: monthly mean values
print("\nMonthly mean values of seasonal indicators (CL only):")
print(
    features_df[features_df["instrument"] == "cl1s"]
    .assign(month=features_df["date"].dt.month)
    .groupby("month")[
        ["heating_season", "driving_season_progress", "hurricane_season_indicator"]
    ]
    .mean()
    .round(3)
)

# --- [notebook cell 69] ------------------------------------------------
# 3-2-1 crack, oil/gas ratio, sub-cracks, plus z-scores and changes.
# These features are the SAME for every energy instrument — they describe the 
# state of the energy complex. 

# Finding cl, ho, rb, ng in the ohlcv DataFrame
cl_close = ohlcv[ohlcv["instrument"] == "cl1s"].set_index("date")["close"]
ho_close = ohlcv[ohlcv["instrument"] == "ho1s"].set_index("date")["close"]
rb_close = ohlcv[ohlcv["instrument"] == "rb1s"].set_index("date")["close"]
ng_close = ohlcv[ohlcv["instrument"] == "ng1s"].set_index("date")["close"]

# Finding crack_321, oil/gas ratio, and their z-scores and changes
crack_321 = 3 * rb_close + 2 * ho_close - 3 * cl_close
crack_321_zscore_252d = (crack_321 - crack_321.rolling(window=252).mean()) / crack_321.rolling(window=252).std()

crack_321_change_5d = crack_321.diff(5)

oil_gas_ratio = cl_close / ng_close
oil_gas_ratio_zscore_252d = (oil_gas_ratio - oil_gas_ratio.rolling(window=252).mean()) / oil_gas_ratio.rolling(window=252).std()

# --- [notebook cell 70] ------------------------------------------------
# Joining the metrics with the ohlcv_energy DataFrame
new_features = pd.DataFrame({
    "crack_321": crack_321,
    "crack_321_zscore_252d": crack_321_zscore_252d,
    "crack_321_change_5d": crack_321_change_5d,
    "oil_gas_ratio": oil_gas_ratio,
    "oil_gas_ratio_zscore_252d": oil_gas_ratio_zscore_252d
}).reset_index()

features_df = features_df.merge(new_features, on="date", how="left")

# --- [notebook cell 71] ------------------------------------------------
features_df[
        [
            "date",
            "instrument",
            "crack_321",
            "crack_321_zscore_252d",
            "crack_321_change_5d",
            "oil_gas_ratio",
            "oil_gas_ratio_zscore_252d"
        ]
    ]

# --- [notebook cell 74] ------------------------------------------------
# Finding the close price series for ES, NQ, FESX, GC, and HG
es_close = ohlcv[ohlcv["instrument"] == "es1s"].set_index("date")["close"]
nq_close = ohlcv[ohlcv["instrument"] == "nq1s"].set_index("date")["close"]
fesx_close = ohlcv[ohlcv["instrument"] == "fesx1s"].set_index("date")["close"]
gc_close = ohlcv[ohlcv["instrument"] == "gc1s"].set_index("date")["close"]
hg_close = ohlcv[ohlcv["instrument"] == "hg1s"].set_index("date")["close"]

# Calculate log returns for each instrument
es_ret = np.log(es_close).diff()
nq_ret = np.log(nq_close).diff()
fesx_ret = np.log(fesx_close).diff()
gold_ret = np.log(gc_close).diff()
copper_ret = np.log(hg_close).diff()

# --- [notebook cell 75] ------------------------------------------------
# Group the equity returns into a DataFrame for easier calculations
equity_ret_df = pd.DataFrame({
    "es1s": es_ret,
    "nq1s": nq_ret,
    "fesx1s": fesx_ret
})

# Calculate the average return across all equity instruments
equity_ret = equity_ret_df.mean(axis=1)

# Calculate the 5-day rolling sum of equity returns
equity_avg_ret_5d = equity_ret.rolling(5).sum()

# Volatility features: rolling 20-day volatility of equity returns, then average across instruments, then z-score vs 252-day history
es_vol   = es_ret.rolling(20).std()   * np.sqrt(252)
nq_vol   = nq_ret.rolling(20).std()   * np.sqrt(252)
fesx_vol = fesx_ret.rolling(20).std() * np.sqrt(252)

equity_avg_vol_20d = pd.concat([es_vol, nq_vol, fesx_vol], axis=1).mean(axis=1)
equity_vol_zscore_252d = (
    equity_avg_vol_20d - equity_avg_vol_20d.rolling(252).mean()
) / equity_avg_vol_20d.rolling(252).std()

# Risk-on score (sum of 5-day returns of equities minus gold plus copper)
risk_on_score = equity_ret.rolling(5).sum() - gold_ret.rolling(5).sum() + copper_ret.rolling(5).sum()

# --- [notebook cell 76] ------------------------------------------------
# Join all these cross-asset metrics into a single DataFrame, then merge with features_df
cross_asset_metrics_df = pd.DataFrame({
    "equity_avg_ret_5d": equity_avg_ret_5d,
    "equity_avg_vol_20d": equity_avg_vol_20d,
    "equity_vol_zscore_252d": equity_vol_zscore_252d,
    "risk_on_score": risk_on_score
}).reset_index()

# Joining the cross-asset metrics with the features_df
features_df = features_df.merge(cross_asset_metrics_df, on="date", how="left")

# --- [notebook cell 77] ------------------------------------------------
features_df[['date', 'instrument', 'equity_avg_ret_5d', 'equity_avg_vol_20d', 'equity_vol_zscore_252d', 'risk_on_score']]

# --- [notebook cell 80] ------------------------------------------------
# Getting the close price series for HG, GC, and SI
hg_close = ohlcv[ohlcv["instrument"] == "hg1s"].set_index("date")["close"]
gc_close = ohlcv[ohlcv["instrument"] == "gc1s"].set_index("date")["close"]
si_close = ohlcv[ohlcv["instrument"] == "si1s"].set_index("date")["close"]

# Calculate the ratio Cobre / Oro
copper_gold_ratio = hg_close / gc_close

# Z-score for the copper/gold ratio vs its own 252-day history
copper_gold_ratio_zscore_252d = (copper_gold_ratio - copper_gold_ratio.rolling(window=252).mean()) / copper_gold_ratio.rolling(window=252).std()

# 20-day percentage change in the copper/gold ratio
copper_gold_ratio_chg_20d = copper_gold_ratio.pct_change(20)

#  Calculate the ratio Gold / Silver
gold_silver_ratio = gc_close / si_close

# --- [notebook cell 81] ------------------------------------------------
# Join all these metal metrics into a single DataFrame, then merge with features_df
metals_metrics_df = pd.DataFrame({
    "copper_gold_ratio": copper_gold_ratio,
    "copper_gold_ratio_zscore_252d": copper_gold_ratio_zscore_252d,
    "copper_gold_ratio_chg_20d": copper_gold_ratio_chg_20d,
    "gold_silver_ratio": gold_silver_ratio
}).reset_index()

# Joining the metals metrics with the features_df
features_df = features_df.merge(metals_metrics_df, on="date", how="left")

# --- [notebook cell 82] ------------------------------------------------
features_df[["date", "instrument", "copper_gold_ratio", "copper_gold_ratio_zscore_252d", "copper_gold_ratio_chg_20d", "gold_silver_ratio"]]

# --- [notebook cell 85] ------------------------------------------------
# Extracting the close price series for HG, GC, and ES to compute their log returns
hg_close = ohlcv[ohlcv["instrument"] == "hg1s"].set_index("date")["close"]
gc_close = ohlcv[ohlcv["instrument"] == "gc1s"].set_index("date")["close"]
es_close = ohlcv[ohlcv["instrument"] == "es1s"].set_index("date")["close"]

hg_ret = np.log(hg_close).diff()
gc_ret = np.log(gc_close).diff()
es_ret = np.log(es_close).diff()

#  List of your 4 target assets
list_metrics = []

# Calculating the rolling correlations for each target asset against HG, GC, and ES
for target_asset in ENERGY_INSTRUMENTS:
    target_close = ohlcv[ohlcv["instrument"] == target_asset].set_index("date")["close"]
    target_ret = np.log(target_close).diff()
    
    # Calculate rolling correlations with a 60-day window
    asset_copper_corr = target_ret.rolling(60).corr(hg_ret)
    asset_gold_corr = target_ret.rolling(60).corr(gc_ret)
    asset_equity_corr = target_ret.rolling(60).corr(es_ret)
    
    # Temporal Dataframe for this asset's correlations
    df_temporal = pd.DataFrame({
        "asset_copper_corr_60d": asset_copper_corr,
        "asset_gold_corr_60d": asset_gold_corr,
        "asset_equity_corr_60d": asset_equity_corr
    }).reset_index()
    
    df_temporal["instrument"] = target_asset
    
    list_metrics.append(df_temporal)

# --- [notebook cell 86] ------------------------------------------------
# Concatenating all the correlation DataFrames into a single DataFrame
correlation_df = pd.concat(list_metrics, ignore_index=True)

# Joining the correlation metrics with the features_df
features_df = features_df.merge(correlation_df, on=["date", "instrument"], how="left")

# --- [notebook cell 89] ------------------------------------------------
# 1. Extract close prices and calculate log returns for ALL reference macro assets
hg_close = ohlcv[ohlcv["instrument"] == "hg1s"].set_index("date")["close"]  # Copper
gc_close = ohlcv[ohlcv["instrument"] == "gc1s"].set_index("date")["close"]  # Gold
es_close = ohlcv[ohlcv["instrument"] == "es1s"].set_index("date")["close"]  # S&P 500
nq_close = ohlcv[ohlcv["instrument"] == "nq1s"].set_index("date")["close"]  # Nasdaq
fesx_close = ohlcv[ohlcv["instrument"] == "fesx1s"].set_index("date")["close"]  # Euro Stoxx

hg_ret = np.log(hg_close).diff()
gc_ret = np.log(gc_close).diff()
es_ret = np.log(es_close).diff()
nq_ret = np.log(nq_close).diff()
fesx_ret = np.log(fesx_close).diff()

# 2. Calculate Global Reference Momentums
# For Equities, we combine the 3 indices by taking the daily cross-sectional mean of their returns
eq_returns_df = pd.DataFrame({"es": es_ret, "nq": nq_ret, "fesx": fesx_ret})
eq_mom = eq_returns_df.mean(axis=1).rolling(20).sum()

cu_mom = hg_ret.rolling(20).sum()
au_mom = gc_ret.rolling(20).sum()

# Direct Copper features
copper_mom_20d = cu_mom
copper_lead_5d = hg_ret.rolling(5).sum()

# 3. Process Concordance for each of your target assets (Energy Universe)
asset_mom_list = []
target_assets = ["cl1s", "ho1s", "rb1s", "ng1s"]  # Defined in Phase 1 of your notebook
signals_df = signals.set_index("date")

for target_asset in target_assets:
    # Get the complete timeline in ohlcv for this specific asset
    target_dates = ohlcv[ohlcv["instrument"] == target_asset]["date"].unique()
    
    # Safely reindex the global macro metrics to the asset's timeline
    cu_mom_target = cu_mom.reindex(target_dates)
    au_mom_target = au_mom.reindex(target_dates)
    eq_mom_target = eq_mom.reindex(target_dates)
    cu_mom_20d_target = copper_mom_20d.reindex(target_dates)
    cu_lead_5d_target = copper_lead_5d.reindex(target_dates)
    
    # Extract the asset's signal and align it to the same timeline
    if target_asset in signals_df.columns:
        sig = signals_df[target_asset].reindex(target_dates)
    else:
        sig = pd.Series(np.nan, index=target_dates)
        
    # Replace inactive signals (0) with NaN as required by the original design
    sig_sign = np.sign(sig).replace(0, np.nan)
    
    # Calculate bit-by-bit concordance (range 0 to 3)
    # Note: NaN == anything evaluates to False, protecting the initial rolling windows
    concord = (
        (np.sign(eq_mom_target) == sig_sign).astype(int)
        + (np.sign(cu_mom_target) == sig_sign).astype(int)
        + (np.sign(au_mom_target) == sig_sign).astype(int)
    )
    
    # Apply the .where() mask: the indicator is only valid on days with an active signal (+1 or -1)
    cross_asset_momentum_concordance = concord.where(sig_sign.notna())
    
    # Build the clean DataFrame for the current asset
    temp_df = pd.DataFrame({
        "date": target_dates,
        "copper_mom_20d": cu_mom_20d_target.values,
        "copper_lead_5d": cu_lead_5d_target.values,
        "cross_asset_momentum_concordance": cross_asset_momentum_concordance.values
    })
    temp_df["instrument"] = target_asset
    
    asset_mom_list.append(temp_df)

# 4. Concatenate the results vertically to consolidate the long format
all_mom_metrics_df = pd.concat(asset_mom_list, ignore_index=True)

# 5. Integration via left merge into your general features dataset (features_df)
features_df = features_df.merge(all_mom_metrics_df, on=["date", "instrument"], how="left")

# --- [notebook cell 90] ------------------------------------------------
features_df[["date", "instrument", "copper_mom_20d", "copper_lead_5d", "cross_asset_momentum_concordance"]].head(70)

# --- [notebook cell 93] ------------------------------------------------
# Universe-wide vol regime + how stressed this instrument is vs the rest.
# 1. Pivot the dataset natively to get a "wide" format (dates as rows, instruments as columns)
px = ohlcv.pivot(index="date", columns="instrument", values="close")

# 2. Calculate daily log returns for ALL instruments simultaneously
ret = np.log(px).diff()

# Define the asset classes (Make sure these match the exact strings in your dataset)
energy_cols = ["cl1s", "ho1s", "rb1s", "ng1s"]
equity_cols = ["es1s", "nq1s", "fesx1s"]

# Get a list of all instruments dynamically from the pivoted DataFrame
universe_cols = list(px.columns)

# 3. Calculate the 20-day rolling volatility for every instrument, annualized
# Pandas applies this column by column automatically
vol_per_instr = ret.rolling(20).std() * np.sqrt(252)

# 4. Calculate the macro volatility metrics using cross-sectional means (axis=1)
multiasset_vol_index = vol_per_instr[universe_cols].mean(axis=1)

# Manual Z-score for the multi-asset volatility index (252-day window)
multiasset_vol_zscore_252d = (
    multiasset_vol_index - multiasset_vol_index.rolling(window=252).mean()
) / multiasset_vol_index.rolling(window=252).std()

equity_vol_avg_20d = vol_per_instr[equity_cols].mean(axis=1)

# How stressed the energy complex is relative to the entire market
relative_energy_vol = vol_per_instr[energy_cols].mean(axis=1) / multiasset_vol_index

# --- [notebook cell 94] ------------------------------------------------
# 5. Assemble the final metrics DataFrame
volatility_metrics_df = pd.DataFrame({
    "multiasset_vol_index": multiasset_vol_index,
    "multiasset_vol_zscore_252d": multiasset_vol_zscore_252d,
    "equity_vol_avg_20d": equity_vol_avg_20d,
    "relative_energy_vol": relative_energy_vol
}).reset_index() # Converts the "date" index back into a standard column

# 6. Merge with your main features dataset (features_df)
# Since these are macro indicators (not specific to one instrument), we only merge on "date"
features_df = features_df.merge(volatility_metrics_df, on="date", how="left")

# --- [notebook cell 95] ------------------------------------------------
features_df[["date", "instrument", "multiasset_vol_index", "multiasset_vol_zscore_252d", "equity_vol_avg_20d", "relative_energy_vol"]].head(70)

# --- [notebook cell 98] ------------------------------------------------
def calculate_backward_trend_t_value(price_series: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculates the t-value of the slope from an OLS regression of price on time,
    computed over a fixed rolling lookback window.
    """
    # Pre-allocate output series with the same timeline index
    t_values = pd.Series(index=price_series.index, dtype=float)
    p = price_series.values
    
    # Pre-compute parts of X since the time index array [0, 1, 2, ..., window-1] 
    # is identical for every single rolling window
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_dev = x - x_mean
    ssx = (x_dev ** 2).sum()
    df = window - 2
    
    if ssx == 0 or df <= 0:
        return t_values

    # Rolling window loop
    for i in range(window, len(p)):
        y = p[i - window : i]  # Strict past slice of length 'window'
        y_mean = y.mean()
        
        # OLS slope (beta) coefficient calculation
        beta = (x_dev * (y - y_mean)).sum() / ssx
        
        # Intercept (alpha) calculation
        alpha = y_mean - beta * x_mean
        
        # Residuals and Standard Error calculation
        residuals = y - (alpha + beta * x)
        sigma2 = (residuals ** 2).sum() / df
        
        if sigma2 <= 0:
            continue
            
        se_beta = np.sqrt(sigma2 / ssx)
        
        if se_beta == 0:
            continue
            
        # Calculate and store the t-statistic (beta / standard error)
        t_values.iat[i] = beta / se_beta
        
    return t_values

# --- [notebook cell 99] ------------------------------------------------
# --- Main Feature Calculation and Integration Pipeline ---

# 1. Isolate the Copper close prices and set the date timeline as index
hg_close = ohlcv[ohlcv["instrument"] == "hg1s"].set_index("date")["close"]

# 2. Calculate the 20-day rolling Z-score manually
copper_zscore_20d = (
    hg_close - hg_close.rolling(window=20).mean()
) / hg_close.rolling(window=20).std()

# 3. Calculate the rolling 30-day OLS trend t-value
copper_trend_t_value = calculate_backward_trend_t_value(hg_close, window=30)

# 4. Construct the clean macro metrics DataFrame
copper_metrics_df = pd.DataFrame({
    "copper_zscore_20d": copper_zscore_20d,
    "copper_trend_t_value": copper_trend_t_value
}).reset_index()  # Convert the 'date' index back into a column

# 5. Integrate into your main features DataFrame
# Since Copper indicators are macro/exogenous regimes, we only merge on "date"
features_df = features_df.merge(copper_metrics_df, on="date", how="left")

# --- [notebook cell 100] ------------------------------------------------
features_df[["date", "instrument", "copper_zscore_20d", "copper_trend_t_value"]].head(70)

# --- [notebook cell 103] ------------------------------------------------
# Pulling DXY, VIX, US 10y yield from Yahoo Finance for the date range in features_df.
# Returns a long-form DataFrame indexed by date that we can use to derive features.
tickers = {"dxy": "DX-Y.NYB", "vix": "^VIX", "us10y": "^TNX"}
frames = []
for name, tkr in tickers.items():
    df = yf.download(
        tkr,
        start=features_df["date"].min(),
        end=features_df["date"].max() + pd.Timedelta(days=1),  # yfinance end is exclusive
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        print(f"  Warning: no data returned for {tkr}")
        continue
    # yfinance can return a MultiIndex column ('Close','^VIX'); flatten if so
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.rename(name)
    frames.append(s)

macro_df = pd.concat(frames, axis=1).sort_index()
macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
macro_df = macro_df.reset_index().rename(columns={macro_df.index.name or "Date": "date"})
# Normalise the date column name (yfinance returns 'Date' with capital D)
macro_df.columns = ["date" if c.lower() == "date" else c for c in macro_df.columns]

macro_df.head()

# --- [notebook cell 104] ------------------------------------------------
# Building macro features and merging into features_df.

# Reindex macro data onto the trading-day calendar from features_df, forward-fill
# small gaps (US holidays vs commodity holidays differ by 1-2 days).
trading_days = features_df["date"].drop_duplicates().sort_values()
macro_aligned = (
    macro_df.set_index("date")
    .reindex(trading_days)
    .ffill(limit=2)
)

# DXY: log returns over 5d and 20d, plus its correlation with the energy basket.
dxy_ret = np.log(macro_aligned["dxy"]).diff()
dxy_ret_5d  = dxy_ret.rolling(5).sum()
dxy_ret_20d = dxy_ret.rolling(20).sum()

# Energy basket return (mean of CL, HO, RB, NG log returns) for the DXY correlation
# This way, the dxy_corr_60d feature is meaningful for all four energy instruments.
cl_ret = np.log(cl_close).diff()
ho_ret = np.log(ho_close).diff()
rb_ret = np.log(rb_close).diff()
ng_ret = np.log(ng_close).diff()
energy_basket_ret = pd.concat([cl_ret, ho_ret, rb_ret, ng_ret], axis=1).mean(axis=1)
energy_basket_ret = energy_basket_ret.reindex(trading_days)

dxy_corr_60d = energy_basket_ret.rolling(60).corr(dxy_ret)

# VIX: level and 5-day change
vix_level     = macro_aligned["vix"]
vix_change_5d = macro_aligned["vix"].diff(5)

# US 10y: 5-day change in yield
us10y_change_5d = macro_aligned["us10y"].diff(5)

# Assemble and merge
macro_metrics_df = pd.DataFrame({
    "dxy_ret_5d":      dxy_ret_5d,
    "dxy_ret_20d":     dxy_ret_20d,
    "dxy_corr_60d":    dxy_corr_60d,
    "vix_level":       vix_level,
    "vix_change_5d":   vix_change_5d,
    "us10y_change_5d": us10y_change_5d,
}).reset_index().rename(columns={"index": "date"})

features_df = features_df.merge(macro_metrics_df, on="date", how="left")

features_df[[
    "date", "instrument",
    "dxy_ret_5d", "dxy_ret_20d", "dxy_corr_60d",
    "vix_level", "vix_change_5d", "us10y_change_5d",
]].head()

# --- [notebook cell 106] ------------------------------------------------
# ============================================================
# Feature Set Summary
# ============================================================

feature_columns = [
    col for col in features_df.columns
    if col not in [
        "date", "instrument", "open", "high", "low", "close",
        "volume", "open_interest"
    ]
]

print("Number of engineered features:", len(feature_columns))
print("\nFeature columns:")
for col in feature_columns:
    print("-", col)

missing_summary = (
    features_df[feature_columns]
    .isna()
    .mean()
    .sort_values(ascending=False)
    .to_frame("missing_ratio")
)

print(missing_summary.head(40))

# --- [notebook cell 108] ------------------------------------------------
# ============================================================
# Dynamic Cross-Sectional Features: Basket Correlation, Lead-Lag, and Beta
# ============================================================

ANCHOR = "cl1s"

# ------------------------------------------------------------
# Build the peer-basket return (equal-weighted mean of all OTHER instruments)
# at each date, using the leave-one-out trick:
#  peer_basket = (sum_of_all - own) / (n_instruments - 1)
# ------------------------------------------------------------

n_instruments_per_date = (
    features_df
    .groupby("date")["log_return"]
    .transform("count")
)

sum_returns_per_date = (
    features_df
    .groupby("date")["log_return"]
    .transform("sum")
)

features_df["peer_basket_return"] = (
    (sum_returns_per_date - features_df["log_return"])
    / (n_instruments_per_date - 1)
)

# ------------------------------------------------------------
# Feature 1: 60-day rolling correlation with peer basket
# ------------------------------------------------------------

features_df["corr_basket_60d"] = (
    features_df
    .groupby("instrument", group_keys=False)
    .apply(
        lambda g: g["log_return"].rolling(60).corr(g["peer_basket_return"])
    )
)

# ------------------------------------------------------------
# Feature 2: 60-day rolling correlation with lagged anchor (CL)
# ------------------------------------------------------------

# Extract the anchor's daily return series, shifted by one day
anchor_lagged_return = (
    features_df.loc[features_df["instrument"] == ANCHOR]
    .set_index("date")["log_return"]
    .shift(1)
)

# Map the lagged anchor onto every row by date
features_df["anchor_lagged_return"] = (
    features_df["date"].map(anchor_lagged_return)
)

features_df["leadlag_anchor"] = (
    features_df
    .groupby("instrument", group_keys=False)
    .apply(
        lambda g: (
            g["log_return"].rolling(60).corr(g["anchor_lagged_return"])
            if g.name != ANCHOR
            else pd.Series(np.nan, index=g.index)
        )
    )
)

# ------------------------------------------------------------
# Feature 3: 60-day rolling beta on the peer basket
#     beta = Cov(r, b) / Var(b)
# ------------------------------------------------------------

def _rolling_beta(g, window=60):
    cov_rb = (
        (g["log_return"] * g["peer_basket_return"]).rolling(window).mean()
        - g["log_return"].rolling(window).mean()
          * g["peer_basket_return"].rolling(window).mean()
    )
    var_b = g["peer_basket_return"].rolling(window).var()
    return cov_rb / var_b.replace(0, np.nan)

features_df["beta_basket_60d"] = (
    features_df
    .groupby("instrument", group_keys=False)
    .apply(_rolling_beta)
)

# ------------------------------------------------------------
# Clean up temporary columns
# ------------------------------------------------------------

features_df = features_df.drop(
    columns=["peer_basket_return", "anchor_lagged_return"]
)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print(
    features_df[
        [
            "date",
            "instrument",
            "corr_basket_60d",
            "leadlag_anchor",
            "beta_basket_60d",
        ]
    ]
    .dropna()
    .head()
)

# ============================================================
# CACHE 1 : full-history features (pre-merge) for HMM training
# ============================================================

# ============================================================
# Chronological 80/20 split on signal dates -> global_train_end_date
# (HMM-independent: depends only on primary-signal dates)
# ============================================================
TEST_SIZE = 0.20
signal_split_info = {}
for instrument in ENERGY_INSTRUMENTS:
    signal_dates = (
        signals_long
        .loc[signals_long["instrument"] == instrument, "date"]
        .sort_values().drop_duplicates().reset_index(drop=True)
    )
    split_idx = int(len(signal_dates) * (1 - TEST_SIZE))
    signal_split_info[instrument] = {"train_end": signal_dates.iloc[:split_idx].max()}

global_train_end_date = min(info["train_end"] for info in signal_split_info.values())
print("Global train_end_date:", global_train_end_date)


features_df.to_parquet("base_features_full.parquet", index=False)
import builtins
with builtins.open("split_meta.json", "w") as _f:
    json.dump({"ENERGY_INSTRUMENTS": ENERGY_INSTRUMENTS,
               "global_train_end_date": str(global_train_end_date)}, _f)
print("Saved base_features_full.parquet + split_meta.json")

# ======================================================================
# MERGE WITH PRIMARY SIGNALS + PRIMARY INTERACTIONS
# ======================================================================

# --- [notebook cell 128] ------------------------------------------------
# ============================================================
# Merge engineered features with primary signals
# ============================================================

model_base_df = (
    signals_long
    .merge(
        features_df,
        on=["date", "instrument"],
        how="left"
    )
    .sort_values(["instrument", "date"])
    .reset_index(drop=True)
)

print("Model base dataframe shape:", model_base_df.shape)

print(
    model_base_df[
        [
            "date",
            "instrument",
            "primary_signal",
            "close",
            "log_return",
            "momentum_20d",
            "realized_vol_20d",
        ]
    ].head(20)
)

# --- [notebook cell 129] ------------------------------------------------
# ============================================================
# Check merge quality
# ============================================================

print("Missing close after merge:", model_base_df["close"].isna().sum())
print("Missing log_return after merge:", model_base_df["log_return"].isna().sum())

print("\nPrimary signal distribution:")
print(
    model_base_df["primary_signal"]
    .value_counts()
    .sort_index()
)

print("\nPrimary signal distribution by instrument:")
print(
    pd.crosstab(
        model_base_df["instrument"],
        model_base_df["primary_signal"]
    )
)

# --- [notebook cell 130] ------------------------------------------------
# Rows where signals exist but no matching OHLCV/features were found
missing_feature_rows = model_base_df[model_base_df["close"].isna()].copy()

print("Number of rows with missing features:", len(missing_feature_rows))

print(
    missing_feature_rows[
        ["date", "instrument", "primary_signal"]
    ]
    .sort_values(["instrument", "date"])
    .head(50)
)

# --- [notebook cell 131] ------------------------------------------------
# Count missing feature rows by instrument
print(
    missing_feature_rows["instrument"]
    .value_counts()
)

# --- [notebook cell 132] ------------------------------------------------
# Compare available dates between signals and features for each instrument
for instrument in ENERGY_INSTRUMENTS:
    
    signal_dates = set(
        signals_long.loc[
            signals_long["instrument"] == instrument,
            "date"
        ]
    )
    
    feature_dates = set(
        features_df.loc[
            features_df["instrument"] == instrument,
            "date"
        ]
    )
    
    missing_dates = sorted(signal_dates - feature_dates)
    
    print(f"\n{instrument}")
    print("Missing dates:", len(missing_dates))
    print(missing_dates[:10])

# --- [notebook cell 133] ------------------------------------------------
# ============================================================
# Drop signal rows without matching OHLCV/features
# ============================================================

model_base_df = (
    model_base_df
    .dropna(subset=["close"])
    .copy()
    .reset_index(drop=True)
)

print("Clean model base dataframe shape:", model_base_df.shape)

print("Remaining missing close:", model_base_df["close"].isna().sum())

# --- [notebook cell 135] ------------------------------------------------
# ============================================================
# Primary Signal Interaction Features
# ============================================================

# Feature 1: did the signal change today vs yesterday?
model_base_df["signal_changed"] = (
    model_base_df
    .groupby("instrument")["primary_signal"]
    .transform(lambda x: (x != x.shift(1)).astype(int))
)

# Feature 2: days since the signal last changed (signal persistence)
# Trick: cumulative count of changes defines a "run id", then count within each run
def _persistence(s):
    changes = (s != s.shift(1)).astype(int)
    run_id = changes.cumsum()
    return run_id.groupby(run_id).cumcount()

model_base_df["signal_persistence"] = (
    model_base_df
    .groupby("instrument")["primary_signal"]
    .transform(_persistence)
)

# Feature 3: concordance of signal with 50-day return direction
# Need 50-day log return; build it here from close (already available)
model_base_df["log_return_50d"] = (
    model_base_df
    .groupby("instrument")["close"]
    .transform(lambda x: np.log(x / x.shift(50)))
)

model_base_df["signal_trend_concord"] = (
    np.sign(model_base_df["primary_signal"])
    * np.sign(model_base_df["log_return_50d"])
)

# Drop the helper column
model_base_df = model_base_df.drop(columns=["log_return_50d"])

# Feature 4: fraction of last 20 days where signal was non-zero
model_base_df["signal_density_20d"] = (
    model_base_df
    .groupby("instrument")["primary_signal"]
    .transform(lambda x: (x != 0).rolling(20).mean())
)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print(
    model_base_df[
        [
            "date",
            "instrument",
            "primary_signal",
            "signal_changed",
            "signal_persistence",
            "signal_trend_concord",
            "signal_density_20d",
        ]
    ]
    .dropna()
    .head(10)
)

# Quick sanity check: distribution of signal_persistence by instrument
print("\nSignal persistence stats by instrument:")
print(
    model_base_df
    .groupby("instrument")["signal_persistence"]
    .describe()
    .round(1)
)

# ======================================================================
# TRIPLE-BARRIER LABELING (HMM-independent)
# ======================================================================

# --- [notebook cell 139] ------------------------------------------------
# ============================================================
# Phase 3.1 — Construct Active-Signal Meta-Labeling Dataset
# ============================================================

# Keep a full copy for reference, feature analysis, and possible regime work
full_model_df = model_base_df.copy()

# Meta-labeling is only meaningful when the primary model proposes a trade
meta_df = (
    model_base_df
    .loc[model_base_df["primary_signal"] != 0]
    .copy()
    .sort_values(["instrument", "date"])
    .reset_index(drop=True)
)

print("Full model dataframe shape:", full_model_df.shape)
print("Meta-labeling dataframe shape:", meta_df.shape)

print("\nPrimary signal distribution in full dataset:")
print(
    full_model_df["primary_signal"]
    .value_counts()
    .sort_index()
)

print("\nPrimary signal distribution in meta-labeling dataset:")
print(
    meta_df["primary_signal"]
    .value_counts()
    .sort_index()
)

print("\nActive signal distribution by instrument:")
print(
    pd.crosstab(
        meta_df["instrument"],
        meta_df["primary_signal"]
    )
)

# --- [notebook cell 141] ------------------------------------------------
# ============================================================
# Phase 3.2 — Temporal Structure Check
# ============================================================

for instrument, g in meta_df.groupby("instrument"):
    is_sorted = g["date"].is_monotonic_increasing
    print(f"{instrument}: sorted by date = {is_sorted}, n_active_signals = {len(g)}")

print("\nDate range by instrument:")
print(
    meta_df
    .groupby("instrument")["date"]
    .agg(["min", "max", "count"])
)

# --- [notebook cell 143] ------------------------------------------------
# ============================================================
# Phase 3.3 — Compare volatility estimators
# ============================================================

# Three estimators on cl1s (representative instrument)
ref_inst = meta_df[meta_df["instrument"] == "cl1s"].copy().sort_values("date").reset_index(drop=True)
ref_inst["log_ret"] = np.log(ref_inst["close"] / ref_inst["close"].shift(1))

vol_estimators = pd.DataFrame(index=ref_inst.index)
vol_estimators["date"] = ref_inst["date"].values
vol_estimators["realized_vol_20d"] = ref_inst["log_ret"].rolling(20).std()
vol_estimators["ewma_vol"] = ref_inst["log_ret"].ewm(alpha=1-0.94, adjust=False).std()

# Diagnostics: autocorrelation (stability), median, coefficient of variation
stats = []
for col in ["realized_vol_20d", "ewma_vol"]:
    v = vol_estimators[col].dropna()
    stats.append({
        "estimator": col,
        "median_vol": round(v.median(), 5),
        "cv (std/mean)": round(v.std() / v.mean(), 3),
        "autocorr_lag1": round(v.autocorr(lag=1), 3),
        "n_valid": v.notna().sum(),
    })

stats_df = pd.DataFrame(stats)
print("Volatility estimator comparison on cl1s:")
print(stats_df)

print("\nInterpretation:")
print("- High autocorrelation (>0.9) means the estimator is stable across nearby dates")
print("- Low CV means the estimator doesn't swing wildly day to day")
print("- realized_vol_20d is the most stable and interpretable baseline")

# --- [notebook cell 146] ------------------------------------------------
# ============================================================
# Phase 3.4 — Triple Barrier Labeling Function
# ============================================================

def apply_triple_barrier_to_instrument(
    instrument_df,
    price_col,
    vol_col,
    horizon=10,
    pt_mult=1.5,
    sl_mult=0.75
):
    """
    Apply triple-barrier labeling to one instrument at a time.

    Parameters
    ----------
    instrument_df : pd.DataFrame
        DataFrame containing only one instrument, sorted by date.
    price_col : str
        Column used as the trade entry price.
    vol_col : str
        Backward-looking volatility column used to scale barriers.
    horizon : int
        Maximum holding period in number of rows/trading days.
    pt_mult : float
        Profit-taking multiplier.
    sl_mult : float
        Stop-loss multiplier.

    Returns
    -------
    pd.DataFrame
        Same dataframe with triple-barrier outputs added.
    """

    df_inst = instrument_df.copy().sort_values("date").reset_index(drop=True)

    meta_labels = []
    event_types = []
    exit_dates = []
    exit_prices = []
    trade_returns = []

    prices = df_inst[price_col].values
    vols = df_inst[vol_col].values
    signals = df_inst["primary_signal"].values
    dates = df_inst["date"].values

    n = len(df_inst)

    for i in range(n):

        entry_price = prices[i]
        vol_t = vols[i]
        signal = signals[i]

        # If volatility or price is missing, we cannot define reliable barriers
        if pd.isna(entry_price) or pd.isna(vol_t) or vol_t <= 0:
            meta_labels.append(np.nan)
            event_types.append("missing_input")
            exit_dates.append(pd.NaT)
            exit_prices.append(np.nan)
            trade_returns.append(np.nan)
            continue

        # If there is not enough future data to reach the vertical barrier,
        # we cannot label the trade cleanly
        end_i = min(i + horizon, n - 1)

        if end_i == i:
            meta_labels.append(np.nan)
            event_types.append("no_future_data")
            exit_dates.append(pd.NaT)
            exit_prices.append(np.nan)
            trade_returns.append(np.nan)
            continue

        upper_barrier = entry_price * (1 + pt_mult * vol_t)
        lower_barrier = entry_price * (1 - sl_mult * vol_t)

        label = None
        event_type = None
        exit_i = None

        # Scan the future path until the vertical barrier
        for j in range(i + 1, end_i + 1):

            price_j = prices[j]

            if signal == 1:
                # Long trade
                if price_j >= upper_barrier:
                    label = 1
                    event_type = "profit_taking"
                    exit_i = j
                    break

                elif price_j <= lower_barrier:
                    label = 0
                    event_type = "stop_loss"
                    exit_i = j
                    break

            elif signal == -1:
                # Short trade
                if price_j <= lower_barrier:
                    label = 1
                    event_type = "profit_taking"
                    exit_i = j
                    break

                elif price_j >= upper_barrier:
                    label = 0
                    event_type = "stop_loss"
                    exit_i = j
                    break

        # If no horizontal barrier is hit, use vertical barrier outcome
        if label is None:
            exit_i = end_i
            exit_price = prices[exit_i]

            realized_trade_return = signal * (exit_price / entry_price - 1)

            label = int(realized_trade_return > 0)
            event_type = "vertical_barrier"

        else:
            exit_price = prices[exit_i]
            realized_trade_return = signal * (exit_price / entry_price - 1)

        meta_labels.append(label)
        event_types.append(event_type)
        exit_dates.append(dates[exit_i])
        exit_prices.append(exit_price)
        trade_returns.append(realized_trade_return)

    df_inst["meta_label"] = meta_labels
    df_inst["tb_event_type"] = event_types
    df_inst["tb_exit_date"] = exit_dates
    df_inst["tb_exit_price"] = exit_prices
    df_inst["tb_trade_return"] = trade_returns

    return df_inst

# --- [notebook cell 149] ------------------------------------------------
# ============================================================
# Phase 3.5 — Sensitivity grid for barrier parameters
# ============================================================

import itertools
import numpy as np

# Fine-resolution grid using step intervals rather than fixed values
PT_GRID = np.round(np.arange(1.0, 3.1, 0.25), 2)   # 1.00, 1.25, 1.50, ..., 3.00  (9 values)
SL_GRID = np.round(np.arange(0.5, 1.6, 0.25), 2)   # 0.50, 0.75, 1.00, 1.25, 1.50  (5 values)
H_GRID  = [5, 7, 10, 12, 15]                        # holding horizons (integers)

# Weighted scoring — class balance dominates because it matters most for classifier training
WEIGHTS = {
    "balance":      0.45,
    "inst_balance": 0.15,
    "timeout":      0.15,
    "holding":      0.10,
    "realism":      0.15,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6

# Close-only barrier detection (per professor's guidance)
price_col_grid = "close"
vol_col_grid = "realized_vol_20d"

grid_results = []
n_total = len(PT_GRID) * len(SL_GRID) * len(H_GRID)
print(f"Testing {n_total} parameter combinations...")
print(f"  pt_mult: {PT_GRID.min():.2f} to {PT_GRID.max():.2f}, step 0.25")
print(f"  sl_mult: {SL_GRID.min():.2f} to {SL_GRID.max():.2f}, step 0.25")
print(f"  horizon: {H_GRID}")

for pt, sl, h in itertools.product(PT_GRID, SL_GRID, H_GRID):

    labeled_groups = []
    for instrument, g in meta_df.groupby("instrument"):
        g = g.copy()
        labeled_g = apply_triple_barrier_to_instrument(
            g, price_col=price_col_grid, vol_col=vol_col_grid,
            horizon=h, pt_mult=float(pt), sl_mult=float(sl),
        )
        labeled_g["instrument"] = instrument
        labeled_groups.append(labeled_g)

    grid_df = (
        pd.concat(labeled_groups, axis=0)
        .dropna(subset=["meta_label"])
        .reset_index(drop=True)
    )

    if len(grid_df) == 0:
        continue

    # Aggregate diagnostics
    pos_rate = grid_df["meta_label"].mean()
    pct_vertical = grid_df["tb_event_type"].value_counts(normalize=True).get("vertical_barrier", 0.0)
    grid_df["holding_days"] = (
        pd.to_datetime(grid_df["tb_exit_date"]) - pd.to_datetime(grid_df["date"])
    ).dt.days
    median_holding = grid_df["holding_days"].median()

    # Sub-scores
    balance_score = 1 - abs(0.5 - pos_rate) * 2
    timeout_score = 1 - min(pct_vertical / 0.3, 1.0)
    holding_score = 1 - abs(median_holding - h / 2) / h

    # Per-instrument balance
    inst_pos = grid_df.groupby("instrument")["meta_label"].mean()
    worst_imbalance = (inst_pos - 0.5).abs().max()
    inst_balance_score = 1 - min(worst_imbalance * 2, 1.0)

    # Realism — penalise extreme reward-risk ratios
    rr = pt / sl
    realism_score = 1.0 if rr <= 2.5 else max(0, 1 - (rr - 2.5) / 2)

    composite = (
        WEIGHTS["balance"]      * balance_score
        + WEIGHTS["inst_balance"] * inst_balance_score
        + WEIGHTS["timeout"]    * timeout_score
        + WEIGHTS["holding"]    * holding_score
        + WEIGHTS["realism"]    * realism_score
    )

    grid_results.append({
        "pt_mult": float(pt),
        "sl_mult": float(sl),
        "h": h,
        "reward_risk_ratio": round(rr, 2),
        "n_events": len(grid_df),
        "pct_positive": round(pos_rate, 3),
        "pct_vertical_barrier": round(pct_vertical, 3),
        "median_holding": round(median_holding, 1),
        "worst_inst_imbalance": round(worst_imbalance, 3),
        "balance": round(balance_score, 3),
        "inst_bal": round(inst_balance_score, 3),
        "timeout": round(timeout_score, 3),
        "holding": round(holding_score, 3),
        "realism": round(realism_score, 3),
        "composite": round(composite, 3),
    })

grid_results_df = pd.DataFrame(grid_results)

print(f"\nCompleted {len(grid_results_df)} combinations.")
print("\nTop 10 by weighted composite score:")
print(grid_results_df.sort_values("composite", ascending=False).head(10))

# --- [notebook cell 151] ------------------------------------------------
# ============================================================
# Phase 3.5 — Initial Triple Barrier Parameters
# ============================================================

# Entry price used for trades
price_col = "close"

# Volatility estimate
vol_col = "realized_vol_20d"

# Vertical barrier
HORIZON = 10

# Horizontal barriers
PT_MULT = 1.5
SL_MULT = 0.75


print("Selected baseline parameters")
print("--------------------------------")
print("Price column:", price_col)
print("Volatility:", vol_col)
print("Holding horizon:", HORIZON)
print("PT multiplier:", PT_MULT)
print("SL multiplier:", SL_MULT)

# --- [notebook cell 153] ------------------------------------------------
# ============================================================
# Phase 3.6 — Apply Triple Barrier Labeling
# ============================================================

labeled_groups = []

for instrument, g in meta_df.groupby("instrument"):
    g = g.copy()
    g["instrument"] = instrument  # keep instrument explicitly as a column
    
    labeled_g = apply_triple_barrier_to_instrument(
        g,
        price_col=price_col,
        vol_col=vol_col,
        horizon=HORIZON,
        pt_mult=PT_MULT,
        sl_mult=SL_MULT
    )
    
    labeled_g["instrument"] = instrument
    labeled_groups.append(labeled_g)

meta_labeled_df = (
    pd.concat(labeled_groups, axis=0)
    .sort_values(["instrument", "date"])
    .reset_index(drop=True)
)

print("Final labeled dataset shape:")
print(meta_labeled_df.shape)

print("\nColumns check:")
print("instrument" in meta_labeled_df.columns)

print("\nMeta-label distribution:")
print(
    meta_labeled_df["meta_label"]
    .value_counts(dropna=False)
)

print("\nBarrier event distribution:")
print(
    meta_labeled_df["tb_event_type"]
    .value_counts(dropna=False)
)

# --- [notebook cell 154] ------------------------------------------------
print(
    meta_df[
        meta_labeled_df["tb_event_type"]=="no_future_data"
    ][
        [
            "instrument",
            "date",
            "primary_signal"
        ]
    ]
)

# --- [notebook cell 156] ------------------------------------------------
# ============================================================
# Phase 3.7 — Remove Unlabelable Observations
# ============================================================

n_before = len(meta_labeled_df)

meta_labeled_df = (
    meta_labeled_df
    .dropna(subset=["meta_label"])
    .reset_index(drop=True)
)

n_after = len(meta_labeled_df)

print("Rows before cleaning:", n_before)
print("Rows after cleaning:", n_after)

print(
    f"\nRemoved observations: {n_before-n_after}"
)

print(
    f"Percentage removed: "
    f"{100*(n_before-n_after)/n_before:.3f}%"
)

# --- [notebook cell 157] ------------------------------------------------
# ============================================================
# Phase 3.8 — Label Distribution by Instrument
# ============================================================

print("Meta-label distribution by instrument:")

print(
    pd.crosstab(
        meta_labeled_df["instrument"],
        meta_labeled_df["meta_label"],
        margins=True
    )
)

print("\nPercentages by instrument:")

print(
    pd.crosstab(
        meta_labeled_df["instrument"],
        meta_labeled_df["meta_label"],
        normalize="index"
    ).round(3)
)

# --- [notebook cell 158] ------------------------------------------------
# ============================================================
# Phase 3.9 — Barrier Event Distribution by Instrument
# ============================================================

print("Barrier event distribution by instrument:")

print(
    pd.crosstab(
        meta_labeled_df["instrument"],
        meta_labeled_df["tb_event_type"],
        margins=True
    )
)

print("\nPercentages by instrument:")

print(
    pd.crosstab(
        meta_labeled_df["instrument"],
        meta_labeled_df["tb_event_type"],
        normalize="index"
    ).round(3)
)

# ============================================================
# CACHE 2 : merged + labeled subset (NO hmm_* columns)
# ============================================================
meta_labeled_df.to_parquet("labeled_base.parquet", index=False)
print("Saved labeled_base.parquet  shape:", meta_labeled_df.shape)
