"""
python run_hmm.py
python evaluate_model.py
run_hmm.py  --  RE-RUN PER EXPERIMENT (fast: HMM only).

Loads meta_labeled_dataset.csv, drops the old HMM columns, trains BOTH HMMs
and writes:
  * hmm_features_<TAG>.parquet : [date, instrument, hmm_*]  (+ macro hmm cols)

Change HMM_FEATURES / MACRO_HMM_FEATURES / N_HMM_STATES below, re-run, then
run evaluate_model.py with the same HMM_TAG. The heavy base features are never
recomputed.
"""
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# ============================================================
# CONFIG  --  edit these between experiments
# ============================================================
HMM_FEATURES =   ["log_return", "realized_vol_20d", "skew_20d", "kurt_20d"]

MACRO_HMM_FEATURES =  ["multiasset_vol_index", "risk_on_score", "dxy_ret_5d", "copper_gold_ratio"]
N_HMM_STATES       = 3
N_MACRO_HMM_STATES = 2                  # macro regime typically needs fewer states
HMM_TAG            = "baseline"         # label appended to outputs
# ============================================================

meta = json.load(open("split_meta.json"))
ENERGY_INSTRUMENTS    = meta["ENERGY_INSTRUMENTS"]
global_train_end_date = pd.Timestamp(meta["global_train_end_date"])

# ============================================================
# LOAD CSV
# ============================================================
print("Loading meta_labeled_dataset.csv ...")
base = pd.read_csv("meta_labeled_dataset.csv", parse_dates=["date"])
print(f"  Loaded {base.shape[0]:,} rows x {base.shape[1]} cols")

# ============================================================
# DROP OLD HMM COLUMNS so we recompute them cleanly
# ============================================================
OLD_HMM_COLS = [c for c in base.columns if c.startswith("hmm_")]
if OLD_HMM_COLS:
    print(f"  Dropping old HMM columns: {OLD_HMM_COLS}")
    base = base.drop(columns=OLD_HMM_COLS)

features_df = base.copy()

# ======================================================================
# PART 1 — PER-INSTRUMENT HMM
# ======================================================================
print(f"\n{'='*60}")
print(f"Per-instrument HMM   features={HMM_FEATURES}   k={N_HMM_STATES}")
print(f"{'='*60}")

# Initialise output columns
features_df["hmm_regime"] = np.nan
for state in range(N_HMM_STATES):
    features_df[f"hmm_prob_state_{state}"] = np.nan

hmm_models  = {}
hmm_scalers = {}

for instrument in ENERGY_INSTRUMENTS:

    print(f"\n  Training per-instrument HMM for {instrument} ...")

    instrument_df = (
        features_df
        .loc[features_df["instrument"] == instrument]
        .sort_values("date")
        .copy()
    )

    # Drop rows where any HMM input is missing
    instrument_df = instrument_df.dropna(subset=HMM_FEATURES)

    train_df = instrument_df.loc[instrument_df["date"] <= global_train_end_date]

    X_train = train_df[HMM_FEATURES].values
    X_full  = instrument_df[HMM_FEATURES].values

    # Fit scaler on TRAIN only — no look-ahead
    scaler = StandardScaler().fit(X_train)
    hmm_scalers[instrument] = scaler

    X_train_scaled = scaler.transform(X_train)
    X_full_scaled  = scaler.transform(X_full)

    hmm = GaussianHMM(
        n_components=N_HMM_STATES,
        covariance_type="full",
        n_iter=300,
        random_state=42,
    )
    hmm.fit(X_train_scaled)
    hmm_models[instrument] = hmm

    hidden_states = hmm.predict(X_full_scaled)
    state_probs   = hmm.predict_proba(X_full_scaled)

    features_df.loc[instrument_df.index, "hmm_regime"] = hidden_states
    for state in range(N_HMM_STATES):
        features_df.loc[instrument_df.index, f"hmm_prob_state_{state}"] = (
            state_probs[:, state]
        )

    ll_train = hmm.score(X_train_scaled) / len(X_train_scaled)
    print(f"    Done — train LL/obs: {ll_train:.4f}")

# Sanity check: prob sums
features_df["hmm_prob_sum"] = sum(
    features_df[f"hmm_prob_state_{s}"] for s in range(N_HMM_STATES)
)
print(f"\nPer-instrument HMM prob sum — min: {features_df['hmm_prob_sum'].min():.6f}  "
      f"max: {features_df['hmm_prob_sum'].max():.6f}")

# ======================================================================
# PART 2 — MACRO HMM (one model, cross-instrument)
# ======================================================================
print(f"\n{'='*60}")
print(f"Macro HMM   features={MACRO_HMM_FEATURES}   k={N_MACRO_HMM_STATES}")
print(f"{'='*60}")

# Macro features are date-level (identical across instruments on a given date),
# so build the macro series from ALL dates — one row per date, from whichever
# instrument has the values. Using a single representative instrument with a
# shorter history left ~22% of dates unmatched on the broadcast merge.
macro_df = (
    features_df
    .dropna(subset=MACRO_HMM_FEATURES)
    .sort_values("date")
    .drop_duplicates(subset="date", keep="first")
    .copy()
)

macro_train_df = macro_df.loc[macro_df["date"] <= global_train_end_date]

X_macro_train = macro_train_df[MACRO_HMM_FEATURES].values
X_macro_full  = macro_df[MACRO_HMM_FEATURES].values

macro_scaler = StandardScaler().fit(X_macro_train)
X_macro_train_scaled = macro_scaler.transform(X_macro_train)
X_macro_full_scaled  = macro_scaler.transform(X_macro_full)

macro_hmm = GaussianHMM(
    n_components=N_MACRO_HMM_STATES,
    covariance_type="full",
    n_iter=300,
    random_state=42,
)
macro_hmm.fit(X_macro_train_scaled)

macro_hidden = macro_hmm.predict(X_macro_full_scaled)
macro_probs  = macro_hmm.predict_proba(X_macro_full_scaled)

ll_macro = macro_hmm.score(X_macro_train_scaled) / len(X_macro_train_scaled)
print(f"  Done — train LL/obs: {ll_macro:.4f}")

# Build a date → macro regime mapping, then broadcast to ALL instruments
macro_date_map = pd.DataFrame({
    "date":            macro_df["date"].values,
    "hmm_macro_regime": macro_hidden,
})
for state in range(N_MACRO_HMM_STATES):
    macro_date_map[f"hmm_macro_prob_state_{state}"] = macro_probs[:, state]

macro_date_map["hmm_macro_prob_sum"] = sum(
    macro_date_map[f"hmm_macro_prob_state_{s}"] for s in range(N_MACRO_HMM_STATES)
)

# Merge back on date so every instrument row gets the same macro regime
features_df = features_df.merge(macro_date_map, on="date", how="left")

print(f"Macro HMM prob sum — min: {features_df['hmm_macro_prob_sum'].min():.6f}  "
      f"max: {features_df['hmm_macro_prob_sum'].max():.6f}")

# ======================================================================
# EXPORT — only hmm_* columns keyed by (date, instrument)
# ======================================================================
hmm_cols = [c for c in features_df.columns if c.startswith("hmm_")]
out = features_df[["date", "instrument"] + hmm_cols].copy()
out_path = f"hmm_features_{HMM_TAG}.parquet"
out.to_parquet(out_path, index=False)
print(f"\nSaved {out_path}")
print(f"  Columns : {hmm_cols}")
print(f"  Rows    : {len(out):,}")