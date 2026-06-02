"""
run_hmm.py  --  RE-RUN PER EXPERIMENT (fast: HMM only).

Loads base_features_full.parquet, trains BOTH HMMs (per-instrument + macro) with the
inputs configured in the CONFIG block below, and writes:
  * hmm_features.parquet : [date, instrument, hmm_*]  (+ macro hmm cols)

Change HMM_FEATURES / MACRO_HMM_FEATURES / N_HMM_STATES below, re-run, then run
evaluate_model.py. The heavy base features are never recomputed.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# ============================================================
# CONFIG  --  edit these between experiments
# ============================================================
HMM_FEATURES       = ["log_return", "realized_vol_20d", "momentum_20d"]   # per-instrument HMM inputs
MACRO_HMM_FEATURES = None      # set a list to override the macro-HMM inputs; None = use notebook default
N_HMM_STATES       = 3
HMM_TAG            = "baseline" # label appended to outputs so experiments don't overwrite each other
# ============================================================

base = pd.read_parquet("base_features_full.parquet")
meta = json.load(open("split_meta.json"))
ENERGY_INSTRUMENTS  = meta["ENERGY_INSTRUMENTS"]
global_train_end_date = pd.Timestamp(meta["global_train_end_date"])
features_df = base.copy()          # the HMM cells below mutate `features_df`


# ======================================================================
# TRAIN HMMs (per-instrument + macro)
# ======================================================================

# --- [notebook cell 116] ------------------------------------------------
# ============================================================
# Initialize HMM outputs
# ============================================================
features_df["hmm_regime"] = np.nan

for state in range(N_HMM_STATES):
    features_df[f"hmm_prob_state_{state}"] = np.nan

# --- [notebook cell 117] ------------------------------------------------
# ============================================================
# Train HMMs and infer regimes
# ============================================================

hmm_models = {}
hmm_scalers = {}

for instrument in ENERGY_INSTRUMENTS:
    
    print(f"\nTraining HMM for {instrument}...")
    
    # --------------------------------------------------------
    # Instrument subset
    # --------------------------------------------------------
    
    instrument_df = (
        features_df
        .loc[features_df["instrument"] == instrument]
        .sort_values("date")
        .copy()
    )
    
    # --------------------------------------------------------
    # Keep only rows with valid HMM features
    # --------------------------------------------------------
    
    instrument_df = instrument_df.dropna(subset=HMM_FEATURES)
    
    # --------------------------------------------------------
    # Train/test cutoff
    # --------------------------------------------------------
    
    train_end_date = global_train_end_date
    
    train_mask = instrument_df["date"] <= train_end_date
    
    train_df = instrument_df.loc[train_mask]
    
    # --------------------------------------------------------
    # Extract matrices
    # --------------------------------------------------------
    
    X_train = train_df[HMM_FEATURES].values
    X_full = instrument_df[HMM_FEATURES].values
    
    # --------------------------------------------------------
    # Standardize using TRAIN ONLY
    # --------------------------------------------------------
    
    scaler = StandardScaler()
    
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_full_scaled = scaler.transform(X_full)
    
    hmm_scalers[instrument] = scaler
    
    # --------------------------------------------------------
    # Fit Gaussian HMM
    # --------------------------------------------------------
    
    hmm = GaussianHMM(
        n_components=N_HMM_STATES,
        covariance_type="full",
        n_iter=300,
        random_state=42
    )
    
    hmm.fit(X_train_scaled)
    
    hmm_models[instrument] = hmm
    
    # --------------------------------------------------------
    # Infer regimes on full sample
    # --------------------------------------------------------
    
    hidden_states = hmm.predict(X_full_scaled)
    
    state_probs = hmm.predict_proba(X_full_scaled)
    
    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------
    
    features_df.loc[instrument_df.index, "hmm_regime"] = hidden_states
    
    for state in range(N_HMM_STATES):
        
        features_df.loc[
            instrument_df.index,
            f"hmm_prob_state_{state}"
        ] = state_probs[:, state]
    
    print("Done.")

# --- [notebook cell 119] ------------------------------------------------
# ============================================================
# Inspect HMM feature columns
# ============================================================

hmm_cols = [
    "hmm_regime",
    "hmm_prob_state_0",
    "hmm_prob_state_1",
    "hmm_prob_state_2"
]

print(
    features_df[
        ["date", "instrument"] + HMM_FEATURES + hmm_cols
    ]
    .dropna(subset=hmm_cols)
    .head(20)
)

# --- [notebook cell 120] ------------------------------------------------
# ============================================================
# Check HMM probability sums
# ============================================================

features_df["hmm_prob_sum"] = (
    features_df["hmm_prob_state_0"]
    + features_df["hmm_prob_state_1"]
    + features_df["hmm_prob_state_2"]
)

print(
    features_df[
        ["date", "instrument", "hmm_prob_sum"]
    ]
    .dropna()
    .head(20)
)

print(
    "Min probability sum:",
    features_df["hmm_prob_sum"].min()
)

print(
    "Max probability sum:",
    features_df["hmm_prob_sum"].max()
)

# ============================================================
# EXPORT only the hmm_* columns keyed by (date, instrument)
# ============================================================
hmm_cols = [c for c in features_df.columns if c.startswith("hmm_")]
out = features_df[["date", "instrument"] + hmm_cols].copy()
out.to_parquet(f"hmm_features_{HMM_TAG}.parquet", index=False)
print(f"Saved hmm_features_{HMM_TAG}.parquet  cols:", hmm_cols)
