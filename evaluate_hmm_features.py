"""
evaluate_hmm_features.py  --  RE-RUN PER HMM EXPERIMENT (no Optuna).

Same role as evaluate_model.py, BUT the LightGBM hyperparameters are FIXED to the
tuned values already found by the notebook (cell 196, robust 80-trial study).
No hyperparameter search is performed. This:
  * isolates the effect of the HMM features (no confounding re-tuning), and
  * runs much faster (skips the Optuna step entirely).

Workflow:
    edit run_hmm.py  (HMM_FEATURES / N_HMM_STATES / HMM_TAG)
    python run_hmm.py
    python evaluate_hmm_features.py        # set HMM_TAG to match

Outputs (per HMM_TAG):
  * lgb_model_<tag>.joblib              : the trained LightGBM (fixed params)
  * importance_cluster_<tag>.parquet/csv: cluster-level MDA
  * importance_individual_<tag>.parquet/csv: per-feature MDA
  * hmm_importance_<tag>.csv            : just the hmm_* rows, ranked
"""
import os, json, warnings, joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from itertools import combinations

RANDOM_STATE = 42
LAMBDA_RISK  = 0.5   # inter-fold dispersion penalty: robust = mean(AUC) - LAMBDA_RISK*std(AUC)

# ============================================================
# CONFIG  --  must match the HMM_TAG used in run_hmm.py
# ============================================================
HMM_TAG = "baseline"

# Tuned LightGBM hyperparameters from the notebook (cell 196, robust study).
# Loaded from JSON if present, otherwise these defaults are used.
FIXED_PARAMS_FILE = "best_lgb_params_tuned.json"
DEFAULT_TUNED_PARAMS = {
    "num_leaves":        13,
    "max_depth":         6,
    "learning_rate":     0.01963796059333161,
    "n_estimators":      616,
    "min_child_samples": 27,
    "subsample":         0.8496647060377931,
    "colsample_bytree":  0.9602072630324785,
    "reg_alpha":         8.020045867492858,
    "reg_lambda":        0.0035269228944229884,
}
# ============================================================

if os.path.exists(FIXED_PARAMS_FILE):
    best_params_lgb = json.load(open(FIXED_PARAMS_FILE))
    print(f"Loaded fixed hyperparameters from {FIXED_PARAMS_FILE}")
else:
    best_params_lgb = DEFAULT_TUNED_PARAMS
    print("Using built-in tuned hyperparameters (notebook cell 196)")

# Mirror the notebook's refit config (cell 223): wrap the tuned params with the
# fixed training settings. metric is irrelevant for a fitted classifier's AUC.
lgb_full = dict(objective="binary", metric="binary_logloss", verbosity=-1,
                random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
                **best_params_lgb)

# ============================================================
# LOAD
# ============================================================
labeled = pd.read_parquet("labeled_base.parquet")
hmm     = pd.read_parquet(f"hmm_features_{HMM_TAG}.parquet")
meta    = json.load(open("split_meta.json"))
global_train_end_date = pd.Timestamp(meta["global_train_end_date"])

meta_labeled_df = labeled.merge(hmm, on=["date", "instrument"], how="left")

# ======================================================================
# PHASE 4 PREP : target, leakage removal, split, one-hot, preprocessor
# ======================================================================
ml_df = meta_labeled_df.copy()
target_col = "meta_label"

LEAK_AND_ID = [
    "date", "meta_label",
    "tb_event_type", "tb_exit_date", "tb_exit_price", "tb_trade_return",
    "hmm_prob_sum",
]
RAW_PRICE = ["open", "high", "low", "close", "volume", "open_interest"]

ml_df = ml_df.replace([np.inf, -np.inf], np.nan)

META_FEATURES = [c for c in ml_df.columns if c not in LEAK_AND_ID and c not in RAW_PRICE]
print(f"TOTAL META_FEATURES          : {len(META_FEATURES)}")

n_total = len(ml_df)
missing_counts = ml_df[META_FEATURES].isna().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
high_missing = missing_counts[missing_counts > 0.01 * n_total].index.tolist()

ml_df = ml_df[[c for c in ml_df.columns if c not in high_missing]]
META_FEATURES = [c for c in META_FEATURES if c not in high_missing]

train_df = ml_df[ml_df["date"] <= global_train_end_date]
test_df  = ml_df[ml_df["date"] >  global_train_end_date]

X_train = train_df[META_FEATURES]
y_train = train_df["meta_label"].values.astype(int)
X_test  = test_df[META_FEATURES]
y_test  = test_df["meta_label"].values.astype(int)

categorical_features = ["instrument"]
numerical_features = [c for c in META_FEATURES if c not in categorical_features]

X_train = pd.get_dummies(X_train, columns=categorical_features, dtype=float)
X_test  = pd.get_dummies(X_test,  columns=categorical_features, dtype=float)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
META_FEATURES = list(X_train.columns)

numeric_scaled_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
])
scaled_preprocessor = ColumnTransformer(
    transformers=[("num", numeric_scaled_pipeline, numerical_features)],
    remainder="passthrough",
)
print("Preprocessor ready.")

# ======================================================================
# CPCV  (Combinatorial Purged CV — Prado, identical to evaluate_model.py)
# ======================================================================
class CombinatorialPurgedCV:
    def __init__(self, n_splits=6, n_test_groups=2, embargo_pct=0.01):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n = len(X)
        embargo = int(n * self.embargo_pct)
        indices = np.arange(n)
        group_bounds = np.array_split(indices, self.n_splits)
        for test_group_ids in combinations(range(self.n_splits), self.n_test_groups):
            test_idx = np.concatenate([group_bounds[g] for g in test_group_ids])
            test_set = set(test_idx)
            embargoed = set()
            for g in test_group_ids:
                lo = max(group_bounds[g][0] - embargo, 0)
                hi = min(group_bounds[g][-1] + embargo + 1, n)
                embargoed.update(range(lo, hi))
            train_idx = np.array(sorted(set(indices) - embargoed - test_set))
            yield train_idx, test_idx

cpcv = CombinatorialPurgedCV(n_splits=5, n_test_groups=2, embargo_pct=0.01)

# ======================================================================
# SEMANTIC CLUSTERS  (identical to evaluate_model.py / notebook cell 209)
# ======================================================================
semantic_rules = [
    ("Regimes (HMM)", lambda f: f.startswith("hmm_")),
    ("Signal interaction", lambda f: any(k in f for k in [
        "signal_changed", "signal_persistence", "signal_trend_concord", "signal_density",
    ])),
    ("Seasonality", lambda f: any(k in f for k in [
        "heating_season", "driving_season", "hurricane_season",
        "quarter_progress", "day_of_year_sin", "day_of_year_cos",
    ])),
    ("Macro environment", lambda f: any(k in f for k in [
        "vix_", "dxy_", "us10y", "multiasset_vol", "equity_avg", "equity_vol",
        "equity_ret_daily", "risk_on_score", "hg_ret_daily",
    ])),
    ("Energy spreads", lambda f: any(k in f for k in [
        "crack_321", "oil_gas_ratio", "relative_energy_vol",
    ])),
    ("Metals/Commodities", lambda f: any(k in f for k in [
        "copper", "gold_silver", "copper_gold",
    ])),
    ("Cross-asset correlation", lambda f: any(k in f for k in [
        "corr_basket", "leadlag_anchor", "beta_basket",
        "asset_copper_corr", "asset_gold_corr", "asset_equity_corr",
        "cross_asset_momentum_concordance",
    ])),
    ("Relative strength", lambda f: any(k in f for k in [
        "relative_vol", "relative_momentum", "momentum_rank",
        "sector_momentum", "sector_vol",
    ])),
    ("Volume & Liquidity", lambda f: any(k in f for k in [
        "obv", "mfi", "volume_", "oi_change", "oi_momentum", "vol_oi",
        "dollar_volume", "log_volume", "log_oi",
        "amihud", "kyle_lambda",
        "roll_spread", "bid_ask", "oc_spread", "hl_spread",
    ])),
    ("Volatility", lambda f: any(k in f for k in [
        "vol_5", "vol_10", "vol_20", "vol_60",
        "realized_vol", "vol_parkinson", "vol_garman", "vol_rogers",
        "yang_zhang", "atr_", "atx_", "bb_width", "bb_position",
        "vol_ratio_20_60", "ewma_vol", "downside_vol", "garman_klass", "vol_change",
    ])),
    ("Momentum/Trend", lambda f: any(k in f for k in [
        "momentum_", "macd", "rsi_", "stoch", "willr", "adx",
        "close_to_sma", "distance_from_200d", "trend_",
    ])),
    ("Returns/Price", lambda f: any(k in f for k in [
        "returns", "log_return", "mean_return", "abs_return",
        "close_position", "price_zscore", "close_fracdiff",
        "positive_return", "ret_", "price_change",
        "price_range_position", "range_position",
    ])),
    ("Distribution/Complexity", lambda f: any(k in f for k in [
        "skew", "kurt", "autocorr", "shannon_entropy", "lz_complexity", "sadf",
        "hurst", "dfa", "spectral_entropy", "approx_entropy", "dominant_cycle",
        "return_vol_correl",
    ])),
    ("Signal & Instrument", lambda f: (
        f == "primary_signal"
        or f.startswith(("is_", "inst_"))
        or f == "instrument"
    )),
]

def assign_cluster(feat):
    for name, rule in semantic_rules:
        if rule(feat):
            return name
    return "Other"

feature_clusters = pd.Series({f: assign_cluster(f) for f in META_FEATURES}, name="cluster")
clusters = {}
for feat, cl in feature_clusters.items():
    clusters.setdefault(cl, []).append(feat)

assigned = sum(len(v) for v in clusters.values())
assert assigned == len(META_FEATURES), f"Mismatch: {assigned} vs {len(META_FEATURES)}"
if "Other" in clusters:
    print(f"⚠ 'Other' cluster contents: {clusters['Other']}")
else:
    print("✓ No feature fell into 'Other'.")

# ======================================================================
# FIT the official LightGBM on full train  (FIXED params — no Optuna)
# ======================================================================
print(f"\nFitting LightGBM with fixed tuned params:\n  {best_params_lgb}")
best_lgb = lgb.LGBMClassifier(**lgb_full)
best_lgb.fit(X_train, y_train)
joblib.dump(best_lgb, f"lgb_model_{HMM_TAG}.joblib")
joblib.dump(scaled_preprocessor, f"preprocessor_{HMM_TAG}.joblib")
print(f"✓ Saved lgb_model_{HMM_TAG}.joblib  and  preprocessor_{HMM_TAG}.joblib")

# ======================================================================
# CLUSTER-LEVEL MDA  (joint permutation, frozen per-fold models)
# ======================================================================
N_REPEATS = 5
rng = np.random.default_rng(RANDOM_STATE)
cluster_names = list(clusters.keys())
drops = {c: [] for c in cluster_names}
fold_aucs = []   # unpermuted held-out AUC per CPCV fold -> robust CPCV AUC

for tr, val in cpcv.split(X_train):
    pipe = Pipeline([("preprocessor", clone(scaled_preprocessor)),
                     ("model", lgb.LGBMClassifier(**lgb_full))])
    pipe.fit(X_train.iloc[tr], y_train[tr])
    yv = y_train[val]; Xv = X_train.iloc[val]
    base_auc = roc_auc_score(yv, pipe.predict_proba(Xv)[:, 1])
    fold_aucs.append(base_auc)
    for cl in cluster_names:
        cols = [c for c in clusters[cl] if c in Xv.columns]
        if not cols:
            drops[cl].append(0.0); continue
        d = []
        for _ in range(N_REPEATS):
            perm = rng.permutation(len(val))
            Xp = Xv.copy(); Xp[cols] = Xp[cols].values[perm]
            d.append(base_auc - roc_auc_score(yv, pipe.predict_proba(Xp)[:, 1]))
        drops[cl].append(np.mean(d))

# Robust CPCV AUC from the fixed-param model (same metric the notebook's Optuna
# maximized): mean of per-fold held-out AUC, penalized by inter-fold dispersion.
fold_aucs = np.array(fold_aucs)
robust_cpcv_auc = fold_aucs.mean() - LAMBDA_RISK * fold_aucs.std(ddof=1)
print(f"\nRobust CPCV AUC: {robust_cpcv_auc:.4f}  "
      f"(mean {fold_aucs.mean():.4f} - {LAMBDA_RISK}*std {fold_aucs.std(ddof=1):.4f}, "
      f"{len(fold_aucs)} folds)")

imp_cluster = pd.DataFrame({
    "mda_mean": {c: np.mean(drops[c]) for c in cluster_names},
    "mda_std":  {c: np.std(drops[c], ddof=1) for c in cluster_names},
}).sort_values("mda_mean", ascending=False)
print("\n" + "="*60 + "\nCLUSTER-LEVEL MDA (drop in CPCV AUC)\n" + "="*60)
print(imp_cluster.round(4).to_string())
imp_cluster.to_csv(f"importance_cluster_{HMM_TAG}.csv")
imp_cluster.to_parquet(f"importance_cluster_{HMM_TAG}.parquet")

# ======================================================================
# INDIVIDUAL MDA  (single-feature permutation, frozen per-fold models)
# ======================================================================
feat_drops = {f: [] for f in X_train.columns}
for tr, val in cpcv.split(X_train):
    pipe = Pipeline([("preprocessor", clone(scaled_preprocessor)),
                     ("model", lgb.LGBMClassifier(**lgb_full))])
    pipe.fit(X_train.iloc[tr], y_train[tr])
    yv = y_train[val]; Xv = X_train.iloc[val]
    base_auc = roc_auc_score(yv, pipe.predict_proba(Xv)[:, 1])
    for f in X_train.columns:
        d = []
        for _ in range(N_REPEATS):
            perm = rng.permutation(len(val))
            Xp = Xv.copy(); Xp[f] = Xp[f].values[perm]
            d.append(base_auc - roc_auc_score(yv, pipe.predict_proba(Xp)[:, 1]))
        feat_drops[f].append(np.mean(d))

imp_indiv = (pd.Series({f: np.mean(v) for f, v in feat_drops.items()}, name="mda")
             .sort_values(ascending=False).to_frame())
print("\n" + "="*60 + "\nINDIVIDUAL MDA — top 20 features\n" + "="*60)
print(imp_indiv.head(20).round(4).to_string())
imp_indiv.to_csv(f"importance_individual_{HMM_TAG}.csv")
imp_indiv.to_parquet(f"importance_individual_{HMM_TAG}.parquet")

# ======================================================================
# HMM FEATURE RANKING  (the thing you actually want to compare across tags)
# ======================================================================
hmm_rows = imp_indiv[imp_indiv.index.str.startswith("hmm_")].sort_values("mda", ascending=False)
print("\n" + "="*60 + f"\nHMM FEATURES — tag '{HMM_TAG}'\n" + "="*60)
if len(hmm_rows):
    print(hmm_rows.round(5).to_string())
    print(f"\nTotal HMM MDA (sum): {hmm_rows['mda'].sum():.5f}")
    if "Regimes (HMM)" in imp_cluster.index:
        r = imp_cluster.loc["Regimes (HMM)"]
        print(f"HMM cluster MDA    : {r['mda_mean']:.5f} ± {r['mda_std']:.5f}")
else:
    print("No hmm_* features found.")
hmm_rows.to_csv(f"hmm_importance_{HMM_TAG}.csv")

print(f"\nDone. Re-run with a different HMM_TAG to compare configurations.")