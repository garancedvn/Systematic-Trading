"""
evaluate_model_with_model_save.py  --  ENHANCED VERSION with model persistence

This is an enhanced version of evaluate_model.py that:
1. Saves the trained LightGBM model to parquet + joblib
2. Exports model hyperparameters to parquet
3. Exports feature importance (cluster + individual) to parquet
4. Allows quick loading of the model without re-training

Outputs (per HMM_TAG):
  * lgb_model_{tag}.joblib            : trained LightGBM classifier (pickled)
  * lgb_model_{tag}.pkl               : alternative pickle format
  * best_lgb_params_{tag}.parquet     : hyperparameters as parquet (easy to reload)
  * importance_cluster_{tag}.parquet  : cluster-level MDA 
  * importance_individual_{tag}.parquet: per-feature MDA (top features)
  * model_metadata_{tag}.parquet      : model shape, feature names, performance metrics
"""
import os, json, warnings, joblib, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from itertools import combinations

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42
LAMBDA_RISK  = 0.5
N_TRIALS     = 25

# ============================================================
# CONFIG  --  must match the HMM_TAG used in run_hmm.py
# ============================================================
HMM_TAG = "baseline"
SAVE_MODEL = True  # Set to False to skip model serialization for faster runs
# ============================================================

labeled = pd.read_parquet("labeled_base.parquet")
hmm     = pd.read_parquet(f"hmm_features_{HMM_TAG}.parquet")
meta    = json.load(open("split_meta.json"))
global_train_end_date = pd.Timestamp(meta["global_train_end_date"])

# Join HMM columns onto the labeled subset by (date, instrument) -- causal, no recompute
meta_labeled_df = labeled.merge(hmm, on=["date", "instrument"], how="left")


# ======================================================================
# PHASE 4 PREP : target, leakage removal, split, one-hot, preprocessor
# ======================================================================

ml_df = meta_labeled_df.copy()
target_col = "meta_label"

LEAK_AND_ID = [
    "date",
    "meta_label",
    "tb_event_type", "tb_exit_date", "tb_exit_price", "tb_trade_return",
    "hmm_prob_sum",
]

RAW_PRICE = ["open", "high", "low", "close", "volume", "open_interest"]

ml_df = ml_df.replace([np.inf, -np.inf], np.nan)

META_FEATURES = [col for col in ml_df.columns if col not in LEAK_AND_ID and col not in RAW_PRICE]

print(f"TOTAL META_FEATURES          : {len(META_FEATURES)}")

n_total = len(ml_df)
mask_nan = ml_df[META_FEATURES].isna().any(axis=1)
print(f"Rows : {n_total} | with ≥1 NaN : {mask_nan.sum()})")

missing_counts = ml_df[META_FEATURES].isna().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

high_missing = missing_counts[missing_counts > 0.01 * n_total].index.tolist()

ml_df = ml_df[[c for c in ml_df.columns if c not in high_missing]]
META_FEATURES = [c for c in META_FEATURES if c not in high_missing]

n_total = len(ml_df)
mask_nan = ml_df[META_FEATURES].isna().any(axis=1)

missing_counts = ml_df[META_FEATURES].isna().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

train_df = ml_df[ml_df["date"] <= global_train_end_date]
test_df  = ml_df[ml_df["date"] >  global_train_end_date]

X_train = train_df[META_FEATURES]
y_train = train_df["meta_label"].values.astype(int)

X_test  = test_df[META_FEATURES]
y_test  = test_df["meta_label"].values.astype(int)

categorical_features = ["instrument"]

numerical_features = [
    col for col in META_FEATURES
    if col not in categorical_features
]

X_train = pd.get_dummies(X_train, columns=categorical_features, dtype=float)
X_test  = pd.get_dummies(X_test,  columns=categorical_features, dtype=float)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

META_FEATURES = list(X_train.columns)

numeric_scaled_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ]
)

numeric_tree_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

scaled_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_scaled_pipeline, numerical_features),
    ],
    remainder="passthrough"
)

tree_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_tree_pipeline, numerical_features),
    ],
    remainder="passthrough"
)

print("Scaled preprocessor ready.")
print("Tree preprocessor ready.")

# ======================================================================
# CPCV
# ======================================================================

class CombinatorialPurgedCV:
    """
    CPCV de Prado (Advances in Financial ML, ch. 12).
    """
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = max(1, int(n_samples * self.embargo_pct))

        indices = np.arange(n_samples)
        
        for test_start in range(0, n_samples - fold_size + 1, fold_size):
            test_end = test_start + fold_size
            test_idx = indices[test_start:test_end]

            train_start = test_end + embargo_size
            if train_start >= n_samples:
                train_start = 0

            if train_start < test_start:
                train_idx = np.concatenate([
                    indices[train_start:test_start - embargo_size],
                    indices[test_end + embargo_size:]
                ])
            else:
                train_idx = indices[test_end + embargo_size:]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

cpcv = CombinatorialPurgedCV(n_splits=5, embargo_pct=0.01)

# ======================================================================
# FEATURE CLUSTERING
# ======================================================================

semantic_rules = [
    ("HMM regimes", lambda f: f.startswith("hmm_")),
    
    ("Macro context", lambda f: any(k in f for k in [
        "ovx_", "vix_", "ted_", "hml_", "smb_", "wml_", "fed_", "term_",
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


feature_clusters = pd.Series(
    {f: assign_cluster(f) for f in META_FEATURES},
    name="cluster",
)

clusters = {}
for feat, cl in feature_clusters.items():
    clusters.setdefault(cl, []).append(feat)

for cl in sorted(clusters, key=lambda c: -len(clusters[c])):
    members = clusters[cl]

assigned = sum(len(v) for v in clusters.values())
assert assigned == len(META_FEATURES), f"Mismatch: {assigned} vs {len(META_FEATURES)}"

if "Other" in clusters:
    print(f"⚠ 'Other' cluster contents: {clusters['Other']}")
else:
    print("✓ No feature fell into 'Other'.")


# =====================================================================
# OPTUNA  --  LightGBM ROBUST objective only, warm-started (lightweight)
# =====================================================================
def objective_lgb_robust(trial):
    params = {
        "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
        "random_state": RANDOM_STATE, "n_jobs": 1, "class_weight": "balanced",
        "num_leaves":        trial.suggest_int("num_leaves", 7, 63),
        "max_depth":         trial.suggest_int("max_depth", 2, 6),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    aucs = []
    for tr, val in cpcv.split(X_train):
        pipe = Pipeline([("preprocessor", clone(scaled_preprocessor)),
                         ("model", lgb.LGBMClassifier(**params))])
        pipe.fit(X_train.iloc[tr], y_train[tr])
        p = pipe.predict_proba(X_train.iloc[val])[:, 1]
        aucs.append(roc_auc_score(y_train[val], p))
    aucs = np.array(aucs)
    return aucs.mean() - LAMBDA_RISK * aucs.std(ddof=1)

sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True,
                                     group=True, n_startup_trials=10, constant_liar=True)
study = optuna.create_study(direction="maximize", sampler=sampler,
                            pruner=optuna.pruners.NopPruner())

# Warm-start: enqueue the previous experiment's best params if available
_prev = f"best_lgb_params_{HMM_TAG}.json"
if os.path.exists(_prev):
    try:
        study.enqueue_trial(json.load(open(_prev)))
        print(f"Warm-started from {_prev}")
    except Exception as e:
        print(f"Warm-start skipped ({e})")

study.optimize(objective_lgb_robust, n_trials=N_TRIALS, show_progress_bar=False, n_jobs=1)
best_params_lgb = study.best_params
print(f"\nRobust CPCV AUC: {study.best_value:.4f}")
print("Best params:", best_params_lgb)
json.dump(best_params_lgb, open(f"best_lgb_params_{HMM_TAG}.json", "w"))

# ✅ NEW: Also save params to parquet for easier programmatic access
params_df = pd.DataFrame([best_params_lgb])
params_df.to_parquet(f"best_lgb_params_{HMM_TAG}.parquet", index=False)
print(f"✓ Saved best_lgb_params_{HMM_TAG}.parquet")

# Refit the official LightGBM on full train
lgb_full = dict(objective="binary", metric="binary_logloss", verbosity=-1,
                random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", **best_params_lgb)
best_lgb = lgb.LGBMClassifier(**lgb_full)
best_lgb.fit(X_train, y_train)

# ✅ NEW: Save the trained model
if SAVE_MODEL:
    # Option 1: Save with joblib (faster, more standard for sklearn-like objects)
    joblib.dump(best_lgb, f"lgb_model_{HMM_TAG}.joblib")
    print(f"✓ Saved lgb_model_{HMM_TAG}.joblib")
    
    # Option 2: Also save with pickle for compatibility
    with open(f"lgb_model_{HMM_TAG}.pkl", "wb") as f:
        pickle.dump(best_lgb, f)
    print(f"✓ Saved lgb_model_{HMM_TAG}.pkl")
    
    # Save the preprocessor pipeline as well (needed for predictions)
    joblib.dump(scaled_preprocessor, f"preprocessor_{HMM_TAG}.joblib")
    print(f"✓ Saved preprocessor_{HMM_TAG}.joblib")

# ✅ NEW: Save model metadata to parquet
metadata = pd.DataFrame({
    "n_features": [len(META_FEATURES)],
    "n_train_samples": [len(X_train)],
    "n_test_samples": [len(X_test)],
    "class_balance_train": [y_train.mean()],
    "class_balance_test": [y_test.mean()],
    "hmm_tag": [HMM_TAG],
    "cpcv_auc": [study.best_value],
    "n_estimators": [best_lgb.n_estimators],
    "random_state": [RANDOM_STATE],
})
metadata.to_parquet(f"model_metadata_{HMM_TAG}.parquet", index=False)
print(f"✓ Saved model_metadata_{HMM_TAG}.parquet")

# =====================================================================
# CLUSTER-LEVEL MDA  (permute each cluster jointly on frozen per-fold models)
# =====================================================================
N_REPEATS = 5
rng = np.random.default_rng(RANDOM_STATE)
cluster_names = list(clusters.keys())
drops = {c: [] for c in cluster_names}

for tr, val in cpcv.split(X_train):
    pipe = Pipeline([("preprocessor", clone(scaled_preprocessor)),
                     ("model", lgb.LGBMClassifier(**lgb_full))])
    pipe.fit(X_train.iloc[tr], y_train[tr])
    yv = y_train[val]; Xv = X_train.iloc[val]
    base_auc = roc_auc_score(yv, pipe.predict_proba(Xv)[:, 1])
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

imp_cluster = pd.DataFrame({
    "mda_mean": {c: np.mean(drops[c]) for c in cluster_names},
    "mda_std":  {c: np.std(drops[c], ddof=1) for c in cluster_names},
}).sort_values("mda_mean", ascending=False)
print("\n" + "="*60 + "\nCLUSTER-LEVEL MDA (drop in CPCV AUC)\n" + "="*60)
print(imp_cluster.round(4).to_string())

# ✅ NEW: Save to both CSV and parquet
imp_cluster.to_csv(f"importance_cluster_{HMM_TAG}.csv")
imp_cluster.to_parquet(f"importance_cluster_{HMM_TAG}.parquet")
print(f"✓ Saved importance_cluster_{HMM_TAG}.parquet")

# =====================================================================
# INDIVIDUAL MDA  (single-feature permutation; Breiman, frozen per-fold models)
# =====================================================================
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

# ✅ NEW: Save to both CSV and parquet
imp_indiv.to_csv(f"importance_individual_{HMM_TAG}.csv")
imp_indiv.to_parquet(f"importance_individual_{HMM_TAG}.parquet")
print(f"✓ Saved importance_individual_{HMM_TAG}.parquet")

# =====================================================================
# ✅ NEW: IDENTIFY HMM FEATURES IN TOP FEATURES (key insight!)
# =====================================================================
print("\n" + "="*60)
print("HMM FEATURES AMONG TOP PREDICTORS")
print("="*60)

hmm_features = [f for f in imp_indiv.index if f.startswith('hmm_')]
if hmm_features:
    hmm_importance = imp_indiv.loc[hmm_features].sort_values('mda', ascending=False)
    print(f"\nFound {len(hmm_features)} HMM features in top features:")
    print(hmm_importance.round(4).to_string())
    print(f"\nTop 5 HMM features for next run:")
    for i, (feat, imp) in enumerate(hmm_importance.head(5).iterrows(), 1):
        print(f"  {i}. {feat:30s} : {imp.values[0]:.6f}")
else:
    print("\nNo HMM features found in importance ranking.")

print(f"\nAll outputs saved with tag: {HMM_TAG}")
print(f"To reload model: best_lgb = joblib.load('lgb_model_{HMM_TAG}.joblib')")
