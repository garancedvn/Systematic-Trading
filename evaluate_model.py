"""
evaluate_model.py  --  RE-RUN PER EXPERIMENT.

Joins labeled_base.parquet with the current hmm_features.parquet, re-optimizes the
LightGBM (ROBUST objective only, warm-started from the previous best) via Optuna,
refits the best model, and computes cluster-level + individual MDA importance.

Outputs (per HMM_TAG):
  * best_lgb_params_<tag>.json   : winning hyperparameters (also reused as warm-start)
  * importance_cluster_<tag>.csv : cluster-level MDA
  * importance_individual_<tag>.csv : per-feature MDA (top features)
"""
import os, json, warnings
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
N_TRIALS     = 25          # lightweight per-experiment budget

# ============================================================
# CONFIG  --  must match the HMM_TAG used in run_hmm.py
# ============================================================
HMM_TAG = "baseline"
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

# --- [notebook cell 161] ------------------------------------------------
# ============================================================
# Phase 4.1 — Define Target and Remove Leakage Columns
# ============================================================

ml_df = meta_labeled_df.copy()
target_col = "meta_label"

LEAK_AND_ID = [
    "date",          # identité
    "meta_label",                   # signal primaire et cible qu on remettra ensuite dans y_train
    "tb_event_type", "tb_exit_date", "tb_exit_price", "tb_trade_return",  # FUTUR du trade → fuite
    "hmm_prob_sum",# colonnes de vérification, redondantes
]

# --- 2. Prix bruts non stationnaires : on garde leurs versions dérivées,
#        pas les niveaux absolus (close=140 en 2008 n'a pas de sens prédictif futur).
RAW_PRICE = ["open", "high", "low", "close", "volume", "open_interest"]



# Replace infinite feature values by NaN directly in ml_df.
# The imputers inside the preprocessing pipelines will handle them later.
ml_df = ml_df.replace([np.inf, -np.inf], np.nan)


# --- 5. Le pool final exhaustif ---
META_FEATURES = [col for col in ml_df.columns if col not in LEAK_AND_ID and col not in RAW_PRICE]


print(f"TOTAL META_FEATURES          : {len(META_FEATURES)}")
print("\nListe complète :")
for f in META_FEATURES:
    print("  ", f)

# --- [notebook cell 163] ------------------------------------------------
# ============================================================
# Phase 4.2 — Missing Values Check
# ============================================================

n_total = len(ml_df)
mask_nan = ml_df[META_FEATURES].isna().any(axis=1)
print(f"Rows : {n_total} | with ≥1 NaN : {mask_nan.sum()} ({100*mask_nan.sum()/n_total:.1f}%)")

missing_counts = ml_df[META_FEATURES].isna().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

print("\nFeatures with NaN :")
print(missing_counts.to_string() if len(missing_counts) else "  No one ✓")
print("\nPercentage missing per feature:")
print((100*missing_counts/n_total).to_string() if len(missing_counts) else "  0% per feature ✓")

# --- [notebook cell 165] ------------------------------------------------
high_missing = missing_counts[missing_counts > 0.01 * n_total].index.tolist()

ml_df = ml_df[[c for c in ml_df.columns if c not in high_missing]]
META_FEATURES = [c for c in META_FEATURES if c not in high_missing]

print("Features with more than 1% missing values:")
print("\n\nDeleted feautres:")
print(high_missing)

# --- [notebook cell 166] ------------------------------------------------
# ============================================================
# Second Missing Values Check
# ============================================================

n_total = len(ml_df)
mask_nan = ml_df[META_FEATURES].isna().any(axis=1)
print(f"Rows : {n_total} | with ≥1 NaN : {mask_nan.sum()} ({100*mask_nan.sum()/n_total:.1f}%)")

missing_counts = ml_df[META_FEATURES].isna().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

print("\nFeatures with NaN :")
print(missing_counts.to_string() if len(missing_counts) else "  No one ✓")
print("\nPercentage missing per feature:")
print((100*missing_counts/n_total).to_string() if len(missing_counts) else "  0% per feature ✓")

# --- [notebook cell 169] ------------------------------------------------
# ============================================================
#  Create train and test dataframes
# ============================================================

train_df = ml_df[ml_df["date"] <= global_train_end_date]
test_df  = ml_df[ml_df["date"] >  global_train_end_date]

X_train = train_df[META_FEATURES]
y_train = train_df["meta_label"].values.astype(int)

X_test  = test_df[META_FEATURES]
y_test  = test_df["meta_label"].values.astype(int)


print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape} ")
print(f"Nombre de features : {len(META_FEATURES)}")

print(f"y_train : {y_train.shape}")
print(f"y_test : {y_test.shape}")

print(f"Distribution y_train : {np.bincount(y_train)}")
print(f"Distribution y_test  : {np.bincount(y_test)}")

# --- [notebook cell 170] ------------------------------------------------
# ============================================================
#  Identify Numerical and Categorical Features
# ============================================================

categorical_features = [
    "instrument"
]

# We keep primary_signal as numerical because it is directional: -1 or +1
numerical_features = [
    col for col in META_FEATURES
    if col not in categorical_features
]

print("Number of categorical features:", len(categorical_features))
print(categorical_features)

print("\nNumber of numerical features:", len(numerical_features))
print(numerical_features)

print("\nCheck dtypes:")
print(X_train[META_FEATURES].dtypes)

# --- [notebook cell 171] ------------------------------------------------
# One-hot en amont, une bonne fois pour toutes
X_train = pd.get_dummies(X_train, columns=categorical_features, dtype=float)
X_test  = pd.get_dummies(X_test,  columns=categorical_features, dtype=float)

# Aligne le test sur les colonnes du train (au cas où un instrument manque d'un côté)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

META_FEATURES = list(X_train.columns)

# --- [notebook cell 172] ------------------------------------------------
# ============================================================
# Preprocessing Pipelines
# ============================================================


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

# --- [notebook cell 176] ------------------------------------------------
class CombinatorialPurgedCV:
    """
    CPCV de Prado (Advances in Financial ML, ch. 12).
    Découpe la série en N groupes contigus, et pour chaque combinaison de k
    groupes mis en test, le reste forme le train. Applique un embargo
    (purge) autour des frontières pour éviter le leakage label↔feature.

    Paramètres
    ----------
    n_splits : int — nombre de groupes (N)
    n_test_groups : int — k groupes en test par combinaison
    embargo_pct : float — fraction des données embargo de chaque côté du test
    """
    def __init__(self, n_splits=6, n_test_groups=2, embargo_pct=0.01):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n = len(X)
        embargo = int(n * self.embargo_pct)
        indices = np.arange(n)
        # Découpage en N groupes contigus
        group_bounds = np.array_split(indices, self.n_splits)

        for test_group_ids in combinations(range(self.n_splits), self.n_test_groups):
            test_idx = np.concatenate([group_bounds[g] for g in test_group_ids])
            test_set = set(test_idx)

            # Embargo : retirer les voisins immédiats des blocs test
            embargoed = set()
            for g in test_group_ids:
                lo = max(group_bounds[g][0] - embargo, 0)
                hi = min(group_bounds[g][-1] + embargo + 1, n)
                embargoed.update(range(lo, hi))

            train_idx = np.array(sorted(set(indices) - embargoed - test_set))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        from math import comb
        return comb(self.n_splits, self.n_test_groups)


# ============================================================
# CPCV configuration
# ============================================================
n_splits, k = 5, 2
n = len(X_train)
block_size = n // n_splits
embargo_obs = int(n * 0.01)

print(f"Train size       : {n}")
print(f"Block size       : {block_size} obs (~{block_size/252:.1f} ans daily)")
print(f"Test per fold    : {block_size * k} obs ({k} blocs)")
print(f"Train per fold   : {block_size * (n_splits - k)} obs ({n_splits-k} blocs - embargo)")
print(f"Embargo          : {embargo_obs} obs")
print(f"Nb folds         : {n_splits * (n_splits-1) // (k * 1) // 2 if k==2 else 'C(N,k)'}")


# --- [notebook cell 177] ------------------------------------------------
cpcv = CombinatorialPurgedCV(n_splits=n_splits, n_test_groups=k, embargo_pct=0.01)
print(f"CPCV : {cpcv.get_n_splits()} folds générés (N={n_splits}, k={k})\n")

# ======================================================================
# SEMANTIC CLUSTERS
# ======================================================================

# --- [notebook cell 205] ------------------------------------------------
# ==============================================================================
# FEATURE CLUSTERING (hierarchical, correlation distance)
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------------------------
# 1. Correlation matrix on the TRAINING features only (no leakage)
# ------------------------------------------------------------------------------
X_train_df = pd.DataFrame(X_train, columns=META_FEATURES)
corr = X_train_df.corr(method="spearman")   # Spearman: robust to non-linearity & outliers

# ------------------------------------------------------------------------------
# 2. Correlation -> distance matrix: d = sqrt(0.5 * (1 - corr))
#    Perfectly correlated -> 0, uncorrelated -> ~0.707, anti-correlated -> 1
# ------------------------------------------------------------------------------
dist_matrix = np.sqrt(0.5 * (1.0 - corr.values))
np.fill_diagonal(dist_matrix, 0.0)                 # numerical safety
condensed = squareform(dist_matrix, checks=False)  # condensed form for linkage

# ------------------------------------------------------------------------------
# 3. Hierarchical clustering with average linkage : measure of distance between clusters
# ------------------------------------------------------------------------------
Z = linkage(condensed, method="average")

# ------------------------------------------------------------------------------
# 4. Optimal number of clusters via silhouette score (ONC spirit)
#    We scan candidate cluster counts and keep the one maximizing silhouette.
# ------------------------------------------------------------------------------
candidate_k = range(3, 16)
sil_scores = []
for k in candidate_k:
    labels_k = fcluster(Z, t=k, criterion="maxclust")
    # silhouette needs the precomputed distance matrix
    score = silhouette_score(dist_matrix, labels_k, metric="precomputed")
    sil_scores.append(score)

best_k = list(candidate_k)[int(np.argmax(sil_scores))]
onc_k = best_k
print(f"Optimal number of clusters (max silhouette): {best_k}")

# Final cluster assignment
cluster_labels = fcluster(Z, t=best_k, criterion="maxclust")
feature_clusters = pd.Series(cluster_labels, index=META_FEATURES, name="cluster")

# ------------------------------------------------------------------------------
# 5. Build the cluster -> features mapping (used in Phase 2 for group permutation)
# ------------------------------------------------------------------------------
clusters = {}
for feat, cl in feature_clusters.items():
    clusters.setdefault(int(cl), []).append(feat)

print(f"\n{len(clusters)} clusters formed:\n")
for cl in sorted(clusters):
    members = clusters[cl]
    print(f"  Cluster {cl} ({len(members)} features): {', '.join(members[:6])}"
          + (f", ... (+{len(members)-6})" if len(members) > 6 else ""))

# --- [notebook cell 209] ------------------------------------------------
# ==============================================================================
# SEMANTIC FEATURE CLUSTERS — adapted to current Phase 2 feature set
# ==============================================================================
# Rules are ORDER-SENSITIVE (first match wins) to resolve ambiguous names:
#   - vol_oi_ratio              -> Volume/OI    (not Volatility, despite "vol_")
#   - relative_vol_*             -> Cross-asset (relative, not raw volatility)
#   - sector_vol_dispersion     -> Cross-asset
#   - momentum_rank_*            -> Cross-asset (ranked, not raw momentum)
#   - bb_width / bb_position    -> Volatility  (volatility envelope)
#   - obv / mfi / vol_oi        -> Volume/OI   (volume-based indicators)
# ==============================================================================

import pandas as pd

semantic_rules = [
    # 1. Regimes
    ("Regimes (HMM)", lambda f: f.startswith("hmm_")),

    # 2. Signal interaction — kept separate (4 features) to test signal-engineering value
    ("Signal interaction", lambda f: any(k in f for k in [
        "signal_changed", "signal_persistence", "signal_trend_concord", "signal_density",
    ])),

    # 3. Seasonality — energy-specific calendar features
    ("Seasonality", lambda f: any(k in f for k in [
        "heating_season", "driving_season", "hurricane_season",
        "quarter_progress", "day_of_year_sin", "day_of_year_cos",
    ])),

    # 4. Macro environment — VIX, DXY, US rates, broad multi-asset
    ("Macro environment", lambda f: any(k in f for k in [
        "vix_", "dxy_", "us10y", "multiasset_vol", "equity_avg", "equity_vol",
        "equity_ret_daily", "risk_on_score", "hg_ret_daily",
    ])),

    # 5. Energy-specific spreads (3:2:1 crack, oil-gas, energy-vs-market)
    ("Energy spreads", lambda f: any(k in f for k in [
        "crack_321", "oil_gas_ratio", "relative_energy_vol",
    ])),

    # 6. Metals & cross-commodity ratios
    ("Metals/Commodities", lambda f: any(k in f for k in [
        "copper", "gold_silver", "copper_gold",
    ])),

    # 7. Cross-asset correlation / lead-lag
    ("Cross-asset correlation", lambda f: any(k in f for k in [
        "corr_basket", "leadlag_anchor", "beta_basket",
        "asset_copper_corr", "asset_gold_corr", "asset_equity_corr",
        "cross_asset_momentum_concordance",
    ])),

    # 8. Relative strength within the energy basket
    ("Relative strength", lambda f: any(k in f for k in [
        "relative_vol", "relative_momentum", "momentum_rank",
        "sector_momentum", "sector_vol",
    ])),

    # 9. Volume & Liquidity — merged microstructure into here for min size
    ("Volume & Liquidity", lambda f: any(k in f for k in [
        "obv", "mfi", "volume_", "oi_change", "oi_momentum", "vol_oi",
        "dollar_volume", "log_volume", "log_oi",
        "amihud", "kyle_lambda",
        "roll_spread", "bid_ask", "oc_spread", "hl_spread",
    ])),

    # 10. Volatility (raw estimators)
    ("Volatility", lambda f: any(k in f for k in [
        "vol_5", "vol_10", "vol_20", "vol_60",
        "realized_vol", "vol_parkinson", "vol_garman", "vol_rogers",
        "yang_zhang", "atr_", "atx_", "bb_width", "bb_position",
        "vol_ratio_20_60", "ewma_vol", "downside_vol", "garman_klass", "vol_change",
    ])),

    # 11. Momentum / Trend / Oscillators
    ("Momentum/Trend", lambda f: any(k in f for k in [
        "momentum_", "macd", "rsi_", "stoch", "willr", "adx",
        "close_to_sma", "distance_from_200d", "trend_",
    ])),

    # 12. Returns / Price action
    ("Returns/Price", lambda f: any(k in f for k in [
        "returns", "log_return", "mean_return", "abs_return",
        "close_position", "price_zscore", "close_fracdiff",
        "positive_return", "ret_", "price_change",
        "price_range_position", "range_position",
    ])),

    # 13. Distribution / Complexity / Memory (incl. autocorr)
    ("Distribution/Complexity", lambda f: any(k in f for k in [
        "skew", "kurt", "autocorr", "shannon_entropy", "lz_complexity", "sadf",
        "hurst", "dfa", "spectral_entropy", "approx_entropy", "dominant_cycle",
        "return_vol_correl",
    ])),

    # 14. Signal & Instrument controls — merged the two single-feature clusters
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

# Report, largest cluster first
print(f"{len(clusters)} semantic clusters across {len(META_FEATURES)} features:\n")
for cl in sorted(clusters, key=lambda c: -len(clusters[c])):
    members = clusters[cl]
    preview = ", ".join(members[:6])
    extra = f", ... (+{len(members) - 6})" if len(members) > 6 else ""
    print(f"  {cl:26s} ({len(members):>2}): {preview}{extra}")

# Safety checks
assigned = sum(len(v) for v in clusters.values())
assert assigned == len(META_FEATURES), f"Mismatch: {assigned} vs {len(META_FEATURES)}"
print(f"\n✓ All {len(META_FEATURES)} features assigned.")

if "Other" in clusters:
    print(f"⚠ 'Other' cluster contents: {clusters['Other']}")
    print("  → If unexpected, add rules above to capture these features.")
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
    return aucs.mean() - LAMBDA_RISK * aucs.std(ddof=1)   # ROBUST objective

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

# Refit the official LightGBM on full train
lgb_full = dict(objective="binary", metric="binary_logloss", verbosity=-1,
                random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", **best_params_lgb)
best_lgb = lgb.LGBMClassifier(**lgb_full)
best_lgb.fit(X_train, y_train)

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
            Xp = Xv.copy(); Xp[cols] = Xp[cols].values[perm]   # joint shuffle
            d.append(base_auc - roc_auc_score(yv, pipe.predict_proba(Xp)[:, 1]))
        drops[cl].append(np.mean(d))

imp_cluster = pd.DataFrame({
    "mda_mean": {c: np.mean(drops[c]) for c in cluster_names},
    "mda_std":  {c: np.std(drops[c], ddof=1) for c in cluster_names},
}).sort_values("mda_mean", ascending=False)
print("\n" + "="*60 + "\nCLUSTER-LEVEL MDA (drop in CPCV AUC)\n" + "="*60)
print(imp_cluster.round(4).to_string())
imp_cluster.to_csv(f"importance_cluster_{HMM_TAG}.csv")

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
imp_indiv.to_csv(f"importance_individual_{HMM_TAG}.csv")
print(f"\nSaved best_lgb_params_{HMM_TAG}.json, importance_cluster/individual_{HMM_TAG}.csv")
