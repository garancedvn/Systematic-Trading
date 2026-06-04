"""
load_and_predict.py  --  UTILITY for reusing saved models

After running evaluate_model_with_model_save.py, you can:
1. Load the trained LightGBM model
2. Make predictions on new data
3. Extract feature importance without re-training
4. Reload parameters and metadata

Usage:
    python load_and_predict.py
"""
import joblib
import pickle
import pandas as pd
import json
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
HMM_TAG = "baseline"  # Must match the tag from your evaluate_model run
# ============================================================

def load_model(hmm_tag):
    """Load the trained LightGBM model"""
    model_path = f"lgb_model_{hmm_tag}.joblib"
    if Path(model_path).exists():
        model = joblib.load(model_path)
        print(f"✓ Loaded model from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

def load_preprocessor(hmm_tag):
    """Load the preprocessing pipeline"""
    preproc_path = f"preprocessor_{hmm_tag}.joblib"
    if Path(preproc_path).exists():
        preprocessor = joblib.load(preproc_path)
        print(f"✓ Loaded preprocessor from {preproc_path}")
        return preprocessor
    else:
        raise FileNotFoundError(f"Preprocessor not found: {preproc_path}")

def load_params(hmm_tag):
    """Load hyperparameters from parquet"""
    params_path = f"best_lgb_params_{hmm_tag}.parquet"
    if Path(params_path).exists():
        params_df = pd.read_parquet(params_path)
        params = params_df.iloc[0].to_dict()
        print(f"✓ Loaded parameters from {params_path}")
        return params
    else:
        raise FileNotFoundError(f"Parameters not found: {params_path}")

def load_metadata(hmm_tag):
    """Load model metadata"""
    meta_path = f"model_metadata_{hmm_tag}.parquet"
    if Path(meta_path).exists():
        metadata = pd.read_parquet(meta_path)
        print(f"✓ Loaded metadata from {meta_path}")
        return metadata
    else:
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

def load_feature_importance(hmm_tag, importance_type="individual"):
    """Load feature importance (individual or cluster)"""
    if importance_type not in ["individual", "cluster"]:
        raise ValueError("importance_type must be 'individual' or 'cluster'")
    
    imp_path = f"importance_{importance_type}_{hmm_tag}.parquet"
    if Path(imp_path).exists():
        importance = pd.read_parquet(imp_path)
        print(f"✓ Loaded {importance_type} importance from {imp_path}")
        return importance
    else:
        raise FileNotFoundError(f"Importance file not found: {imp_path}")

def get_hmm_feature_ranking(hmm_tag):
    """Get HMM features ranked by importance"""
    importance = load_feature_importance(hmm_tag, "individual")
    hmm_features = importance[importance.index.str.startswith('hmm_')].sort_values('mda', ascending=False)
    return hmm_features

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LOADING SAVED MODELS AND IMPORTANCE")
    print("=" * 70)
    
    # Load everything
    try:
        model = load_model(HMM_TAG)
        preprocessor = load_preprocessor(HMM_TAG)
        params = load_params(HMM_TAG)
        metadata = load_metadata(HMM_TAG)
        importance_individual = load_feature_importance(HMM_TAG, "individual")
        importance_cluster = load_feature_importance(HMM_TAG, "cluster")
        
        print("\n" + "=" * 70)
        print("MODEL METADATA")
        print("=" * 70)
        print(metadata.T.to_string())
        
        print("\n" + "=" * 70)
        print("BEST HYPERPARAMETERS")
        print("=" * 70)
        for key, value in params.items():
            print(f"  {key:25s}: {value}")
        
        print("\n" + "=" * 70)
        print("CLUSTER-LEVEL IMPORTANCE (Top 10)")
        print("=" * 70)
        print(importance_cluster.head(10).round(6).to_string())
        
        print("\n" + "=" * 70)
        print("TOP 15 INDIVIDUAL FEATURES")
        print("=" * 70)
        print(importance_individual.head(15).round(6).to_string())
        
        print("\n" + "=" * 70)
        print("HMM FEATURES RANKING")
        print("=" * 70)
        hmm_ranking = get_hmm_feature_ranking(HMM_TAG)
        if len(hmm_ranking) > 0:
            print(hmm_ranking.round(6).to_string())
            print(f"\n✓ {len(hmm_ranking)} HMM features found")
            print("\nRECOMMENDATION for next HMM experiments:")
            print("The following HMM features have the highest predictive power:")
            for i, (feat, mda_val) in enumerate(hmm_ranking.head(5).iterrows(), 1):
                print(f"  {i}. {feat:35s} (importance={mda_val.values[0]:.6f})")
        else:
            print("⚠ No HMM features found in importance ranking")
        
        print("\n" + "=" * 70)
        print("HOW TO USE THE SAVED MODEL")
        print("=" * 70)
        print(f"""
1. Load in Python:
   >>> import joblib
   >>> model = joblib.load('lgb_model_{HMM_TAG}.joblib')
   >>> preprocessor = joblib.load('preprocessor_{HMM_TAG}.joblib')

2. Make predictions on new data (X_new):
   >>> X_preprocessed = preprocessor.transform(X_new)
   >>> predictions = model.predict(X_preprocessed)
   >>> probabilities = model.predict_proba(X_preprocessed)

3. Get feature importance:
   >>> importance_df = pd.read_parquet('importance_individual_{HMM_TAG}.parquet')
   >>> importance_df.head(20)

4. Reload hyperparameters:
   >>> params = pd.read_parquet('best_lgb_params_{HMM_TAG}.parquet').iloc[0].to_dict()
        """)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have run evaluate_model_with_model_save.py first!")
