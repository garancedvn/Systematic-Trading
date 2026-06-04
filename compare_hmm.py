"""
compare_hmm.py  --  Compare HMM feature importance across experiments.

After running evaluate_hmm_features.py for several HMM_TAGs, this loads every
hmm_importance_<tag>.csv it finds and ranks the configurations by total HMM MDA.

    python compare_hmm.py
"""
import glob, os
import pandas as pd

files = sorted(glob.glob("hmm_importance_*.csv"))
if not files:
    raise SystemExit("No hmm_importance_*.csv found. Run evaluate_hmm_features.py first.")

rows = []
per_tag = {}
for path in files:
    tag = os.path.basename(path)[len("hmm_importance_"):-len(".csv")]
    df = pd.read_csv(path, index_col=0)
    per_tag[tag] = df
    rows.append({
        "tag":            tag,
        "n_hmm_features": len(df),
        "total_hmm_mda":  df["mda"].sum(),
        "mean_hmm_mda":   df["mda"].mean(),
        "top_hmm_feature": df.index[0] if len(df) else "—",
        "top_hmm_mda":    df["mda"].iloc[0] if len(df) else 0.0,
    })

summary = pd.DataFrame(rows).sort_values("total_hmm_mda", ascending=False)
print("="*78)
print("HMM CONFIGURATION COMPARISON (ranked by total HMM MDA)")
print("="*78)
print(summary.round(5).to_string(index=False))

best = summary.iloc[0]["tag"]
print(f"\n>>> Best HMM configuration: '{best}'")
print(f"\nTop HMM features in '{best}':")
print(per_tag[best].sort_values("mda", ascending=False).head(8).round(5).to_string())
