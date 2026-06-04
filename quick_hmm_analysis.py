"""
quick_hmm_analysis.py  --  ONE-LINE HMM FEATURE ANALYSIS

Use this to instantly check HMM feature importance WITHOUT re-training anything.

After you've run evaluate_model_with_model_save.py once, you can use this script
to instantly load importance rankings and compare HMM features.

Usage:
    python quick_hmm_analysis.py                      # Default: baseline
    python quick_hmm_analysis.py --tag v1 --top 15   # Custom tag & top N
    python quick_hmm_analysis.py --compare            # Compare all saved experiments
"""
import pandas as pd
import argparse
from pathlib import Path

def get_hmm_ranking(hmm_tag, top_n=10):
    """Load HMM feature importance ranking"""
    imp_path = f"importance_individual_{hmm_tag}.parquet"
    
    if not Path(imp_path).exists():
        print(f"❌ File not found: {imp_path}")
        print("   Make sure you've run: python evaluate_model_with_model_save.py")
        return None
    
    importance = pd.read_parquet(imp_path)
    hmm_features = importance[importance.index.str.startswith('hmm_')].sort_values('mda', ascending=False)
    
    return hmm_features.head(top_n)

def list_experiments():
    """List all available saved experiments"""
    importance_files = list(Path(".").glob("importance_individual_*.parquet"))
    experiments = [f.stem.replace("importance_individual_", "") for f in importance_files]
    return sorted(experiments)

def compare_experiments(top_n=10):
    """Compare HMM features across all saved experiments"""
    experiments = list_experiments()
    
    if not experiments:
        print("❌ No saved experiments found.")
        print("   Run: python evaluate_model_with_model_save.py")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPARING {len(experiments)} EXPERIMENTS (HMM Features)")
    print(f"{'='*80}\n")
    
    comparison_data = {}
    
    for tag in experiments:
        hmm_ranking = get_hmm_ranking(tag, top_n=999)
        if hmm_ranking is not None:
            comparison_data[tag] = hmm_ranking
            total_hmm_imp = hmm_ranking['mda'].sum()
            top_feature = hmm_ranking.index[0] if len(hmm_ranking) > 0 else "N/A"
            top_value = hmm_ranking.iloc[0, 0] if len(hmm_ranking) > 0 else 0
            
            print(f"Experiment: {tag}")
            print(f"  • Total HMM importance:    {total_hmm_imp:.6f}")
            print(f"  • Number of HMM features:  {len(hmm_ranking)}")
            print(f"  • Top HMM feature:         {top_feature} ({top_value:.6f})")
            
            # Show top 5
            print(f"  • Top 5 HMM features:")
            for i, (feat, mda_val) in enumerate(hmm_ranking.head(5).iterrows(), 1):
                print(f"      {i}. {feat:35s} : {mda_val.values[0]:.6f}")
            print()
    
    # Recommendation
    if len(comparison_data) > 1:
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}\n")
        
        best_exp = max(comparison_data.items(), 
                      key=lambda x: x[1]['mda'].sum())
        print(f"Best HMM configuration: {best_exp[0]}")
        print(f"Total HMM importance: {best_exp[1]['mda'].sum():.6f}")
        print(f"\nTop 3 HMM features from best configuration:")
        for i, (feat, mda_val) in enumerate(best_exp[1].head(3).iterrows(), 1):
            print(f"  {i}. {feat}: {mda_val.values[0]:.6f}")

def print_model_info(hmm_tag):
    """Print comprehensive model information"""
    meta_path = f"model_metadata_{hmm_tag}.parquet"
    params_path = f"best_lgb_params_{hmm_tag}.parquet"
    
    print(f"\n{'='*80}")
    print(f"MODEL: {hmm_tag}")
    print(f"{'='*80}\n")
    
    # Metadata
    if Path(meta_path).exists():
        meta = pd.read_parquet(meta_path)
        print("Training Configuration:")
        for col in meta.columns:
            val = meta[col].iloc[0]
            if isinstance(val, float):
                print(f"  {col:30s}: {val:.6f}")
            else:
                print(f"  {col:30s}: {val}")
    
    # Parameters
    if Path(params_path).exists():
        params = pd.read_parquet(params_path)
        print("\nOptimized Hyperparameters:")
        for col in params.columns:
            val = params[col].iloc[0]
            if isinstance(val, float):
                print(f"  {col:30s}: {val:.6f}")
            else:
                print(f"  {col:30s}: {val}")
    
    # Feature Importance
    print("\nTop 10 Features (All):")
    imp = pd.read_parquet(f"importance_individual_{hmm_tag}.parquet")
    for i, (feat, mda_val) in enumerate(imp.head(10).iterrows(), 1):
        print(f"  {i:2d}. {feat:40s} : {mda_val.values[0]:.6f}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick HMM feature importance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_hmm_analysis.py                    # Show baseline HMM features
  python quick_hmm_analysis.py --tag v1           # Show v1 experiment
  python quick_hmm_analysis.py --compare          # Compare all experiments
  python quick_hmm_analysis.py --tag baseline -n 20  # Show top 20
  python quick_hmm_analysis.py --info baseline    # Show full model info
        """
    )
    
    parser.add_argument('--tag', default='baseline', 
                       help='Experiment tag (default: baseline)')
    parser.add_argument('-n', '--top', type=int, default=10,
                       help='Number of top features to show (default: 10)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all available experiments')
    parser.add_argument('--info', type=str,
                       help='Show full model info for tag')
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        experiments = list_experiments()
        print(f"\nAvailable experiments ({len(experiments)}):")
        for exp in experiments:
            print(f"  • {exp}")
        if not experiments:
            print("  (none yet - run evaluate_model_with_model_save.py first)")
    
    elif args.info:
        print_model_info(args.info)
    
    elif args.compare:
        compare_experiments(args.top)
    
    else:
        # Single experiment view
        print(f"\n{'='*80}")
        print(f"HMM FEATURE IMPORTANCE — Experiment: {args.tag}")
        print(f"{'='*80}\n")
        
        ranking = get_hmm_ranking(args.tag, top_n=args.top)
        
        if ranking is not None:
            if len(ranking) == 0:
                print(f"⚠️  No HMM features found in importance ranking for {args.tag}")
            else:
                print(f"Top {min(args.top, len(ranking))} HMM Features:\n")
                for i, (feat, mda_val) in enumerate(ranking.iterrows(), 1):
                    pct = (mda_val.values[0] / ranking['mda'].sum()) * 100
                    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                    print(f"{i:2d}. {feat:35s} {mda_val.values[0]:8.6f}  [{bar}] {pct:5.1f}%")
                
                print(f"\n📊 Summary:")
                print(f"  Total HMM importance (sum): {ranking['mda'].sum():.6f}")
                print(f"  Average HMM importance:     {ranking['mda'].mean():.6f}")
                print(f"  Number of HMM features:     {len(ranking)}")
        
        # Also show cluster importance
        cluster_path = f"importance_cluster_{args.tag}.parquet"
        if Path(cluster_path).exists():
            clusters = pd.read_parquet(cluster_path)
            hmm_cluster = clusters[clusters.index == 'HMM regimes'] if 'HMM regimes' in clusters.index else None
            
            if hmm_cluster is not None and not hmm_cluster.empty:
                print(f"\n📌 HMM Cluster-Level Importance:")
                print(f"  Cluster MDA:  {hmm_cluster.loc['HMM regimes', 'mda_mean']:.6f} ± {hmm_cluster.loc['HMM regimes', 'mda_std']:.6f}")

    print()
