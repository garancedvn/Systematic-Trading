import pandas as pd
import numpy as np
import itertools
from feature_engineering import yang_zhang_vol

from load_data import load_panel 

def generate_meta_labels(df, signals, p_mult=2.0, sl_mult=1.0, h=10):
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["vol"] = yang_zhang_vol(df["open"], df["high"], df["low"], df["close"], window=20)

    signals_aligned = signals.reindex(df.index).fillna(0).astype(int)
    active_bets = df[signals_aligned != 0].copy()
    rows = []

    for idx in active_bets.index:
        signal = signals.loc[idx]
        entry_price = active_bets.loc[idx, "close"]
        vol = active_bets.loc[idx, "vol"] # DEPRADO mean of the fourteen produce days avg vol

        if pd.isna(vol):
            continue

        upper_barrier = entry_price * (1+p_mult * vol)
        lower_barrier = entry_price * (1-sl_mult*vol)
        
        start_pos = df.index.get_loc(idx) 
        end_pos = min(start_pos + h, len(df) - 1)

        future_path = df.iloc[start_pos+1:end_pos+1][["high", "low"]]
        hit_label = 0
        hit_type = "time_out"
        t1 = df.index[end_pos]

        for future_indx, row in future_path.iterrows():
            high_t = row["high"]
            low_t = row["low"]

            hit_upper = high_t >= upper_barrier
            hit_lower = low_t <= lower_barrier

            if signal == 1:  # Long signal
                if hit_upper and hit_lower:
                    hit_label = 0
                    hit_type = "stop_loss"
                    t1 = future_indx
                    break
                elif hit_upper:
                    hit_label = 1 
                    hit_type = "upper_target"
                    t1 = future_indx
                    break
                elif hit_lower:
                    hit_label = 0
                    hit_type = "stop_loss"
                    t1 = future_indx
            elif signal == -1:  # Short signal
                if hit_upper and hit_lower:
                    hit_label = 0
                    hit_type = "stop_loss"
                    t1 = future_indx
                    break
                elif hit_lower:
                    hit_label = 1
                    hit_type = "lower_target"
                    t1 = future_indx
                    break
                elif hit_upper:
                    hit_label = 0
                    hit_type = "stop_loss"
                    t1 = future_indx
                    break

        rows.append({
            "t0": idx, 
            "t1": t1, 
            "meta_label": hit_label, 
            "hit_type": hit_type, 
            "signal": int(signal), 
            "entry_price": entry_price, 
            "vol_at_entry": vol, 
        })
    
    out = pd.DataFrame(rows).set_index("t0").sort_index()
    return out 


def calculate_trade_returns(df, signals, p_mult, sl_mult, h):
    df = df.copy()
    df["vol"] = yang_zhang_vol(df["open"], df["high"], df["low"], df["close"], window=50)

    df_signal = df.loc[signals.index].copy()
    active_bets = df_signal[signals !=0].copy()
    trade_returns = pd.Series(index=active_bets.index, dtype=float)

    for idx in active_bets.index:
        signal = active_bets.loc[idx, "primary_signal"]
        entry_price = active_bets.loc[idx, "close"]
        vol = active_bets.loc[idx, "vol"]

        if pd.isna(vol):
            continue

        upper_barrier = entry_price * np.exp(+p_mult * vol)
        lower_barrier = entry_price * np.exp(-sl_mult * vol)

        start_pos = df.index.get_loc(idx)
        end_pos = min(start_pos + h, len(df) - 1)

        future_path = df.iloc[start_pos+1:end_pos+1]["close"]
        
        # Default return if the time limit is reached without hitting a barrier
        final_price = df.iloc[end_pos]["close"]
        if signal == 1:
            actual_return = (final_price - entry_price) / entry_price
        else:
            actual_return = (entry_price - final_price) / entry_price

        # Check for barrier hits
        for future_price in future_path:
            if signal == 1:  
                if future_price >= upper_barrier:
                    actual_return = (upper_barrier - entry_price) / entry_price
                    break
                elif future_price <= lower_barrier:
                    actual_return = (lower_barrier - entry_price) / entry_price
                    break
            elif signal == -1:  
                if future_price <= lower_barrier:
                    actual_return = (entry_price - lower_barrier) / entry_price 
                    break
                elif future_price >= upper_barrier:
                    actual_return = (entry_price - upper_barrier) / entry_price 
                    break

        trade_returns.loc[idx] = actual_return

    return trade_returns.dropna()

def optimize_barriers(df, signals, p_mult_range, sl_mult_range, h_range):
    """
    Search the (p_mult, sl_mult, h) grid and report labelling diagnostics
    for each combination. Pick parameters that give:
        - class balance near 50/50
        - low time-out resolution
        - sensible holding periods
    Sharpe of the strategy is NOT the right criterion for labelling.
    """
    results = []
    param_combinations = list(itertools.product(p_mult_range, sl_mult_range, h_range))
    print(f"Testing {len(param_combinations)} parameter combinations...")

    for p_mult, sl_mult, h in param_combinations:
        labels = generate_meta_labels(df, signals,
                                       p_mult=p_mult, sl_mult=sl_mult, h=h)
        if len(labels) == 0:
            continue

        # Class balance
        pos_rate = labels["meta_label"].mean()
        balance_score = 1 - abs(0.5 - pos_rate) * 2   # 1.0 at 50/50, 0.0 at 0/100

        # Barrier resolution
        resolution = labels["hit_type"].value_counts(normalize=True)
        pct_timeout = resolution.get("time_out", 0.0)
        timeout_score = 1 - min(pct_timeout / 0.3, 1.0)  # penalty above 30% time-out

        # Holding period — want median around h/2 (efficient barrier use)
        holding = (labels["t1"] - labels.index).dt.days
        median_holding = holding.median()
        holding_score = 1 - abs(median_holding - h / 2) / h

        # Composite score (equal weights for now)
        composite = (balance_score + timeout_score + holding_score) / 3

        results.append({
            "p_mult": p_mult,
            "sl_mult": sl_mult,
            "h": h,
            "n_events": len(labels),
            "pct_positive": round(pos_rate, 3),
            "pct_timeout": round(pct_timeout, 3),
            "median_holding": round(median_holding, 1),
            "balance_score": round(balance_score, 3),
            "timeout_score": round(timeout_score, 3),
            "holding_score": round(holding_score, 3),
            "composite": round(composite, 3),
        })

    results_df = pd.DataFrame(results)
    best = results_df.sort_values("composite", ascending=False).iloc[0]
    return results_df, best

if __name__ == "__main__":
    panel, primary_signals = load_panel()
    
    labels = generate_meta_labels(
        panel["cl1s"], primary_signals["cl1s"],
        p_mult=2.0, sl_mult=1.0, h=10
    )

    ENERGY = ["cl1s", "ho1s", "rb1s", "ng1s"]
    LABEL_PARAMS = {"p_mult": 2.0, "sl_mult": 1.0, "h": 10}

    print("\n" + "=" * 70)
    print(f"LABELS FOR ALL INSTRUMENTS  (pt_sl={LABEL_PARAMS['p_mult']}/{LABEL_PARAMS['sl_mult']}, h={LABEL_PARAMS['h']})")
    print("=" * 70)

    all_labels = {}
    for tk in ENERGY:
        labels = generate_meta_labels(panel[tk], primary_signals[tk], **LABEL_PARAMS)
        all_labels[tk] = labels
        holding = (labels["t1"] - labels.index).dt.days
        hit_types = labels["hit_type"].value_counts(normalize=True)
        print(f"\n{tk}:")
        print(f"  Events:           {len(labels)}")
        print(f"  Positive labels:  {labels['meta_label'].mean():.1%}")
        print(f"  Mean holding:     {holding.mean():.1f} days")
        print(f"  Median holding:   {holding.median():.1f} days")
        print(f"  Hit types:        {hit_types.to_dict()}")

    # Also save the labels to disk so Part 3 doesn't recompute
    import pickle
    with open("meta_labels.pkl", "wb") as f:
        pickle.dump(all_labels, f)
    print(f"\nSaved all_labels to meta_labels.pkl")
