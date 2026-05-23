import pandas as pd
import numpy as np
import itertools

from load_data import load_panel 

def generate_meta_labels(df, signals, p_mult=2.0, sl_mult=1.0, h=10):
    df["returns"] = df["close"].pct_change()
    df["vol"] = df["returns"].rolling(20).std()

    active_bets = df[signals != 0].copy()
    labels = pd.Series(index=active_bets.index, dtype=float)

    for idx in active_bets.index:
        signal = signals.loc[idx]
        entry_price = active_bets.loc[idx, "close"]
        vol = active_bets.loc[idx, "vol"] # DEPRADO mean of the fourteen produce days avg vol

        if pd.isna(vol):
            continue

        upper_barrier = entry_price * (1 + p_mult * vol)
        lower_barrier = entry_price * (1 - sl_mult * vol)

        start_pos = df.index.get_loc(idx) 
        end_pos = min(start_pos + h, len(df) - 1)

        future_path = df.iloc[start_pos+1:end_pos+1]["close"]
        hit_label = 0
        hit_type = "time_out"

        for future_price in future_path:
            if signal == 1:  # Long signal
                if future_price >= upper_barrier:
                    hit_label = 1 #Profitable 
                    hit_type = "upper_target"
                    break
                elif future_price <= lower_barrier:
                    hit_label = 0 # Stopped out 
                    hit_type = "stop_loss"
                    break
            elif signal == -1:  # Short signal
                if future_price <= lower_barrier:
                    hit_label = 1
                    hit_type = "lower_target"
                    break
                elif future_price >= upper_barrier:
                    hit_label = 0
                    hit_type = "stop_loss"
                    break

        labels.loc[idx] = (hit_label, hit_type)

    return labels.dropna()


def calculate_trade_returns(df, p_mult, sl_mult, h):
    df["returns"] = df["close"].pct_change()
    df["vol"] = df["returns"].rolling(20).std()

    active_bets = df[df["primary_signal"] != 0].copy()
    trade_returns = pd.Series(index=active_bets.index, dtype=float)

    for idx in active_bets.index:
        signal = active_bets.loc[idx, "primary_signal"]
        entry_price = active_bets.loc[idx, "close"]
        vol = active_bets.loc[idx, "vol"]

        if pd.isna(vol):
            continue

        upper_barrier = entry_price * (1 + p_mult * vol)
        lower_barrier = entry_price * (1 - sl_mult * vol)

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

def optimize_barriers(df, signal, p_mult_range, sl_mult_range, h_range):
    results= []
    param_combinations = list(itertools.product(p_mult_range, sl_mult_range, h_range))

    print(f"Testing {len(param_combinations)} parameter combinations...")

    for p_mult, sl_mult, h_mult in param_combinations:
        trade_returns = calculate_trade_returns(df, p_mult, sl_mult, h_mult)
        
        if len(trade_returns)> 0 and trade_returns.std() != 0:
            avg_return = trade_returns.mean()
            std_ret = trade_returns.std()
            sharpe = avg_return / std_ret
            hit_rate = (trade_returns > 0).mean()

        else: 
            sharpe = 0.0
            hit_rate = 0.0
        
        results.append({
            "p_mult": p_mult,
            "sl_mult": sl_mult,
            "hit_rate": hit_rate,
            "sharpe": sharpe,
            "hit_rate": hit_rate
        })
    
    results_df = pd.DataFrame(results)
    best_params = results_df.sort_values("sharpe", ascending=False).iloc[0]
    return results_df, best_params

if __name__ == "__main__":
    panel, primary_signals = load_panel()
    print(primary_signals["cl1s"])
    p_range = [1.0, 1.5, 2.0, 2.5]
    sl_range = [0.5, 1.0, 1.5]
    h_range = [5, 10, 15]

    all_results, winner = optimize_barriers(panel["cl1s"], primary_signals["cl1s"], p_range, sl_range, h_range)
    print("Optimization complete. Best parameters:")
    print(winner)

