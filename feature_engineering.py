import pandas as pd
import numpy as np

from load_data import load_panel

#Group A: Returns and Momentum Features
def log_ret(c,n=1):
    return np.log(c) - np.log(c.shift(n))

def rolling_zscore(s, w):
    mean = s.rolling(w).mean()
    std = s.rolling(w).std()
    zscore = (s - mean) / std
    return zscore

def features_returns_momentum(df):
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    for n in [1, 5, 10, 20, 60]:
        out[f"ret_{n}d"] = log_ret(c, n)
    out["ret_20d_zscore"] = rolling_zscore(out["ret_20d"], 252)
    out["mom_12_1"] = log_ret(c, 252) - log_ret(c, 21)
    out["roc_10d"] = (c / c.shift(10)) - 1.0
    out["mom_3m_minus_1m"] = log_ret(c, 63) - log_ret(c, 21)
    return out

#Group B: Volatility Estimators 

# Parkinson's volatility estimator: less noise than close-to-close, uses high and low prices
def parkinson_vol(high, low, window=20):
    factor = 1.0 / (4.0 * np.log(2.0))
    hl = np.log(high / low) ** 2
    return np.sqrt(factor * hl.rolling(window).mean())

# More efficient than Parkinson 
def garman_klass_vol(o, h, l, c, window: int = 20) -> pd.Series:
    hl = 0.5 * np.log(h / l) ** 2
    co = (2 * np.log(2) - 1) * np.log(c / o) ** 2
    daily_var = (hl - co).clip(lower=0)             # ← clip before rolling
    return np.sqrt(daily_var.rolling(window).mean())

def rogers_satchell_vol(o, h, l, c, window: int = 20) -> pd.Series:
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(rs.clip(lower=0).rolling(window).mean())

def yang_zhang_vol(open, high, low, close, window=20):
    close_prev = close.shift(1)
    overnight = (np.log(open/close_prev)) ** 2
    open_to_close = (np.log(close/open)) ** 2

    sigma_overnight = overnight.rolling(window).mean()
    sigma_open_to_close = open_to_close.rolling(window).mean()

    rs = np.log(high / open) * np.log(high / close) + np.log(low / open) * np.log(low / close)
    sigma_rs = rs.rolling(window).mean()

    k= 0.34 / (1.34 + (window + 1) / (window - 1))
    var = sigma_overnight + k * sigma_open_to_close + (1-k) * sigma_rs
    return np.sqrt(var.clip(lower=0))

def features_volatility(df):
    open, high, low, close = df["open"], df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)

    out["vol_cc_20d"] = log_ret(close).rolling(20).std() 
    out["vol_parkinson_20d"] = parkinson_vol(high, low, window=20)
    out["vol_garman_klass_20d"] = garman_klass_vol(open, high, low, close, window=20)
    out["vol_rogers_satchell_20d"] = rogers_satchell_vol(open, high, low, close, window=20)
    out["vol_yang_zhang_20d"] = yang_zhang_vol(open, high, low, close, window=20)
    
    # How stable is the volatility?
    out["vol_of_vol_60d"] = out["vol_yang_zhang_20d"].rolling(60).std()

    # Z-score of volatility to identify unusually high or low volatility periods
    out["vol_yz_zscore_252d"] = rolling_zscore(out["vol_yang_zhang_20d"], 252)

    # Average daily range as fraction of price 
    out["hl_range_close"] = ((high - low) / close).rolling(20).mean()
    return out

# Group C: Microstructure Features



if __name__ == "__main__":
    panel, primary_signals = load_panel()
    feats_b = features_volatility(panel["cl1s"])
    print("Group B shape:", feats_b.shape)
    print("Columns:", feats_b.columns.tolist())
    print()
    print("Last 5 rows:")
    print(feats_b.tail().round(4))
    print()
    print("Summary statistics:")
    print(feats_b.describe().round(4)) #Big crash 2020-2022 

    # Add this at the end of your driver block
    print("Comparison of mean daily vol across estimators (cl1s, full sample):")
    for col in ["vol_cc_20d", "vol_parkinson_20d", "vol_garman_klass_20d",
                "vol_rogers_satchell_20d", "vol_yang_zhang_20d"]:
        print(f"  {col:30s} = {feats_b[col].mean():.5f}")

