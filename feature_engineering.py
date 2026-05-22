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
def features_microstructure(df):
    close, volume = df["close"], df["volume"]
    ret = log_ret(close)
    dollar_vol = close * volume
    out = pd.DataFrame(index=df.index)

    # Amihud illiquidity: |return| per dollar of volume 
    out["amihud_20d"] = (ret.abs() / dollar_vol.replace(0, np.nan)).rolling(20).mean()

    # Roll's effective spread (only valid where Cov < 0)
    dp = close.diff()
    cov_roll = (dp * dp.shift(1)).rolling(20).mean()
    out["roll_spread_20d"] = 2 * np.sqrt((-cov_roll).clip(lower=0))

    # Volume z-score 
    out["volume_zscore_20d"] = rolling_zscore(volume, 20)

    # Log dollar volume-liquidity level 
    out["dollar_volume_log"] = np.log(dollar_vol.replace(0, np.nan))

    # Kyle's lambda: price impact per unit of volume
    # How much does price move per unit of volume? Higher lambda = less liquid
    abs_ret = ret.abs()
    sqrt_v = np.sqrt(volume.replace(0, np.nan))
    cov_kv = (abs_ret * sqrt_v).rolling(20).mean() - abs_ret.rolling(20).mean() * sqrt_v.rolling(20).mean()
    var_v = sqrt_v.rolling(20).var()
    out["kyle_lambda_20d"] = cov_kv / var_v.replace(0, np.nan)

    return out

# Group D: Mean-Reversion VS Trend 

def features_meanrev_trend(df):
    open, high, low, close = df["open"], df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)
 
    # RSI 14 - Wilder's original EWMA-based formulation 
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    #Bollinger Band position: how close is price to upper/lower band? (in sigma units)
    ma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    out["bb_position_20_2"] = (close - ma_20) / (2*std_20)

    #MACD (12, 26, 9) - trend-following momentum indicator
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_signal"] = macd_signal
    out["macd_histogram"] = macd - macd_signal

    #ADX 14 - average directional index, measures strength of trend (not direction)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx_14"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # William %R 14 - measures overbought/oversold conditions, similar to RSI but inverted
    highest_high_14 = high.rolling(14).max()
    lowest_low_14 = low.rolling(14).min()
    out["williams_r_14"] = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14).replace(0, np.nan)

    # Distance from 200d MA in sigma units (using 60d sd)
    ma_200 = close.rolling(200).mean()
    std_60 = close.rolling(60).std()
    out["distance_from_200d_ma"] = (close - ma_200) / std_60.replace(0, np.nan)

    ## CCI 20 - commodity channel index, measures deviation from mean in terms of typical price
    typical_price = (high + low + close) / 3
    sma_typical = typical_price.rolling(20).mean()
    mad_typical = (typical_price - sma_typical).abs().rolling(20).mean()
    out["cci_20"] = (typical_price - sma_typical) / (0.015 * mad_typical).replace(0, np.nan)

    return out

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    
    panel, primary_signals = load_panel(
        ohlcv_path="ohlcv_data.csv",
        signals_path="primary_signals.csv",
    )
    
    feats_c = features_microstructure(panel["cl1s"])
    feats_d = features_meanrev_trend(panel["cl1s"])
    
    print("Group C (microstructure) shape:", feats_c.shape)
    print(feats_c.describe().round(4))

    print("FINDING BUGS IN GROUP C")
    print(feats_c.describe().apply(lambda x: x.map('{:.3e}'.format)))
    print("Roll spread min:", feats_c["roll_spread_20d"].min())
    dp = panel["cl1s"]["close"].diff()
    cov_roll = (dp * dp.shift(1)).rolling(20).mean()
    print("cov_roll: total =", len(cov_roll))
    print("  NaN:", cov_roll.isna().sum())
    print("  Positive (bad for Roll):", (cov_roll > 0).sum())
    print("  Negative (good for Roll):", (cov_roll < 0).sum())
    print("Roll spread breakdown:")
    print("  Total rows:", len(feats_c))
    print("  NaN values:", feats_c['roll_spread_20d'].isna().sum())
    print("  Zero values:", (feats_c['roll_spread_20d'] == 0).sum())
    print("  Positive values:", (feats_c['roll_spread_20d'] > 0).sum())
    print("Volume zeros in cl1s:", (panel["cl1s"]["volume"] == 0).sum())

    print()
    print("Group D (mean-rev/trend) shape:", feats_d.shape)
    print(feats_d.describe().round(4))



