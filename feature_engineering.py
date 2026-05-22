import pandas as pd
import numpy as np

from load_data import load_panel

from dataclasses import dataclass, field
from typing import Optional

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMM = True
except ImportError:
    _HAS_HMM = False

try:
    from sklearn.mixture import GaussianMixture
    _HAS_GMM = True
except ImportError:
    _HAS_GMM = False

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

# Group E: Latent Regime Features 
@dataclass 
class LatentRegimeModels:
    """
    HMM + GMM fitted on a training fold, applied to any data.
    
    Usage:
        regime = LatentRegimeModels(n_states=3).fit(train_df)
        features_train = regime.transform(train_df)
        features_test  = regime.transform(test_df)   # uses train-fold-fitted models
    """
    n_states: int = 2
    random_state: int = 42
    hmm_model: Optional[object] = field(default=None, init=False)
    gmm_model: Optional[object] = field(default=None, init=False)
    state_order: Optional[np.ndarray] = field(default=None, init=False)
    
    def _feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """The (ret_5d, vol_yz_20d) input matrix used by both models."""
        ret_5d = log_ret(df["close"], 5)
        vol_yz = yang_zhang_vol(df["open"], df["high"], df["low"], df["close"], 20)
        return pd.concat(
            [ret_5d.rename("ret_5d"), vol_yz.rename("vol_yz_20d")],
            axis=1,
        )
    
    def fit(self, df_train: pd.DataFrame):
        """Fit HMM and GMM on training-fold data only."""
        if not _HAS_HMM or not _HAS_GMM:
            raise ImportError("Install hmmlearn and scikit-learn for regime features.")
        
        X = self._feature_matrix(df_train).dropna()
        
        # Fit HMM
        self.hmm_model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
        )
        self.hmm_model.fit(X.values)
        
        # Sort states by mean return ascending: 0 = bear, 1 = chop, 2 = bull
        means_ret = self.hmm_model.means_[:, 0]
        self.state_order = np.argsort(means_ret)
        
        # Fit GMM
        self.gmm_model = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            random_state=self.random_state,
        )
        self.gmm_model.fit(X.values)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.hmm_model is None or self.gmm_model is None:
            raise RuntimeError("Call .fit() before .transform().")
        
        X = self._feature_matrix(df)
        
        # Build column names
        cols = (
            [f"hmm_p_state{i}" for i in range(self.n_states)]
            + ["hmm_state_persistence", "gmm_logdensity", "gmm_argmax"]
        )
        out = pd.DataFrame(np.nan, index=df.index, columns=cols)
        
        mask = X.notna().all(axis=1)
        if mask.sum() == 0:
            return out
        
        X_clean = X[mask].values
        
        # HMM posterior probabilities (reorder columns by state_order)
        post = self.hmm_model.predict_proba(X_clean)
        post = post[:, self.state_order]
        for i in range(self.n_states):
            out.loc[mask, f"hmm_p_state{i}"] = post[:, i]
        
        # HMM MAP state — same ordering
        states_raw = self.hmm_model.predict(X_clean)
        inv = np.argsort(self.state_order)
        states_ordered = np.array([inv[s] for s in states_raw])
        
        # Persistence
        states_series = pd.Series(np.nan, index=df.index, dtype=float)
        states_series.loc[mask] = states_ordered
        change = (states_series != states_series.shift(1)).astype(int)
        grp = change.cumsum()
        persistence = grp.groupby(grp).cumcount()
        out.loc[mask, "hmm_state_persistence"] = persistence[mask].values
        
        # GMM features
        out.loc[mask, "gmm_logdensity"] = self.gmm_model.score_samples(X_clean)
        out.loc[mask, "gmm_argmax"] = self.gmm_model.predict(X_clean)
        
        return out

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    
    panel, primary_signals = load_panel(
        ohlcv_path="ohlcv_data.csv",
        signals_path="primary_signals.csv",
    )
    
    cl_df = panel["cl1s"]
    
    # ----- Fit regime models on training fold only (pre-2020) -----
    train_end = pd.Timestamp("2019-12-31")
    train_df = cl_df.loc[:train_end]
    
    print("=" * 60)
    print("GROUP E — Latent Regime Features")
    print("=" * 60)
    print(f"Train fold: {train_df.index.min().date()} to {train_df.index.max().date()} "
          f"({len(train_df)} rows)")
    print(f"Full sample: {cl_df.index.min().date()} to {cl_df.index.max().date()} "
          f"({len(cl_df)} rows)")
    print()
    
    print("Fitting HMM and GMM on training fold...")
    regime = LatentRegimeModels(n_states=2, random_state=42).fit(train_df)
    print("Done.")
    print()
    
    # ----- Inspect what the HMM learned -----
    print("HMM state means (after ordering by mean return):")
    print(f"{'State':<8} {'Label':<8} {'ret_5d':>10} {'vol_yz_20d':>12}")
    labels = ["stressed", "normal"]
    for i in range(2):
        raw_state = regime.state_order[i]
        m = regime.hmm_model.means_[raw_state]
        print(f"{i:<8} {labels[i]:<8} {m[0]:>10.4f} {m[1]:>12.4f}")
    print()
    
    print("HMM transition matrix (rows = from, cols = to, in ordered labels):")
    inv = np.argsort(regime.state_order)
    raw_tm = regime.hmm_model.transmat_
  
    ordered_tm = raw_tm[regime.state_order][:, regime.state_order]
    print(pd.DataFrame(ordered_tm, index=labels, columns=labels).round(3))
    print()
    
    # ----- Apply to full sample -----
    feats_e = regime.transform(cl_df)
    print(f"Group E shape: {feats_e.shape}")
    print()
    
    print("Summary statistics:")
    print(feats_e.describe().round(4))
    print()
    
    # ----- Sanity checks -----
    post_sum = feats_e[["hmm_p_state0", "hmm_p_state1"]].dropna().sum(axis=1)
    print(f"Posterior probabilities sum to ~1 (valid rows only)?  "
        f"mean={post_sum.mean():.6f}, std={post_sum.std():.2e}")
    print(f"Warmup rows (NaN posteriors): {feats_e['hmm_p_state0'].isna().sum()}")
    print()
    
    print("Last 5 days — recent regime state:")
    print(feats_e[[
        "hmm_p_state0", "hmm_p_state1",
        "hmm_state_persistence", "gmm_logdensity", "gmm_argmax"
    ]].tail().round(3))
    print()
    
    # ----- Find the most anomalous days -----
    print("Top 5 most anomalous days by GMM log-density:")
    anomalies = feats_e.nsmallest(5, "gmm_logdensity")[
        ["gmm_logdensity", "hmm_p_state0", "hmm_p_state1"]
    ]
    print(anomalies.round(3))
