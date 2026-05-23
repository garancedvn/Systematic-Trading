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
    #train until 2021-12-30, and test on the rest 
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
    
# Group F: Spectral and Fractal Feeatures 
def rolling_apply_array(s, w, fn):
    vals = s.values 
    n = len(s)
    out_vals = np.full(n, np.nan)
    for i in range(w - 1, n):
        window = vals[i - w + 1 : i + 1]
        if np.all(np.isfinite(window)):
            try:
                out_vals[i] = fn(window)
            except Exception:
                out_vals[i] = np.nan
    return pd.Series(out_vals, index=s.index)

def dominant_cycle_period(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 8 or np.std(x) == 0:
        return np.nan
    
    t = np.arange(len(x))
    coeffs = np.polyfit(t, x, 1)
    detrended = x - np.polyval(coeffs, t)
    
    spec = np.abs(np.fft.rfft(detrended))
    # Zero out: DC + the lowest 2 bins (which absorb residual trend curvature)
    spec[:3] = 0
    
    if spec.sum() == 0:
        return np.nan
    
    k = np.argmax(spec)
    return len(x) / max(k, 1)

def spectral_entropy(x):
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    spec = spec[1:]  # drop DC component
    total = spec.sum()
    
    if total <= 0:
        return np.nan
    
    p = spec / total
    p = p[p > 0]
    return -(p * np.log(p)).sum() / np.log(len(p))

def hurst(x: np.ndarray) -> float:
    """
    Hurst exponent via the rescaled range (R/S) method.
    Original Hurst-Mandelbrot estimator.
    H ≈ 0.5: random walk
    H > 0.5: persistent (trending)
    H < 0.5: anti-persistent (mean-reverting)
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 20 or np.std(x) == 0:
        return np.nan
    
    # Work on increments (returns), not levels
    returns = np.diff(x)
    if len(returns) < 10 or np.std(returns) == 0:
        return np.nan
    
    # Build R/S for several scales
    scales = []
    rs_values = []
    for n in [10, 20, 30, 40, 50, 60, 80]:
        if n > len(returns):
            break
        # Split into chunks of size n
        n_chunks = len(returns) // n
        if n_chunks < 2:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = returns[i * n : (i + 1) * n]
            mean_c = chunk.mean()
            z = np.cumsum(chunk - mean_c)
            R = z.max() - z.min()         # range of cumulative deviations
            S = np.std(chunk, ddof=1)     # std of chunk
            if S > 0 and np.isfinite(R / S):
                rs_chunk.append(R / S)
        if len(rs_chunk) >= 1:
            scales.append(n)
            rs_values.append(np.mean(rs_chunk))
    
    if len(scales) < 3:
        return np.nan
    
    # log(R/S) ~ H * log(n) for fBm
    slope, _ = np.polyfit(np.log(scales), np.log(rs_values), 1)
    return slope
def dfa(x):
    if len(x) < 16:
        return np.nan
    
    y = np.cumsum(x - x.mean())
    
    scales = [s for s in [4, 8, 16] if s < len(x) // 2]
    if len(scales) < 2:
        return np.nan
    
    f = []
    for s in scales:
        n_segments = len(y) // s
        rms_list = []
        for i in range(n_segments):
            seg = y[i * s : (i + 1) * s]
            t = np.arange(s)
            poly = np.polyfit(t, seg, 1)
            trend = np.polyval(poly, t)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        f.append(np.mean(rms_list) if rms_list else np.nan)
    
    f = np.array(f)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        return np.nan
    
    slope, _ = np.polyfit(np.log(scales), np.log(f), 1)
    return slope

def approx_entropy(x: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """
    Approximate entropy (Pincus 1991). Lower = more predictable.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return np.nan
    
    sd = np.std(x)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    r = r_mult * sd
    
    def _phi(m_):
        # Build all sub-sequences of length m_
        N = n - m_ + 1
        if N <= 0:
            return np.nan
        patterns = np.zeros((N, m_))
        for i in range(N):
            patterns[i] = x[i : i + m_]
        
        # For each pattern, count how many others are within r in Chebyshev distance
        counts = np.zeros(N)
        for i in range(N):
            diff = np.abs(patterns - patterns[i])
            max_diff = diff.max(axis=1)
            counts[i] = np.sum(max_diff <= r) / N
        
        # Avoid log(0)
        counts = np.where(counts > 0, counts, 1e-12)
        return np.mean(np.log(counts))
    
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    if not (np.isfinite(phi_m) and np.isfinite(phi_m1)):
        return np.nan
    return phi_m - phi_m1

def features_spectral_fractal(df):
    c = df["close"]
    ret = log_ret(c).fillna(0.0)
    log_c = np.log(c)
    out = pd.DataFrame(index=df.index)
    
    out["dominant_cycle_period"] = rolling_apply_array(log_c, 60, dominant_cycle_period)
    out["spectral_entropy"] = rolling_apply_array(ret, 60, spectral_entropy)
    out["hurst_90d"] = rolling_apply_array(log_c, 90, hurst)
    out["dfa_alpha_90d"] = rolling_apply_array(ret, 90, dfa)
    out["approx_entropy_20d"] = rolling_apply_array(ret, 20, approx_entropy)
    
    return out

# Group G: Cross-Sectional Features

def features_cross_sectional(df, target_ticker, panel, asset_class_tickers, anchor_ticker=None):
    peers = [t for t in asset_class_tickers if t != target_ticker]
    out = pd.DataFrame(index=df.index)
    idx = df.index

    if len(peers) == 0:
        for col in [
            "corr_basket_60d", "xs_rank_5d", "xs_dispersion_20d",
            "leadlag_anchor", "vol_ratio_basket", "beta_basket_60d",
        ]:
            out[col] = np.nan
        return out
    
    # Target's returns at relevant horizons
    ret_target_1d = log_ret(df["close"], 1)
    ret_target_5d = log_ret(df["close"], 5)
    ret_target_20d = log_ret(df["close"], 20)
    
    # Peer returns at three horizons, all aligned to target's index
    peer_rets_1d = pd.DataFrame({
        t: log_ret(panel[t]["close"], 1) for t in peers
    }).reindex(idx)
    peer_rets_5d = pd.DataFrame({
        t: log_ret(panel[t]["close"], 5) for t in peers
    }).reindex(idx)
    peer_rets_20d = pd.DataFrame({
        t: log_ret(panel[t]["close"], 20) for t in peers
    }).reindex(idx)
    
    # Equal-weight basket return (peers only, target excluded)
    basket_ret = peer_rets_1d.mean(axis=1)
    
    # Feature 1: 60-day correlation with basket
    out["corr_basket_60d"] = ret_target_1d.rolling(60).corr(basket_ret)
    
    # Feature 2: cross-sectional rank of 5d return within asset class
    all_5d = peer_rets_5d.copy()
    all_5d[target_ticker] = ret_target_5d
    ranks = all_5d.rank(axis=1, pct=True)
    out["xs_rank_5d"] = ranks[target_ticker]
    
    # Feature 3: cross-sectional dispersion of 20d returns
    all_20d = peer_rets_20d.copy()
    all_20d[target_ticker] = ret_target_20d
    out["xs_dispersion_20d"] = all_20d.std(axis=1)
    
    # Feature 4: 60-day correlation with lagged anchor (lead-lag relationship)
    if anchor_ticker is not None and anchor_ticker != target_ticker and anchor_ticker in panel:
        anchor_ret_lagged = log_ret(panel[anchor_ticker]["close"], 1).shift(1).reindex(idx)
        out["leadlag_anchor"] = ret_target_1d.rolling(60).corr(anchor_ret_lagged)
    else:
        out["leadlag_anchor"] = np.nan
    
    # Feature 5: vol ratio (target / basket mean)
    vol_target = yang_zhang_vol(
        df["open"], df["high"], df["low"], df["close"], 20
    )
    vol_peers = pd.DataFrame({
        t: yang_zhang_vol(
            panel[t]["open"], panel[t]["high"], panel[t]["low"], panel[t]["close"], 20
        ) for t in peers
    }).reindex(idx)
    out["vol_ratio_basket"] = vol_target / vol_peers.mean(axis=1).replace(0, np.nan)
    
    # Feature 6: 60-day rolling beta of target on basket
    cov_tb = (
        (ret_target_1d * basket_ret).rolling(60).mean()
        - ret_target_1d.rolling(60).mean() * basket_ret.rolling(60).mean()
    )
    var_b = basket_ret.rolling(60).var()
    out["beta_basket_60d"] = cov_tb / var_b.replace(0, np.nan)
    
    return out

# Group H: Primary Signal Interaction Features 

def features_primary_signal_interaction(df, primary_signal):
    out = pd.DataFrame(index=df.index)
    signal_full = primary_signal.reindex(df.index)
    signal = signal_full.dropna().astype(int)
    
    out["primary_signal"] = signal_full

    out["signal_changed"] = (signal != signal.shift(1)).astype(int)

    change = (signal != signal.shift(1)).astype(int)
    grp = change.cumsum()
    out["signal_persistence"] = grp.groupby(grp).cumcount()

    ret_50 = log_ret(df["close"], 50)
    concord_full = np.sign(signal_full) * np.sign(ret_50)
    out["signal_trend_concord"] = concord_full

    out["signal_density_20d"] = (signal!=0).rolling(20).mean()

    return out 

#Group I: Seasonality Features 
def seasonal_ramp(months, days_in_month, start_month, peak_month, end_month):
    month_frac = (months - 1) + (days_in_month - 1) / 31.0

    diff = month_frac - (peak_month - 1)
    diff = (diff + 6) % 12 - 6   # wrap to [-6, 6]
    
    # Distance from peak to start (negative direction) and to end (positive direction)
    dist_to_start = (peak_month - start_month) % 12
    if dist_to_start == 0:
        dist_to_start = 1
    dist_to_end = (end_month - peak_month) % 12
    if dist_to_end == 0:
        dist_to_end = 1
    
    # Ramp: 1 at the peak (diff=0), 0 at edges
    ramp = np.where(
        diff < 0,
        1 + diff / dist_to_start,   # rising side
        1 - diff / dist_to_end,     # falling side
    )
    return np.clip(ramp, 0, 1)

def features_seasonality(df):
    out = pd.DataFrame(index=df.index)

    doy = df.index.dayofyear
    out["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    months = df.index.month
    days_in_month = df.index.days_in_month
    out["heating_season"] = seasonal_ramp(months, days_in_month, start_month=10, peak_month=1, end_month=4)

    out["driving_season_progress"] = seasonal_ramp(months, days_in_month, start_month=4, peak_month=7, end_month=8)

    out["hurricane_season_indicator"] = seasonal_ramp(months, days_in_month, start_month=6, peak_month=9, end_month=11)

    quarter_starts = pd.Series(df.index, index= df.index).dt.to_period("Q").dt.start_time
    quarter_ends = pd.Series(df.index, index= df.index).dt.to_period("Q").dt.end_time
    quarter_length = (quarter_ends - quarter_starts).dt.days 
    elapsed = (df.index - quarter_starts).dt.days
    out["quarter_progress"] = elapsed / quarter_length.clip(lower=1).values 

    return out

# Group J: Higher moments and range position 
def autocorr_lag1(x):
    """
    Lag-1 autocorrelation of a series.
    Positive = trending (consecutive moves same direction).
    Negative = mean-reverting (consecutive moves reverse).
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 10 or np.std(x) == 0:
        return np.nan
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def features_higher_moments_range(df):
    open, high, low, close = df["open"], df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)

    ret = log_ret(close)

    # Skew of recent returns - asymmetry of the return distribution
    out["skew_20d"] = ret.rolling(20).skew()
    out["skew_60d"] = ret.rolling(60).skew()

    # Kurtosis - tail thickness, captures jump risk
    out["kurt_20d"] = ret.rolling(20).kurt()

    # Downside volatility - std of negative returns only, Sortino denominator
    out["downside_vol_20d"] = ret.rolling(20).apply(
        lambda x: np.std(x[x < 0]) if (x < 0).sum() > 1 else np.nan, raw=True
    )

    # Position in 60d high-low range: 0 = at low, 1 = at high
    hh_60 = high.rolling(60).max()
    ll_60 = low.rolling(60).min()
    out["price_range_position_60d"] = (close - ll_60) / (hh_60 - ll_60).replace(0, np.nan)

    # Change in range position over 5 days - velocity of range positioning
    out["range_position_5d_chg"] = out["price_range_position_60d"].diff(5)

    # Return-vol correlation: leverage effect indicator
    # Negative = drops drive vol spikes (canonical bear), positive = unusual
    vol_yz = yang_zhang_vol(open, high, low, close, window=20)
    out["return_vol_correl_60d"] = ret.rolling(60).corr(vol_yz.diff())

    # Short-window Hurst, complements the 90d Hurst in Group F
    ret_filled = ret.fillna(0.0)
    out["return_autocorr_30d"] = rolling_apply_array(ret_filled, 30, autocorr_lag1)

    return out

FEATURE_GROUPS = {
    "A_returns_momentum": [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "ret_20d_zscore", "mom_12_1", "roc_10d", "mom_3m_minus_1m",
    ],
    "B_volatility": [
        "vol_cc_20d", "vol_parkinson_20d", "vol_garman_klass_20d",
        "vol_rogers_satchell_20d", "vol_yang_zhang_20d",
        "vol_of_vol_60d", "vol_yz_zscore_252d", "hl_range_close",
    ],
    "C_microstructure": [
        "amihud_20d", "roll_spread_20d", "volume_zscore_20d",
        "dollar_volume_log", "kyle_lambda_20d",
    ],
    "D_meanrev_trend": [
        "rsi_14", "bb_position_20_2", "macd_signal", "macd_histogram",
        "adx_14", "williams_r_14", "distance_from_200d_ma", "cci_20",
    ],
    "E_latent_regime": [
        "hmm_p_state0", "hmm_p_state1", "hmm_state_persistence",
        "gmm_logdensity", "gmm_argmax",
    ],
    "F_spectral_fractal": [
        "dominant_cycle_period", "spectral_entropy",
        "hurst_90d", "dfa_alpha_90d", "approx_entropy_20d",
    ],
    "G_cross_sectional": [
        "corr_basket_60d", "xs_rank_5d", "xs_dispersion_20d",
        "leadlag_anchor", "vol_ratio_basket", "beta_basket_60d",
    ],
    "H_signal_interaction": [
        "primary_signal", "signal_changed", "signal_persistence",
        "signal_trend_concord", "signal_density_20d",
    ],
    "I_seasonality": [
        "day_of_year_sin", "day_of_year_cos",
        "heating_season", "driving_season_progress",
        "hurricane_season_indicator", "quarter_progress",
    ],
    "J_higher_moments_range": [
        "skew_20d", "skew_60d", "kurt_20d", "downside_vol_20d",
        "price_range_position_60d", "range_position_5d_chg",
        "return_vol_correl_60d", "return_autocorr_30d"
    ]
}


def build_features_single(df, primary_signal, regime_models=None,
                          cross_sectional_inputs=None):
    parts = [
        features_returns_momentum(df),
        features_volatility(df),
        features_microstructure(df),
        features_meanrev_trend(df),
        features_spectral_fractal(df),
        features_primary_signal_interaction(df, primary_signal),
        features_seasonality(df),
        features_higher_moments_range(df),
    ]
    
    if regime_models is not None:
        parts.append(regime_models.transform(df))
    
    if cross_sectional_inputs is not None:
        parts.append(features_cross_sectional(
            df=df,
            target_ticker=cross_sectional_inputs["target_ticker"],
            panel=cross_sectional_inputs["panel"],
            asset_class_tickers=cross_sectional_inputs["asset_class_tickers"],
            anchor_ticker=cross_sectional_inputs.get("anchor_ticker"),
        ))
    
    return pd.concat(parts, axis=1)


def build_features_panel(panel, primary_signals, asset_class_tickers,
                         anchor_ticker=None, train_end_date=None,
                         fit_regime_models=True):

    if train_end_date is None:
        train_end_date = pd.Timestamp("2019-12-31")
    
    out = {}
    for tk in asset_class_tickers:
        print(f"Building features for {tk}...")
        
        regime_models = None
        if fit_regime_models:
            train_df = panel[tk].loc[:train_end_date]
            regime_models = LatentRegimeModels(n_states=2, random_state=42).fit(train_df)
        
        xs_inputs = {
            "target_ticker": tk,
            "panel": panel,
            "asset_class_tickers": asset_class_tickers,
            "anchor_ticker": anchor_ticker,
        }
        
        out[tk] = build_features_single(
            df=panel[tk],
            primary_signal=primary_signals[tk],
            regime_models=regime_models,
            cross_sectional_inputs=xs_inputs,
        )
    
    return out

if __name__ == "__main__":
    panel, primary_signals = load_panel()
    feats_j = features_higher_moments_range(panel["cl1s"])
    print("Group J shape:", feats_j.shape)
    print(feats_j.describe().round(4))