# Metamodel for Systematic Trading — Methodology
## Parts 1 & 2: Feature Engineering and Triple-Barrier Labeling

**Asset-class focus:** Energy (CL1S, HO1S, RB1S, NG1S). The codebase is instrument-agnostic and runs on any of the 11 instruments in the universe.

**Reference framework:** Lopez de Prado, *Advances in Financial Machine Learning* (AFML), Chapters 2–5, where the meta-labeling paradigm and triple-barrier method originate.

---

## 0. The Meta-Labeling Frame (why this changes everything)

The brief states that we are given a primary model's daily signal `s_t ∈ {−1, 0, +1}`. The metamodel sits on top of this signal and outputs `P(the bet is worth taking | features) ∈ [0, 1]`. This is **meta-labeling** in the AFML sense.

Three implications follow immediately, and every downstream design choice in this report flows from them:

1. **The label is binary.** Not {−1, 0, +1}. The label answers: *conditional on taking the primary signal, was the trade profitable within the time horizon?*
2. **Events with `s_t = 0` are discarded from training.** There is no bet to evaluate. They re-enter at inference time only as a degenerate prediction (the metamodel never overrides a flat primary signal — the meta-model is a *filter*, not a generator).
3. **The triple-barrier method must be applied *in the direction of the primary signal***. The "profit barrier" is above the entry price when `s_t = +1` and below when `s_t = −1`. This asymmetry matters for the labeling logic and is wrong in most off-the-shelf implementations.

This framing also re-prioritises the feature engineering: features that describe *the relationship between the primary signal and the broader market state* are the largest source of edge, because the metamodel's only job is to know when the primary signal is reliable.

---

## 1. Feature Engineering (20 marks)

### 1.1 Design philosophy

Features are organised into **eight thematic groups**. The grouping is not cosmetic: in Part 4 we compute cluster-level feature importance (MDA/MDI/SHAP at the cluster level, per AFML §8), and pre-defining economically meaningful groups makes the cluster importance analysis interpretable. Correlated features within a group will be jointly attributed; the marginal contribution of *each group* will then be the cleanest measure of where the metamodel's edge comes from.

All rolling statistics are **right-aligned and strictly causal** — the feature at time `t` uses only information available at the close of day `t`. No statistics are fitted on the full sample. HMM/GMM/scalers/PCA are fitted only on the training fold and applied to validation/test.

Each instrument produces an identical feature schema, so the models in Part 3 can be either per-instrument or pooled across the asset class.

### 1.2 Group A — Multi-Horizon Returns and Momentum (9 features)

| Feature | Definition | Captures |
|---|---|---|
| `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d` | Log returns over 1/5/10/20/60d | Term structure of momentum |
| `ret_20d_zscore` | (ret_20d − μ_252) / σ_252 | Standardised medium-term momentum |
| `mom_12_1` | 252d return excluding most recent 21d | Classical skip-recent momentum (Jegadeesh-Titman) |
| `roc_10d` | (P_t / P_{t−10}) − 1 | Rate of change, robust momentum proxy |
| `mom_3m_minus_1m` | 63d return − 21d return | Momentum acceleration / deceleration |

**Justification:** Momentum exists at multiple horizons and the predictive power of each horizon shifts with regime (AFML notes this explicitly; Moskowitz et al. 2012 documents time-series momentum at 1–12 months). Including the full term structure lets tree-based models recover whichever horizon currently matters.

### 1.3 Group B — Volatility Estimators (8 features)

The brief is OHLCV, which is precisely the data needed for the *efficient* volatility estimators that most students ignore.

| Feature | Definition | Why it matters |
|---|---|---|
| `vol_cc_20d` | std of log returns, 20d | Standard close-to-close baseline |
| `vol_parkinson_20d` | Parkinson estimator using H, L | ~5× more efficient than C2C |
| `vol_garman_klass_20d` | Garman-Klass using OHLC | ~7× efficient under no-drift assumption |
| `vol_rogers_satchell_20d` | Rogers-Satchell | Drift-robust |
| `vol_yang_zhang_20d` | Yang-Zhang (handles overnight) | **Most efficient OHLC estimator** |
| `vol_of_vol_60d` | std of `vol_yz_20d` over 60d | Vol regime stability |
| `vol_yz_zscore_252d` | Z-score of YZ vol vs 1y distribution | Vol regime position |
| `hl_range_close` | (H − L) / C, 20d mean | Daily range proxy |

**Justification:** Volatility is the single most important conditioning variable in financial ML and the cheapest place to gain efficiency. Using only close-to-close vol when you have OHLC data is leaving information on the table. The Yang-Zhang estimator is then the one we use to scale the triple-barrier widths in Part 2 — methodologically consistent.

### 1.4 Group C — Microstructure Proxies from Daily OHLCV (5 features)

| Feature | Definition | Captures |
|---|---|---|
| `amihud_illiq_20d` | mean of \|ret\| / dollar_volume, 20d | Price impact per traded dollar |
| `roll_spread_20d` | 2·√(−Cov(Δp_t, Δp_{t−1})) when cov < 0 | Effective bid-ask spread (Roll 1984) |
| `volume_zscore_20d` | (V_t − μ_20) / σ_20 | Unusual volume |
| `dollar_volume_log` | log(C × V) | Liquidity proxy |
| `kyle_lambda_20d` | Slope of \|ret\| on signed-sqrt-volume, 20d rolling | Price impact (Kyle's λ) |

**Justification:** Liquidity and price-impact features are usually associated with high-frequency data, but daily OHLCV is enough for reasonable proxies. These features help the metamodel identify *when* the primary signal is hitting a market that can absorb it — a low-liquidity day with the primary signal flipping is often a bad day to bet.

### 1.5 Group D — Mean Reversion vs Trend Indicators (8 features)

| Feature | Notes |
|---|---|
| `rsi_14` | Standard Wilder RSI |
| `bb_position_20_2` | (C − MA_20) / (2·σ_20) — Bollinger band position in σ units |
| `macd_signal`, `macd_hist` | MACD line and histogram |
| `adx_14` | Trend strength (not direction) |
| `williams_r_14` | Williams %R |
| `dist_ma200_sigma` | (C − MA_200) / σ_60 — distance from long MA in σ units |
| `cci_20` | Commodity Channel Index — particularly relevant for the energy/metals universe |

Standard technical indicators, but normalised in σ-units rather than raw price units so they're cross-instrument comparable.

### 1.6 Group E — Latent Regime Features (6 features) ★ creative

This is where the unsupervised methods earn their place. Two **independent** latent variable models are fitted on `(ret_5d, vol_yz_20d)` tuples within each training fold.

| Feature | Definition |
|---|---|
| `hmm_p_state0`, `hmm_p_state1`, `hmm_p_state2` | Posterior state probabilities from a 3-state Gaussian HMM (interpreted post-hoc as bull / chop / bear by ordering states by mean return) |
| `hmm_state_persistence` | Days since the last MAP-state transition |
| `gmm_logdensity` | Log-likelihood of (ret_5d, vol_yz_20d) under a 3-component GMM — an anomaly / "regime fit" score |
| `gmm_argmax` | Hard cluster assignment from GMM (categorical, target-encoded) |

**Why both HMM and GMM?** They have different inductive biases. The HMM models the Markovian structure of regime *transitions*, so it gives a smooth posterior that depends on recent history. The GMM is memoryless — its cluster assignment depends only on today's features. Empirically they catch different things: the HMM is better at saying "we're in a high-vol regime," the GMM is better at saying "today is statistically unusual." Both as features lets the model use whichever signal is sharper.

This directly extends the **Low Turbulence Model pipeline** from the course — same family of method, applied to a different problem.

### 1.7 Group F — Spectral and Fractal Features (5 features) ★ creative

| Feature | Definition | Captures |
|---|---|---|
| `dominant_cycle_period` | argmax of FFT power spectrum on detrended 60d closes | Periodicity in the price series |
| `spectral_entropy` | Shannon entropy of normalised power spectrum | "How periodic" vs "how noisy" |
| `hurst_90d` | Hurst exponent via rescaled-range, 90d window | >0.5 trending, <0.5 mean-reverting |
| `dfa_alpha_90d` | Detrended fluctuation analysis exponent | More robust Hurst estimator |
| `approx_entropy_20d` | Approximate entropy of 20d returns | Return predictability |

**Justification:** Standard momentum/mean-reversion indicators *proxy* the regime structure of the series. Hurst, DFA, and ApEn measure it directly. This was a topic explicitly covered in the Low Turbulence Model context, and is in scope for "anything else you can justify." Spectral entropy is particularly informative: low entropy = cyclical regime, where mean reversion strategies tend to dominate; high entropy = noisy regime, where momentum signals struggle.

### 1.8 Group G — Cross-Asset / Cross-Sectional Features (6 features) ★ creative

For instruments within an asset class, cross-sectional features are often the largest source of metamodel edge. The primary signal typically *fails* when the instrument is moving idiosyncratically vs its peers.

For the Energy universe, the basket is `{CL, HO, RB, NG}`. Each instrument's cross-sectional features are computed relative to the basket excluding itself.

| Feature | Definition |
|---|---|
| `corr_basket_60d` | 60d correlation of returns with equal-weight basket of peer instruments |
| `xs_rank_5d` | Cross-sectional percentile rank of 5d return within the asset class |
| `xs_dispersion_20d` | Std across asset class of 20d returns (idiosyncratic vs systemic regime) |
| `leadlag_anchor` | Correlation with the asset-class anchor (CL for energy, ES for index, GC for metals) lagged by 1 day |
| `vol_ratio_basket` | YZ vol of instrument / mean YZ vol of basket |
| `beta_basket_60d` | OLS beta of instrument returns on basket returns, 60d window |

**Justification:** When dispersion is high, idiosyncratic moves dominate and instrument-specific primary signals are noisy. When `corr_basket_60d` is low and falling, the instrument is decoupling from its peers — often a sign of a regime-change about which the primary signal has no information. Cross-sectional rank captures whether the instrument is overbought *relative to its peers*, which is a stronger reversal signal than absolute overbought.

### 1.9 Group H — Primary-Signal Interaction Features (5 features) ★ critical

Most students treat the primary signal as just another column. It is not — it is the conditioning variable that defines the meta-labeling problem.

| Feature | Definition |
|---|---|
| `primary_signal` | Raw signal value {−1, 0, +1} |
| `signal_changed` | 1 if signal_t ≠ signal_{t−1}, else 0 |
| `signal_persistence` | Days since the last change in the primary signal |
| `signal_trend_concord` | sign(signal_t) · sign(ret_50d) — does primary signal agree with 50d trend? |
| `signal_density_20d` | Fraction of last 20 days with non-zero signal |

**Justification:** Meta-labeling is fundamentally about learning when a noisy primary signal is reliable. The primary signal's stability (`signal_persistence`), its agreement with longer-horizon trend (`signal_trend_concord`), and how active the primary model has recently been (`signal_density_20d`) are exactly the variables you want the metamodel to condition on.

### 1.10 Calendar Features (4 features, lightweight)

Cyclic encoding (sin/cos) of day-of-week and month-of-year. Cyclic rather than one-hot so distance-based models behave sensibly.

### 1.11 Pre-processing and leakage controls

- **Scaling:** RobustScaler (median, MAD) fitted **inside the training fold only** and applied to validation/test.
- **HMM/GMM:** fitted on the training fold only; the fitted model is used to produce posterior probabilities on validation/test.
- **Missing values:** forward-fill up to 2 days; otherwise drop the row.
- **No global standardisation, no PCA on the full sample, no rolling statistic that peeks at the future.** All windows are right-aligned.

### 1.12 Feature count summary

| Group | Count |
|---|---|
| A — Returns and momentum | 9 |
| B — Volatility estimators | 8 |
| C — Microstructure | 5 |
| D — Mean reversion / trend | 8 |
| E — Latent regime (HMM, GMM) | 6 |
| F — Spectral / fractal | 5 |
| G — Cross-asset / cross-sectional | 6 |
| H — Primary-signal interaction | 5 |
| Calendar | 4 |
| **Total** | **56** |

Rich, but not absurd — and every feature is justified by a hypothesis about what edge it captures. The cluster structure in Part 4 will compress these 56 features into ≈8 economically interpretable drivers.

---

## 2. Triple-Barrier Method (20 marks)

### 2.1 Meta-labeling, not classification

Restating the framing from §0 because it determines every implementation detail:

- We label only days with `primary_signal ≠ 0`.
- Barriers are evaluated **in the direction of the primary signal**:
  - If `s_t = +1`: the *profit-take* barrier is the upper barrier, the *stop-loss* barrier is the lower barrier.
  - If `s_t = −1`: the *profit-take* barrier is the lower barrier, the *stop-loss* barrier is the upper barrier.
- Label = **1** if the profit-take barrier is touched first.
- Label = **0** if the stop-loss barrier is touched first, or if the vertical (time) barrier is touched first.

This is the AFML §3.4 meta-labeling specification.

### 2.2 Barrier widths: volatility-scaled, not fixed

Markets in 2010 are not markets in 2024. Fixed-percent barriers over-trigger in high-vol regimes (mostly noise) and under-trigger in low-vol regimes (missing real moves). Volatility-scaled barriers are regime-equivariant — equivalent across regimes when measured in σ-units.

**Volatility estimator: Yang-Zhang, 50-day window.**

Two reasons:
1. **Efficiency.** YZ is the most efficient OHLC volatility estimator under realistic assumptions (it handles overnight returns, which Parkinson and Garman-Klass do not). With only ~250 trading days per year, efficiency materially reduces estimation noise in the barrier widths.
2. **Methodological consistency.** Group B already uses YZ as the canonical volatility feature, so the barriers are scaled by the same quantity the model sees.

Window length of 50 days is a balance — short enough to respond to regime change, long enough that the estimator is not itself noisy. I will sensitivity-test this at 30 and 80 days in Part 3.

### 2.3 Barrier multipliers: **asymmetric 2:1** (creative, justified)

The standard textbook choice is `pt_sl = [1, 1]` (symmetric ±1σ barriers). I deliberately depart from this:

```
pt_sl = [2.0, 1.0]   # profit-take = 2σ above entry, stop-loss = 1σ below
```

(Sign convention is "in the direction of the primary signal" — so for a short, the 2σ profit-take is *below* the entry price.)

**Why?**

1. **The economic question is the right one.** Symmetric barriers ask "does the primary signal predict direction?" Asymmetric 2:1 barriers ask "does the primary signal predict trades whose reward-risk profile is at least 2:1?" The second question is much closer to what the *strategy* (Part 5) needs to know.
2. **Class balance is informative, not trivial.** With symmetric barriers and a half-decent primary signal, the labels come out close to 50/50 — a classifier that calls everything "1" hits ~50% accuracy and the model has to work hard for marginal improvement. With 2:1 barriers, the base rate of positive labels falls to ≈30–40% under realistic primary signals, which means a model that uses the features genuinely has more room to add value above the trivial baseline.
3. **It's still the right loss surface for position sizing.** The downstream strategy (bonus Part 6) sizes positions by `metamodel_probability`. The probabilities are calibrated to "trade reaches +2σ before −1σ" — a useful, directly tradable quantity.

I will report results with `pt_sl = [1, 1]` as a sensitivity check.

### 2.4 Vertical barrier: 10 trading days

| Choice | Problem |
|---|---|
| 1–3 days | Most events resolve at the vertical barrier; the label collapses to "sign of next-week return" and the time horizon falls below the meaningful range of the daily features. |
| 5 days | Better, but still vertical-dominated for low-vol instruments. |
| **10 days** | Balanced: roughly the half-life of the daily momentum features; barriers and time horizon both bind for ≈70–80% of events. |
| 20+ days | Severe concurrent-label overlap, weak sample efficiency, label leakage into other events' barrier windows. |

10 days is the default. I sensitivity-test 5 and 15 days.

### 2.5 Barrier-touch detection: use intraday H/L

Critically, barrier touches are detected using the **daily High and daily Low**, not just the close. A stop-out that happened intraday must be detected even if the daily close recovered. In commodity futures (CL, NG especially) and equity index futures, the daily range routinely spans both barriers, and using only the close systematically over-labels positives.

This is a small detail that is wrong in many student implementations and easily costs marks.

### 2.6 Algorithm

For each event `t` with `primary_signal_t ≠ 0`:

1. `σ_t` ← Yang-Zhang vol at `t` (50d window).
2. `upper_t` ← `Close_t · exp(+2σ_t)`, `lower_t` ← `Close_t · exp(−1σ_t)` for a long; mirrored for a short.
3. `t1` ← `t + 10` trading days (vertical barrier).
4. For each day `u ∈ (t, t1]`:
   - If `High_u ≥ upper_t`: upper barrier touched at `u`, exit loop.
   - If `Low_u ≤ lower_t`: lower barrier touched at `u`, exit loop.
   - Edge case: if both touched on the same day, conservatively assign **lower** (stop-loss) — this is the AFML convention and reflects the worst-case assumption when intraday path is unknown.
5. If neither touched by `t1`: vertical barrier wins.
6. Translate first-touched barrier into binary meta-label using the direction-of-primary-signal logic in §2.1.

The output dataframe has columns: `t0` (event time), `t1` (label resolution time — needed for sample weighting in Part 3), `barrier_touched ∈ {upper, lower, vertical}`, `realized_return`, `meta_label ∈ {0, 1}`.

### 2.7 Sample weighting — forward reference to Part 3

The triple-barrier method produces **overlapping labels**: events `t` and `t+1` share most of their forward return windows. This violates the i.i.d. assumption that vanilla classifiers make about the loss. In Part 3, I will:

- Compute **average uniqueness** per event (AFML §4.5) — for each event, the average fraction of its barrier window that is *not* shared with other events' windows.
- Apply **uniqueness-weighted sample weights** in the loss function.
- Apply **time-decay weights** so recent events count more.
- Use **sequential bootstrap** (AFML §4.5.4) for the random forest, sampling events approximately proportional to their average uniqueness.

The labeling output carries the `t1` column precisely so these downstream operations are well-defined.

### 2.8 Diagnostics produced per instrument

The labeling pipeline outputs the following diagnostic summary, which I will report in the writeup:

| Diagnostic | What it tells me |
|---|---|
| % of events with `meta_label = 1` | Class balance — should be 30–45% under 2:1 asymmetry |
| Mean holding period (days from `t0` to barrier-touch) | Should be roughly half the vertical barrier; if much shorter, barriers too tight |
| Fraction resolved at upper / lower / vertical | If >50% vertical, barriers too wide; if <10% vertical, barriers too tight |
| Mean and median absolute realised return at touch | Sanity check: should be roughly 1.5σ given the 2:1 asymmetry |
| Average uniqueness over the sample | Should be > 0.3; if much lower, time horizon too long |

If any of these are pathological, the barriers are recalibrated and re-justified.

### 2.9 Out-of-sample carve-out (forward reference to Part 5)

Triple-barrier labels for the test set use only training-period volatility distribution to calibrate the barriers — i.e., the volatility estimator is computed on the test set, but no fitted statistics from the test set bleed into the train set. The test set is the final 20% of the sample, contiguous and chronologically after the training period. This is the *purged + embargoed* split from AFML §7, where the embargo period equals the maximum vertical-barrier horizon (10 days) to prevent any forward-leaking labels.

---

## 3. What's anticipated downstream

I have already structured Parts 1 and 2 in a way that pays dividends in Parts 3–5:

- **Feature groups (Group A–H)** map directly to the clusters used in Part 4's cluster-level feature importance.
- **`t1` column** is required by sample-weight and bootstrap routines in Part 3.
- **Yang-Zhang volatility** is consistent across the feature set and the barrier widths — no methodological mismatch between train labels and test-set inputs.
- **Primary-signal interaction features (Group H)** mean the metamodel can condition on signal stability, which is the cleanest source of meta-labeling edge.
- **Asymmetric 2:1 barriers** produce a calibrated probability output that the bonus Part 6 strategy can use directly for position sizing.

---

## 4. Code organisation

Two modules implement everything in this document:

- **`features.py`** — `FeatureBuilder` class with one method per feature group. The top-level `build_features(ohlcv_dict, primary_signals_dict)` function returns the full feature matrix for one or more instruments.
- **`labeling.py`** — `triple_barrier_meta_label(...)` is the main entry point. Helper functions for Yang-Zhang volatility, vertical-barrier computation, and the per-event touch detection are factored out.

Both modules are pure pandas / numpy / scipy / scikit-learn / hmmlearn. No heavy dependencies.
