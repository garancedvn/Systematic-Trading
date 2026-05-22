Ideas for Systematic: 



**Group A Returns and Momentum: how had the price been moving ?** 

&#x20; - Log return (additive across time, roughly symmetric around 0, standard deviation log returns is the natural vol measure) -- Take them at different horizon (1d, 5d, 10d, 20d, 60d).

&#x20; - ret\_20D\_zscore 20-day return z-score against 252 days dist.   

&#x20; - mom\_12\_1: twelve-minus-one momentum factor: 252-day return excluding the most recent 21 days. 

&#x20; - roc\_10d: rate of change over 10 days almost same as ret\_10d but in simple-return 

&#x20; - mom\_3m\_minus\_1m: 63 return minus 21 day return. Moment acceleration: positive if the longer term trend is stronger than the recent trend (sign of persistance), negative if recent action is much hotter than the broader trend (sign of exhaustion). 



**Group B Volatility Estimators:** 

Basis: 

&#x20;  - Baseline: basic vol estimator (but ignore intraday range)

&#x20;  - Parkinson: Uses daily High and Low, but ignore the open and close not distinguish between trend up all day or ranged sideways

&#x20;  - Garman-Klauss Adds open and close to Parkinson 7xefficient assumes no drift and no overnight returns. Breaks when string drift 

&#x20;  - Rogers-Satchell: one drift-robust, work correctly even if price non-zero mean return. 

&#x20;  - Yang-Zhang minimum estimation variance under the most realistic assumptions handles overnight returns properly. 



Estimators: 

&#x20;  - vol\_of\_vol\_60d: 60-day standard dev of vol\_yz\_20d. How stable the vol regime is. High vol-of-vol = vol is itself bouncing around (regime change), low vol-of-vol = stable vol regime

&#x20;  - vol\_yz\_zscore\_252d: z-score YZ vol vs its one year dist. 

&#x20;  - hl\_range\_close rolling mean of (H-L)/C over 20 days. Simple proxy for intraday range as a fraction of price level. Less efficient than the proper estimators but different angle. 



**Group C: Microstructure Proxies:** 

&#x20;  - amihud\_illiq\_20d: avg absolute return per dollar of trading volume. High value = small dollar volumes are causing large returns = illiquid mkt. Low value = large dollar volumes barely move price = liquid mkt. 

&#x20;   - roll\_spread\_20d: If observe stock bouncing between bid and ask, consecutive price changes will be negatively autocorrelated, when trade at the ask, the net observed price is more likely to be at the bid and vice versa. Magnitude of autocorr tells you the spread. Cov>0, formulate has no sol, set feature to NaN. 

&#x20;    - volume\_zscore\_20d: curr day's volume z-score against its 20-day dist; High value = unusual volume today

&#x20;    - dollar\_volume\_log: ln(Ct\*Vt), liquidity proxy on its own deep mkts have high dollar volume. 

&#x20;    - kyle\_lambda\_20d: price-impact coef. Approx it as rolling regression slope |rt| on sqrt(Vt). In illiquid mkt, same volume produces a larger price move so lambda high. In liquid mkt lambda low. 



**Group D: Mean Reversion vs Trend** 

&#x20;    - rsi\_14: relative strength index bounded btw 0 and 100. Above 7P overbought (likely to reverse down), below 30 oversold (likely to reverse up). Use **EWM** avg instead simple avg. 

&#x20;    - bb\_position\_20\_2: +1 = price upper band (2 sigma above the 20 day MA), -1 at lower band. 

&#x20;    - macd\_signal, macd\_hist: Moving average convergence divergence. MACD line = **EMA(12)-EMA(26).** Signal line = EMA(9) of MACD . Histogram = MACD - Signal. Captures momentum at 1-2 month horizon. 

&#x20;    - adx\_14: avg directional Index: measured trend strength without telling direction. Values > 25 mean "trending mkt" (trend following works), <20 mean ranging mkt (mean-reversion works). Critical for metamodel: +1 primary signal is much mor reliable in a high-ADX env. 

&#x20;    - williams\_r\_14: -100 to 0. Captures overbought/oversold on a 14\_day window 

&#x20;    - dist\_ma200\_sigma: Distance from the 200-day MA expressed in sigma-units . 200-day MA canonical "long-term trend" line. This feature tells the model "we are currently 1.5σ above the long-term trend," which is a much richer signal than just "we are above the MA."

&#x20;     - cci\_20: commodity channel Index. for commodities for energy and metals 



**Group E: Latent Regime Features** 

Hidden states = bear (lowest mean state 0) , chop (middle mean state1), bull (highest mean state 2) , over return and vol 

&#x20;     - hmm\_p\_state0, hmm\_p\_state1, hmm\_p\_state2: posterior probas (being in each state) 

&#x20;     - hmm\_state\_persistence: number of consecutive days in the current **MAP** state 



GMM memoryless reassessed from scratch. 

&#x20;     - gmm\_logdensity: log proba of today's (return, vol) under fitted GMM. Low value means today's combination of return and vol is rare under the trained dist, anomaly score

&#x20;     - gmm\_argmax: hard cluster assignment from GMM 



Having both lets the downstream classifier use whichever signal is sharper in given period. 



**Group F: Spectral and Fractal Features (5 features)**

**Hurst exponent** 

H how range of a time series scales with the length of the window. 3 regimes: 

* H = 0.5: pure rdm walk (BM). Past doesn't predict future. 
* H > 0.5: trending behaviour. Past direction tends to continue. 
* H < 0.5: mean-reverting behaviour. Past directions tends to reverse. 



**Detrended Fluctuation Analysis**

More robust than Hurst to non-stationary 

1. Cumulatively sum the centred returns: Yk=∑i=1k(ri−rˉ)Y\_k = \\sum\_{i=1}^k (r\_i - \\bar{r})

Yk​=∑i=1k​(ri​−rˉ).

2\. For each window size s, divide Y into segments, fit a linear trend in each segment, compute the RMS of the residuals — call this F(s)

3\. Plot logF(s) vs log s — the slope is the DFA exponent α



**FFT-based features** 

The Fast Fourier Transform decomposes a time series into its frequency components — how much of the variance is at period 2 days, 4 days, 8 days, etc.



&#x20;     - dominant\_cycle\_period: period (in days) with highest power in spectrum. If dominant period is 10 days, series is showing rough 10-day cycles, mean-reversion strats can exploit

&#x20;     - spectral entropy: Shannon entropy of normalised power spectrum. Low entropy = power is concentrated at a few frequencies. High entropy = power is spread across all frequencies (like white noise)



**Group G: Cross-Asset / Cross-Sectional Features** 

&#x20;     - corr\_basket\_60d: rolling 60-day corr of instrument's returns with equal-weight basket of its peers. When drops, instruments is decoupling from its peers, which often signals idiosyncratic information dominating systemic factors.

&#x20;     - xs\_rank\_5d: percentile rank of instrument's 5-day return within its asset class. So 1.0 mean "best performer in the energy complex over the last week", 0.0 "worst". This is cross-sectional equivalent of RSI - overbought/oversold relative to peer, not to history

&#x20;     - xs\_dispersion\_20d: standard dev across the asset class 20-day returns. High dispersion = idiosyncratic regime (primary signals might work better), low dispersion = systemic regime (everything moves together, direction signals matter more than instrument selection) 

&#x20;      - leadlag\_anchor: corr with lagged anchor instrument. use CL as energy anchor because crude often leads heating oil and gasoline. If instrument highly corr with yesterday's CL move, strong info about today. 

&#x20;     - vol\_ratio\_basket: instrument YZ vol/ mean basket YZ vol > 1 = instrument is more volatile than its peers

&#x20;     - beta\_basket\_60d = rolling OLS beta of instrument's returns on basket returns. captures sensitivity to systematic moves 



The primary signal typically fails when the instrument is moving for idiosyncratic reasons that the primary model doesn't observe. Cross-sectional features are how the metamodel detects that situation.



**Group H: Primary-Signal Interaction Features:**

&#x20;     - primary\_signal: raw signal

&#x20;     - signal changed: 1 if signal changed from yesterday, 0 otherwise. A fresh signal is different from a persistent one. 

&#x20;     - signal\_persistence: days since last signal change. 

&#x20;     - signal\_trend\_concord: sign(signal) \* sign(50d return). +1 signal agrees with longer-term trend, -1 disagree. +1 signal in a 50d downtrend statistically less reliable.

&#x20;     - signal\_density\_20d: fraction of the last 20 days where primary signal was non-zero. Noisy primary that flips btw 0 and +-1 often is statistically diff from one that maintains positions.  



























