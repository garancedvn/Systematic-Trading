# Systematic Trading Strategies with Machine Learning
### Coursework Project — Meta-Labeling for Trading Signal Filtering
**Group 13:** Agustina Albez, Garance Danvin, Helene Rabain, Kevin Aoun and Nathan Sebbag

---

## Repository Structure

```
coursework_revision/
│
├── README.md
├── coursework.ipynb
│
├── ohlcv_data.csv                  (given)
├── primary_signals.csv             (given)
│
├── energy_external_h2_2022.csv     (created)
├── macro_external_h2_2022.csv      (created)
├── fred_external_h2_2022.csv       (created)
├── metamodel_predictions.csv       (created)
└── strategy_weights.csv            (created)
```

---

## Files

### `coursework.ipynb`
The main project notebook. It implements a full meta-labeling pipeline on top of a primary trading signal for energy futures contracts. The notebook is organized into seven phases:

- **Phase 1 — Data Preparation:** loads and cleans OHLCV and signal data, filters the universe to the four energy instruments (WTI Crude Oil, Heating Oil, RBOB Gasoline, Natural Gas), and builds log returns.
- **Phase 2 — Feature Engineering:** constructs ~166 features across 18 semantic families, including momentum/volatility indicators, microstructure measures, cross-asset correlations, macro context (VIX, DXY, yield curve, credit spreads), inter-energy spreads, and Hidden Markov Model (HMM) regime features.
- **Phase 3 — Triple-Barrier Labeling:** labels each active primary signal as profitable (1) or not (0) under a triple-barrier framework with parameters PT = 1.5σ, SL = 0.75σ, horizon = 10 days, selected via a sensitivity grid.
- **Phase 4 — Model Development:** trains and tunes three model families (L2 Logistic Regression, LightGBM, and a PyTorch MLP) using Combinatorial Purged Cross-Validation (CPCV) and Bayesian optimization (Optuna TPE).
- **Phase 5 — Feature Importance & Ensembling:** measures cluster-level importance (MDI, MDA, SHAP), ensembles the three model families with AUC-weighted averaging, calibrates probabilities, and evaluates whether the meta-filter beats the raw primary signal.
- **Phase 6 — Out-of-Sample Evaluation:** final evaluation on the held-out test set (last 20% of signal dates).
- **Phase 7 — Position Sizing (optional):** builds a position-sizing strategy from the metamodel's calibrated probabilities.

---

### `ohlcv_data.csv` *(given)*
The raw market data provided for the coursework. Contains daily OHLCV (Open, High, Low, Close, Volume) and open interest for **11 futures contracts** across three asset classes:

| Asset class | Instruments |
|---|---|
| **Equity index futures** | ES1S (S&P 500), NQ1S (Nasdaq 100), FESX1S (Euro Stoxx 50) |
| **Energy futures** | CL1S (WTI Crude Oil), HO1S (Heating Oil), RB1S (RBOB Gasoline), NG1S (Natural Gas) |
| **Metals futures** | GC1S (Gold), SI1S (Silver), HG1S (Copper), PL1S (Platinum) |

**Key columns:** `date`, `instrument`, `open`, `high`, `low`, `close`, `volume`, `open_interest`

The notebook filters this file down to the four energy instruments for model training. The equity and metals instruments are used only as cross-asset context features.

---

### `primary_signals.csv` *(given)*
The primary trading signal provided by the course instructors. Contains daily directional signals for the four energy instruments in wide format.

**Key columns:** `date`, `cl1s`, `ho1s`, `rb1s`, `ng1s`

Signal values:
- `+1` — long signal
- `-1` — short signal
- `0` — no position

The notebook reshapes this file into long format (`date`, `instrument`, `primary_signal`) and uses it as the input to the triple-barrier labeling step. The metamodel does **not** re-predict direction; it estimates the probability that a given primary signal will be profitable.

---

### `energy_external_h2_2022.csv` *(created)*
Energy-specific external data fetched from Yahoo Finance, covering the second half of 2022 (the out-of-sample test period). Used in Phase 2.14 to construct energy market context features that cannot be derived from the OHLCV file alone.

**Key columns:** `date`, `ovx` (CBOE crude oil implied volatility), `brent` (Brent front future), `wti_yf` (WTI front future from Yahoo), `uso` (front-month crude ETF), `usl` (12-month-laddered crude ETF), `xle` (energy equity ETF), `xop` (E&P ETF), `oih` (oil services ETF), `usdcad` (USD/CAD), `usdnok` (USD/NOK)

These series feed into features including the OVX/VIX ratio, Brent–WTI spread, crude term-structure proxy (USO/USL), XLE beta, petro-currency correlations, and oil-services momentum (`oih_mom_20d`).

---

### `macro_external_h2_2022.csv` *(created)*
Global macro data fetched from Yahoo Finance, covering the second half of 2022. Used in Phase 2.14 to construct the macro risk environment features.

**Key columns:** `date`, `dxy` (US Dollar Index), `vix` (CBOE VIX), `us10y` (US 10-year Treasury yield)

These series feed into features including `dxy_ret_5d`, `dxy_ret_20d`, `dxy_corr_60d`, `vix_level`, `vix_change_5d`, and `us10y_change_5d`.

---

### `fred_external_h2_2022.csv` *(created)*
US rates, credit, and inflation data fetched from the FRED API (St. Louis Fed), covering the second half of 2022. Used in Phase 2.14 to construct the rates & yield curve and credit & inflation feature clusters.

**Key columns:** `date`, `dgs2` (2-year Treasury yield), `dgs10` (10-year Treasury yield), `dgs3mo` (3-month Treasury yield), `aaa` (Moody's Aaa corporate yield), `baa` (Moody's Baa corporate yield), `breakeven10` (10-year breakeven inflation rate), `real_yld10` (10-year real yield)

These series feed into features including `curve_10y2y`, `curve_10y3m`, `quality_spread_baa_aaa`, `breakeven10_z`, and `real_yld10_z`.

---

## Output Files Created by the Notebook

The notebook produces several intermediate and final datasets during execution. The most important ones are described below.

### `meta_labeled_df` *(in-memory, not saved to disk by default)*
The fully labeled dataset used for model training. One row per active primary signal (`+1` or `-1`), with all engineered features and the triple-barrier meta-label attached.

**Key columns added:**
- `meta_label` — binary target: 1 if the trade was profitable under the triple-barrier rule, 0 otherwise.
- `tb_event_type` — how the trade resolved: `profit_taking`, `stop_loss`, or `vertical_barrier`.
- `tb_exit_date`, `tb_exit_price`, `tb_trade_return` — trade exit information (excluded from model features to prevent leakage).

### `metamodel_predictions.csv` *(created)*
The out-of-sample predictions produced by the final weighted ensemble (Logistic + LightGBM + MLP) on the test set. One row per active primary signal in the test period, covering H2 2022.

**Key columns:** `date`, `instrument`, `primary_signal`, `meta_label` (true triple-barrier label), `proba_ensemble` (calibrated probability that the trade is profitable), `pred_ensemble` (binary prediction at the chosen decision threshold), `proba_logreg`, `proba_lgb`, `proba_mlp` (individual model probabilities)

Used in Phase 6 for the final out-of-sample evaluation (AUC, F1, confusion matrix) and in Phase 7 as the input to position sizing.

---

### `strategy_weights.csv` *(created)*
The daily position sizes derived from the metamodel's calibrated probabilities, produced in Phase 7 (optional position-sizing track). One row per active primary signal in the test period.

**Key columns:** `date`, `instrument`, `primary_signal`, `proba_ensemble`, `position_weight` (scaled position size, derived from the metamodel probability above the decision threshold), `filtered_signal` (final signal after meta-filter: 0 if the metamodel rejects the trade, otherwise equal to `primary_signal`)

These weights are used to compute the filtered strategy's Sharpe ratio and compare it against the unfiltered primary signal.

---
- `best_logreg` — L2-regularized logistic regression (best C selected via CPCV + Optuna)
- `best_lgb` — LightGBM gradient boosting (best hyperparameters selected via CPCV + Optuna)
- `best_mlp` — PyTorch MLP (best architecture and regularization selected via CPCV + Optuna)

These can be serialized with `joblib.dump(model, "model_name.pkl")` if persistence is needed.

---

## Data Flow Summary

```
ohlcv_data.csv    primary_signals.csv    energy_external_h2_2022.csv
      │                   │              macro_external_h2_2022.csv
      │                   │              fred_external_h2_2022.csv
      └──────────┬─────────┘                      │
                 │                                │
         Phase 1 — Data Preparation               │
                 │                                │
         Phase 2 — Feature Engineering ───────────┘
                 │
         Phase 3 — Triple-Barrier Labeling
                 │ (filters to active signals only)
         meta_labeled_df
                 │
         Phase 4 — Model Training (LR / LightGBM / MLP)
                 │
         Phase 5 — Ensembling & Feature Importance
                 │
         Phase 6 — Out-of-Sample Evaluation ──────► metamodel_predictions.csv
                 │
         Phase 7 — Position Sizing ────────────────► strategy_weights.csv
```

---

## Notes

- All models are trained on data up to `global_train_end_date` (the last date covering 80% of primary signal dates) and evaluated on the remaining 20%.
- External data (VIX, DXY, OVX, yield curve, credit spreads) is fetched at runtime from Yahoo Finance and FRED. An internet connection and a FRED API key are required for Phase 2.14.
- The notebook is designed to be run sequentially from top to bottom.
