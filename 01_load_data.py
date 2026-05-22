import pandas as pd 
import numpy as np

# Load the data
ohlcv_raw = pd.read_csv('ohlcv_data.csv')
signals_raw = pd.read_csv('primary_signals.csv')

ohlcv_raw["date"] = pd.to_datetime(ohlcv_raw["date"])
ohlcv = ohlcv_raw.sort_values(["instrument", "date"],).reset_index(drop=True)

print("OHLCV AFTER PARSING.")
print(ohlcv.dtypes)
print()

panel = {}
for tk, group in ohlcv.groupby("instrument"):
    df = (
        group 
        .drop(columns=["instrument"])
        .set_index("date")
        .sort_index()
    )
    panel[tk] = df

print("Instruments in panel:", sorted(panel.keys()))
print(f"Number of instruments: {len(panel)}")
print()
for tk in sorted(panel.keys()):
    df = panel[tk]
    print(f"  {tk}: {len(df):>6} rows  |  {df.index.min().date()} to {df.index.max().date()}")
print()
print("Columns in each panel DataFrame:", panel["cl1s"].columns.tolist())
print()
print("Sample — cl1s first 3 rows:")
print(panel["cl1s"].head(3))


#PRIMARY SIGNALS
signals_raw["date"] = pd.to_datetime(signals_raw["date"])
signals_wide = signals_raw.set_index("date").sort_index()

primary_signals = {}
for tk in signals_wide.columns:
    primary_signals[tk] = signals_wide[tk].astype(int)

print("Signal tickers:", sorted(primary_signals.keys()))
print()
for tk in sorted(primary_signals.keys()):
    s=primary_signals[tk]
    counts=s.value_counts().sort_index().to_dict()
    print(f" {tk}: {len(s):>4} days | values: {counts}")

# DATE ALIGNMENT
tk = "cl1s"
ohlcv_dates = set(panel[tk].index)
signal_dates = set(primary_signals[tk].index)

both = signal_dates & ohlcv_dates
only_signal = signal_dates - ohlcv_dates
only_ohlcv_in_signal_period = (
    ohlcv_dates - signal_dates
) & set(pd.date_range(min(signal_dates), max(signal_dates)))

print(f"Alignment for {tk}:")
print(f"Signal dates that exist in OHLCV: {len(both)} / {len(signal_dates)}")
print(f"Signal dates missing in OHLCV: {len(only_signal)}")
print(f"OHLCV dates in signal period but missing in signals: {len(only_ohlcv_in_signal_period)}")

if only_signal:
    print(f"  Examples of misaligned signal dates: {sorted(only_signal)[:5]}")

# Drop dates that are not in OHLCV for each signal

for tk in primary_signals:
    if tk not in panel:
        continue
    ohlcv_idx = panel[tk].index
    signal_series = primary_signals[tk]
    
    # Keep only signal dates that are also trading days
    aligned = signal_series.reindex(ohlcv_idx).dropna().astype(int)
    primary_signals[tk] = aligned

# Verify
print("After alignment:")
for tk in sorted(primary_signals.keys()):
    s = primary_signals[tk]
    counts = s.value_counts().sort_index().to_dict()
    print(f"  {tk}: {len(s):>4} days | values: {counts}")

# Restrict only to ENERGY 
ENERGY = ["cl1s", "ho1s", "rb1s", "ng1s"]

panel = {tk: panel[tk] for tk in ENERGY}
primary_signals = {tk: primary_signals[tk] for tk in ENERGY}

print("Energy panel locked in:")
for tk in ENERGY:
    df = panel[tk]
    s = primary_signals[tk]
    print(f"  {tk}: OHLCV {df.index.min().date()}–{df.index.max().date()} ({len(df)} rows)"
          f" | signal {s.index.min().date()}–{s.index.max().date()} ({len(s)} days)")