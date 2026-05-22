import pandas as pd 

ENERGY = ["cl1s", "ho1s", "rb1s", "ng1s"]
# Load the data
def load_panel(
        ohlcv_path='ohlcv_data.csv',
        signals_path='primary_signals.csv',
        tickers = None 
):
    ohlcv_raw = pd.read_csv(ohlcv_path)
    signals_raw = pd.read_csv(signals_path)

    #OHLCV PANEL
    ohlcv_raw["date"] = pd.to_datetime(ohlcv_raw["date"])
    ohlcv = ohlcv_raw.sort_values(["instrument", "date"],).reset_index(drop=True)

    panel = {}
    for tk, group in ohlcv.groupby("instrument"):
        df = (
            group 
            .drop(columns=["instrument"])
            .set_index("date")
            .sort_index()
        )
        panel[tk] = df

    #PRIMARY SIGNALS
    signals_raw["date"] = pd.to_datetime(signals_raw["date"])
    signals_wide = signals_raw.set_index("date").sort_index()

    primary_signals = {}
    for tk in list(primary_signals.keys()):
        if tk in panel:
            continue 
        ohlcv_idx = panel[tk].index
        aligned = primary_signals[tk].reindex(ohlcv_idx).dropna().astype(int)
        primary_signals[tk] = aligned
    
    return panel, primary_signals

if __name__ == "__main__":
    panel, primary_signals = load_panel()
    print("Energy panel loaded:")
    for tk in panel:
        df = panel[tk]
        s = primary_signals[tk]
        print(f"  {tk}: OHLCV {df.index.min().date()}–{df.index.max().date()} "
              f"({len(df)} rows) | signal {len(s)} days, "
              f"counts {s.value_counts().sort_index().to_dict()}")