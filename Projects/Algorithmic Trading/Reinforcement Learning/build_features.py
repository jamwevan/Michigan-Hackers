# build_features.py
import pandas as pd, numpy as np
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "stock_data"
OUT_DIR  = ROOT / "env_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- helpers ---
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns: Date, Open, High, Low, Close, [Volume]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)

    logp = np.log(df["Close"])
    ret1 = logp.diff()
    ret5 = logp.diff(5)
    ret10 = logp.diff(10)

    # Trend / regime
    e10 = ema(df["Close"], 10)
    e20 = ema(df["Close"], 20)
    e50 = ema(df["Close"], 50)
    e200 = ema(df["Close"], 200)
    ema_spread_10_20 = (e10 - e20) / df["Close"]
    ema50_over_200 = e50 / (e200 + 1e-8) - 1.0
    ema200_slope = e200.pct_change()

    sma20 = df["Close"].rolling(20).mean()
    std20_r = df["Close"].pct_change().rolling(20).std()
    z_close_sma20 = (df["Close"] - sma20) / (std20_r + 1e-8)
    bb_perc = (df["Close"] - sma20) / (2 * std20_r + 1e-8)

    # Volatility / range
    vol10 = df["Close"].pct_change().rolling(10).std()
    vol20 = df["Close"].pct_change().rolling(20).std()
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean() / df["Close"]
    rv_20 = df["Close"].pct_change().rolling(20).std()

    # Momentum
    mom_63 = (logp - logp.shift(63))  # ~quarter

    # Volume
    if "Volume" in df:
        vol_chg1 = df["Volume"].pct_change()
        vol_z20 = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-8)
    else:
        vol_chg1 = pd.Series(index=df.index, dtype=float)
        vol_z20 = pd.Series(index=df.index, dtype=float)

    # Calendar
    dow = df["Date"].dt.weekday / 6.0
    month_sin = np.sin(2*np.pi*(df["Date"].dt.month/12.0))
    month_cos = np.cos(2*np.pi*(df["Date"].dt.month/12.0))

    # Reward target
    ret_fwd = df["Close"].pct_change().shift(-1)

    feats = pd.DataFrame({
        "ret1":ret1, "ret5":ret5, "ret10":ret10,
        "ema_spread_10_20":ema_spread_10_20, "ema50_over_200":ema50_over_200, "ema200_slope":ema200_slope,
        "z_close_sma20":z_close_sma20, "bb_perc":bb_perc,
        "vol10":vol10, "vol20":vol20, "atr14":atr14, "rv_20":rv_20,
        "mom_63":mom_63,
        "vol_chg1":vol_chg1, "vol_z20":vol_z20,
        "dow":dow, "month_sin":month_sin, "month_cos":month_cos,
        "ret_fwd":ret_fwd, "Date":df["Date"]
    }).dropna().reset_index(drop=True)
    return feats

# Load all tickers
rows = []
for csv in sorted(DATA_DIR.glob("*.csv")):
    try:
        f = load_one(csv)
        f["ticker"] = csv.stem.upper()
        rows.append(f)
    except Exception as e:
        print(f"SKIP {csv.name}: {e}")

if not rows:
    raise SystemExit("No valid CSVs in stock_data/")

df = pd.concat(rows, ignore_index=True)

# If benchmark/market exists (e.g., SPY.csv), create per-split market returns
mkt = None
for sym in ["SPY", "IVV", "VOO"]:
    p = DATA_DIR / f"{sym}.csv"
    if p.exists():
        mkt_raw = pd.read_csv(p)
        mkt_raw["Date"] = pd.to_datetime(mkt_raw["Date"], errors="coerce")
        mkt_raw = mkt_raw.sort_values("Date").dropna(subset=["Close"])
        mkt_raw["mkt_ret_fwd"] = mkt_raw["Close"].pct_change().shift(-1)
        mkt = mkt_raw[["Date", "mkt_ret_fwd"]]
        break

# Global date-aligned splits (70/15/15)
dates = df["Date"].sort_values().unique()
t1i = int(len(dates)*0.70); t2i = int(len(dates)*0.85)
date_train_max = pd.Timestamp(dates[t1i])
date_val_max   = pd.Timestamp(dates[t2i])

train = df[df["Date"] <= date_train_max]
val   = df[(df["Date"] > date_train_max) & (df["Date"] <= date_val_max)]
test  = df[df["Date"] > date_val_max]

feat_cols = [c for c in df.columns if c not in ["Date","ret_fwd","ticker"]]

# Train-only scaler
mu = train[feat_cols].mean().values
sd = train[feat_cols].std().values + 1e-8

def pack(split: pd.DataFrame):
    X = ((split[feat_cols].values - mu)/sd).astype(np.float32)
    r = split["ret_fwd"].values.astype(np.float32)
    d = split["Date"].astype("datetime64[ns]").values
    return X, r, d

X_tr, r_tr, d_tr = pack(train)
X_va, r_va, d_va = pack(val)
X_te, r_te, d_te = pack(test)

np.save(OUT_DIR/"X_train.npy", X_tr); np.save(OUT_DIR/"rets_train.npy", r_tr); np.save(OUT_DIR/"dates_train.npy", d_tr)
np.save(OUT_DIR/"X_val.npy",   X_va); np.save(OUT_DIR/"rets_val.npy",   r_va); np.save(OUT_DIR/"dates_val.npy",   d_va)
np.save(OUT_DIR/"X_test.npy",  X_te); np.save(OUT_DIR/"rets_test.npy",  r_te); np.save(OUT_DIR/"dates_test.npy",  d_te)

# Optional market arrays if benchmark present
if mkt is not None:
    def align_mkt(split_df):
        tmp = split_df.merge(mkt, on="Date", how="left").fillna({"mkt_ret_fwd": 0.0})
        return tmp["mkt_ret_fwd"].values.astype(np.float32)
    np.save(OUT_DIR/"rets_train_mkt.npy", align_mkt(train))
    np.save(OUT_DIR/"rets_val_mkt.npy",   align_mkt(val))
    np.save(OUT_DIR/"rets_test_mkt.npy",  align_mkt(test))

meta = {
    "feat_cols": feat_cols,
    "date_train_max": str(date_train_max.date()),
    "date_val_max": str(date_val_max.date()),
    "scaler_mu": mu.tolist(),
    "scaler_sd": sd.tolist(),
    "benchmark": "present" if mkt is not None else "absent",
}
(Path(OUT_DIR/"meta.json")).write_text(json.dumps(meta, indent=2))
print(f"Saved env_data/ with {len(feat_cols)} features. Benchmark:", meta["benchmark"])
