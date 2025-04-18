import pandas as pd
import numpy as np
import os

# Load raw stock data from CSV
def load_stock_data(ticker, folder="stock_data"):
    file_path = os.path.join(folder, f"{ticker}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    else:
        print(f"File not found: {file_path}")
        return None

# Read tickers from file
def read_tickers_from_file(filename="tickers.txt"):
    try:
        with open(filename, "r") as file:
            tickers = [line.strip() for line in file if line.strip()]
        return tickers
    except Exception as e:
        print(f"Error reading tickers from {filename}: {e}")
        return []

# ===== TREND INDICATORS =====
def compute_trend_indicators(df):
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    return df

# ===== MOMENTUM INDICATORS =====
def compute_momentum_indicators(df):
    gain = df["Close"].diff().apply(lambda x: max(x, 0)).rolling(14).mean()
    loss = df["Close"].diff().apply(lambda x: abs(min(x, 0))).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["Momentum_10"] = df["Close"].diff(periods=10)
    return df

# ===== VOLATILITY INDICATORS =====
def compute_volatility_indicators(df):
    rolling_std = df["Close"].rolling(20).std()
    rolling_mean = df["Close"].rolling(20).mean()
    df["Bollinger_Upper"] = rolling_mean + 2 * rolling_std
    df["Bollinger_Lower"] = rolling_mean - 2 * rolling_std
    df["ATR_14"] = df[["High", "Low", "Close"]].apply(
        lambda row: max(row["High"] - row["Low"], abs(row["High"] - row["Close"]), abs(row["Low"] - row["Close"])), axis=1
    ).rolling(window=14).mean()
    return df

# ===== VOLUME-BASED INDICATORS =====
def compute_volume_indicators(df):
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["OBV"] = (df["Volume"] * ((df["Close"].diff() > 0).astype(int) - (df["Close"].diff() < 0).astype(int))).cumsum()
    return df

# ===== LAGGED RETURNS =====
def compute_lagged_returns(df):
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_5"] = df["Close"].shift(5)
    df["Lag_10"] = df["Close"].shift(10)
    return df

# ===== COMBINE ALL FEATURES =====
def add_features(df):
    df = compute_trend_indicators(df)
    df = compute_momentum_indicators(df)
    df = compute_volatility_indicators(df)
    df = compute_volume_indicators(df)
    df = compute_lagged_returns(df)
    return df

# ===== ALIAS FOR MODEL PIPELINE =====
def recalc_indicators(df):
    return add_features(df)

# ===== SAVE ENGINEERED DATA =====
def save_transformed_data(df, ticker, folder="processed_data"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    df.to_csv(os.path.join(folder, f"{ticker}_features.csv"))
    print(f"Processed data saved for {ticker}")

# ===== PROCESS A SINGLE TICKER =====
def process_stock_data(ticker):
    df = load_stock_data(ticker)
    if df is not None:
        df = add_features(df)
        df = df.dropna()
        save_transformed_data(df, ticker)

# ===== MAIN DRIVER =====
def main():
    # EMPTY processed_data FOLDER FIRST
    output_folder = "processed_data"
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    tickers = read_tickers_from_file("tickers.txt")
    for ticker in tickers:
        print(f"Processing {ticker}...")
        process_stock_data(ticker)

if __name__ == "__main__":
    main()
