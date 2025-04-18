import pandas as pd
import matplotlib.pyplot as plt
import os

def standardize_columns(df):
    cols = {col.lower(): col for col in df.columns}
    if "date" in cols:
        df.rename(columns={cols["date"]: "date"}, inplace=True)
    if "close" in cols:
        df.rename(columns={cols["close"]: "Close"}, inplace=True)
    return df

def plot_ticker(ticker):
    pred_path = os.path.join("predictions", f"{ticker}_iterative_predictions.csv")
    actual_path = os.path.join("stock_data", f"{ticker}.csv")

    if not os.path.exists(pred_path):
        print(f"[SKIP] No prediction file for {ticker}")
        return
    if not os.path.exists(actual_path):
        print(f"[SKIP] No actual file for {ticker}")
        return

    pred_df = pd.read_csv(pred_path)
    actual_df = pd.read_csv(actual_path)

    pred_df = standardize_columns(pred_df)
    actual_df = standardize_columns(actual_df)

    if "date" not in pred_df.columns or "Close" not in pred_df.columns:
        print(f"[SKIP] Prediction file missing required columns for {ticker}")
        return
    if "date" not in actual_df.columns or "Close" not in actual_df.columns:
        print(f"[SKIP] Actual file missing required columns for {ticker}")
        return

    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
    actual_df["date"] = pd.to_datetime(actual_df["date"], errors="coerce")

    pred_df = pred_df.dropna(subset=["date", "Close"])
    actual_df = actual_df.dropna(subset=["date", "Close"])

    start_date = pred_df["date"].min()
    end_date = pred_df["date"].max()
    actual_df = actual_df[(actual_df["date"] >= start_date) & (actual_df["date"] <= end_date)]

    if actual_df.empty:
        print(f"[SKIP] No actual data in prediction window for {ticker}")
        return

    michigan_blue = "#003865"
    michigan_maize = "#FFB000"

    print(f"Plotting {ticker}")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(pred_df["date"], pred_df["Close"], color=michigan_blue, label="Predicted Close", marker="o")
    ax1.set_ylabel("Predicted Close Price", color=michigan_blue)
    ax1.tick_params(axis='y', labelcolor=michigan_blue)

    ax2 = ax1.twinx()
    ax2.plot(actual_df["date"], actual_df["Close"], color=michigan_maize, label="Actual Close", linestyle="--", marker="x")
    ax2.set_ylabel("Actual Close Price", color=michigan_maize)
    ax2.tick_params(axis='y', labelcolor=michigan_maize)

    plt.title(f"{ticker} — Raw Predicted vs. Actual Prices", fontsize=14)
    ax1.set_xlabel("Date")
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.grid(True)
    plt.show()

def main():
    with open("tickers.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    for ticker in tickers:
        print(f"--- {ticker} ---")
        plot_ticker(ticker)

if __name__ == "__main__":
    main()
