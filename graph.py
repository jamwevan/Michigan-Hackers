import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_ticker(ticker):
    pred_path = os.path.join("predictions", f"{ticker}_iterative_predictions.csv")
    actual_path = os.path.join("stock_data", f"{ticker}.csv")

    if not os.path.exists(pred_path) or not os.path.exists(actual_path):
        print(f"Missing file(s) for {ticker}")
        return

    pred_df = pd.read_csv(pred_path)
    actual_df = pd.read_csv(actual_path)

    if "date" not in pred_df.columns or "Close" not in pred_df.columns:
        print(f"Missing 'date' or 'Close' in prediction file for {ticker}")
        return
    if "date" not in actual_df.columns or "Close" not in actual_df.columns:
        print(f"Missing 'date' or 'Close' in actual file for {ticker}")
        return

    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
    actual_df["date"] = pd.to_datetime(actual_df["date"], errors="coerce")

    pred_df = pred_df.dropna(subset=["date"]).sort_values("date").tail(100)
    actual_df = actual_df.dropna(subset=["date"]).sort_values("date")

    pred_df = pred_df.rename(columns={"Close": "Close_predicted"})
    actual_df = actual_df.rename(columns={"Close": "Close_actual"})

    merged = pd.merge(
        pred_df[["date", "Close_predicted"]],
        actual_df[["date", "Close_actual"]],
        on="date", how="left"
    )

    print(f"Plotting {ticker} — {len(merged)} prediction points.")

    plt.figure(figsize=(10, 5))
    plt.plot(merged["date"], merged["Close_predicted"], label="Predicted Close", marker="o")

    if merged["Close_actual"].notnull().any():
        plt.plot(merged["date"], merged["Close_actual"], label="Actual Close", marker="x", linestyle="--")
    else:
        print(f"No actual close prices matched for {ticker}, but still plotting predictions.")

    plt.title(f"{ticker} — Actual vs. Predicted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    with open("tickers.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    for ticker in tickers:
        print(f"--- {ticker} ---")
        plot_ticker(ticker)

if __name__ == "__main__":
    main()
