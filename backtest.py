import pandas as pd
from datetime import datetime
import os

def load_date_ranges(path="ticker_date_ranges.txt"):
    ranges = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                ticker, start, end = parts
                ranges[ticker.strip().upper()] = (start.strip(), end.strip())
    return ranges

def backtest(ticker, start_date, end_date, starting_balance):
    path = os.path.join("stock_data", f"{ticker}.csv")
    if not os.path.exists(path):
        print(f"Data for {ticker} not found.")
        return

    df = pd.read_csv(path)

    # Detect and standardize 'date' column
    date_col = next((col for col in df.columns if col.strip().lower() == "date"), None)
    if date_col is None or "Close" not in df.columns:
        print(f"{ticker} file is missing 'Date' or 'Close' column.")
        return

    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "Close"])
    df = df.sort_values("date")

    df_range = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    if df_range.empty:
        print("No data available in the given date range.")
        return

    buy_row = df_range.iloc[0]
    sell_row = df_range.iloc[-1]

    buy_price = buy_row["Close"]
    sell_price = sell_row["Close"]

    shares = starting_balance / buy_price
    final_value = shares * sell_price
    profit = final_value - starting_balance
    pct_change = (profit / starting_balance) * 100

    print(f"\nBacktest for {ticker} from {buy_row['date'].date()} to {sell_row['date'].date()}:")
    print(f"  Buy Price: ${buy_price:.2f}")
    print(f"  Sell Price: ${sell_price:.2f}")
    print(f"  Shares Bought: {shares:.4f}")
    print(f"  Final Value: ${final_value:.2f}")
    print(f"  Profit/Loss: ${profit:.2f} ({pct_change:+.2f}%)\n")

def main():
    date_ranges = load_date_ranges()

    ticker = input("Enter stock ticker: ").strip().upper()

    if ticker not in date_ranges:
        print(f"{ticker} not found in ticker_date_ranges.txt")
        return

    valid_start, valid_end = date_ranges[ticker]
    print(f"Available data range for {ticker}: {valid_start} to {valid_end}")

    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    starting_balance = float(input("Enter starting balance: ").strip())

    backtest(ticker, start_date, end_date, starting_balance)

if __name__ == "__main__":
    main()
