import os
import pandas as pd

def calculate_trade_range(ticker):
    pred_path = os.path.join("predictions", f"{ticker}_iterative_predictions.csv")

    if not os.path.exists(pred_path):
        return None, None, f"Missing prediction file for {ticker}"

    try:
        df = pd.read_csv(pred_path)
    except Exception as e:
        return None, None, f"Error reading {ticker} file: {e}"

    # Identify and standardize the date column
    date_col = next((col for col in df.columns if col.strip().lower() == "date"), None)
    if date_col is None or "Close" not in df.columns:
        return None, None, f"Missing required columns in {ticker}'s prediction file."

    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "Close"]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return None, None, f"Not enough data for {ticker}"

    # Find first day where price increases the next day (buy point)
    buy_index = None
    for i in range(len(df) - 1):
        if df.loc[i + 1, "Close"] > df.loc[i, "Close"]:
            buy_index = i
            break

    if buy_index is None:
        return None, None, f"No upward trend found for {ticker}"

    buy_price = df.loc[buy_index, "Close"]
    buy_date = df.loc[buy_index, "date"]

    # Look for best sell date after buy date
    max_change = float("-inf")
    sell_index = -1
    for i in range(buy_index + 1, len(df)):
        pct_change = ((df.loc[i, "Close"] - buy_price) / buy_price) * 100
        if pct_change > max_change:
            max_change = pct_change
            sell_index = i

    if sell_index == -1:
        return None, None, f"No valid sell date found after buy date for {ticker}"

    sell_date = df.loc[sell_index, "date"]
    sell_price = df.loc[sell_index, "Close"]

    return max_change, sell_price, (buy_date.strftime("%Y-%m-%d"), sell_date.strftime("%Y-%m-%d"))

def main():
    tickers_file = "coins.txt"
    if not os.path.exists(tickers_file):
        print("Missing coins.txt file.")
        return

    with open(tickers_file, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    results = []
    for ticker in tickers:
        change, price, date_info = calculate_trade_range(ticker)
        if change is None:
            print(f"[SKIP] {ticker}: {date_info}")
            continue
        buy_date, sell_date = date_info
        results.append((ticker, change, price, buy_date, sell_date))

    results.sort(key=lambda x: x[1])  # Sort by percent increase

    print("\nMaximum Percent Changes:")
    for ticker, change, price, buy_date, sell_date in results:
        print(f"{ticker}: {change:.2f}%, {price:.2f}, [{buy_date}, {sell_date}]")

if __name__ == "__main__":
    main()
