import os
import pandas as pd

def calculate_max_percent_change(ticker):
    pred_path = os.path.join("predictions", f"{ticker}_iterative_predictions.csv")

    if not os.path.exists(pred_path):
        return None, f"Missing prediction file for {ticker}"

    df = pd.read_csv(pred_path)

    # Identify and standardize the date column
    date_col = next((col for col in df.columns if col.strip().lower() == "date"), None)
    if date_col is None or "Close" not in df.columns:
        return None, f"Missing required columns in {ticker}'s prediction file."

    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "Close"]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return None, f"Not enough data for {ticker}"

    start_price = df.iloc[0]["Close"]
    max_change = float("-inf")
    max_day = -1

    # Loop through each possible sell day
    for i in range(1, len(df)):
        pct_change = ((df.iloc[i]["Close"] - start_price) / start_price) * 100
        if pct_change > max_change:
            max_change = pct_change
            max_day = i  # zero-based index for df lookup

    best_date = df.loc[max_day, "date"].strftime("%Y-%m-%d")
    return max_change, best_date

def main():
    tickers_file = "coins.txt"
    if not os.path.exists(tickers_file):
        print("Missing coins.txt file.")
        return

    with open(tickers_file, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    results = []
    for ticker in tickers:
        change, detail = calculate_max_percent_change(ticker)
        if change is None:
            print(f"[SKIP] {ticker}: {detail}")
            continue
        results.append((ticker, change, detail))

    # Sort by percent change in ascending order
    results.sort(key=lambda x: x[1])  # ← changed from reverse=True to ascending

    print("\nMaximum Percent Changes:")
    for ticker, change, date in results:
        print(f"{ticker}: {change:.2f}% (best sell date: {date})")

if __name__ == "__main__":
    main()
