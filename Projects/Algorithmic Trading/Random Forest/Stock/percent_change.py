import pandas as pd
import os

def calculate_percent_change(ticker):
    pred_path = os.path.join("predictions", f"{ticker}_iterative_predictions.csv")

    if not os.path.exists(pred_path):
        return None, f"Missing prediction file for {ticker}"

    df = pd.read_csv(pred_path)

    date_col = next((col for col in df.columns if col.strip().lower() == "date"), None)
    if date_col is None or "Close" not in df.columns:
        return None, f"Missing required columns in {ticker}'s prediction file."

    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "Close"]).sort_values("date")

    if len(df) < 2:
        return None, f"Not enough data for {ticker}"

    start_price = df.iloc[0]["Close"]
    end_price = df.iloc[-1]["Close"]
    percent_change = ((end_price - start_price) / start_price) * 100

    return percent_change, None

def main():
    with open("tickers.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    valid_results = []
    errors = []

    for ticker in tickers:
        change, error = calculate_percent_change(ticker)
        if error:
            errors.append(error)
        else:
            valid_results.append((ticker, change))

    print("\nMissing or Invalid Prediction Files:")
    print("------------------------------------")
    for err in errors:
        print(err)

    print("\nValid Percent Changes (Sorted Low → High):")
    print("------------------------------------------")
    for ticker, change in sorted(valid_results, key=lambda x: x[1]):
        print(f"{ticker}: {change:+.2f}%")

if __name__ == "__main__":
    main()
