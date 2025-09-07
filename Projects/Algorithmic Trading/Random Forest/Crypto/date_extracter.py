import os
import pandas as pd

data_folder = "processed_data"
output_file = "coin_data_ranges.txt"
tickers_file = "coins.txt"

with open(tickers_file, "r") as f:
    tickers = [line.strip() for line in f if line.strip()]

with open(output_file, "w") as out:
    for ticker in tickers:
        filename = f"{ticker}_features.csv"
        path = os.path.join(data_folder, filename)
        try:
            df = pd.read_csv(path)
            date_col = next(
                (col for col in df.columns if col.strip().lower() in {"date", "timestamp", "datetime"}),
                None
            )

            if date_col is None:
                raise ValueError("No valid date column found")

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            start_date = df[date_col].min().strftime("%Y-%m-%d")
            end_date = df[date_col].max().strftime("%Y-%m-%d")

            # CHANGED THIS LINE ONLY
            out.write(f"{ticker},{start_date},{end_date}\n")

        except Exception as e:
            out.write(f"{ticker},Error,{e}\n")
