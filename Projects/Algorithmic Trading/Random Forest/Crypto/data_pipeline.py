import os
import time
import requests
import logging
import pandas as pd

API_KEY = "VFS9AYZT18LFWluVpSKnPYxtxzJlWL_K"

def read_tickers_from_file(filename):
    try:
        with open(filename, "r") as file:
            tickers = [line.strip().upper() for line in file if line.strip()]
        return tickers
    except Exception as e:
        print(f"Error reading tickers from {filename}: {e}")
        return []

def fetch_polygon_crypto_data(ticker, output_folder):
    print(f"Fetching crypto data for {ticker}...")

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/X:{ticker}-USD/range/1/day/"
        f"2017-01-01/{pd.Timestamp.today().strftime('%Y-%m-%d')}?adjusted=true&sort=asc"
        f"&limit=50000&apiKey={API_KEY}"
    )

    try:
        response = requests.get(url)
        if response.status_code == 429:
            print(f"Rate limit exceeded for {ticker}. Pausing for 60 seconds.")
            time.sleep(60)
            return fetch_polygon_crypto_data(ticker, output_folder)

        if response.status_code != 200:
            print(f"Failed to fetch {ticker}: {response.status_code} {response.text}")
            logging.warning(f"{ticker}: {response.status_code} {response.text}")
            return

        data = response.json()
        if "results" not in data or not data["results"]:
            print(f"No data for {ticker}")
            return

        df = pd.DataFrame(data["results"])
        df.rename(columns={
            "t": "Date", "o": "Open", "h": "High",
            "l": "Low", "c": "Close", "v": "Volume"
        }, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df.set_index("Date", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        output_file = os.path.join(output_folder, f"{ticker}.csv")
        df.to_csv(output_file)
        print(f"Saved to {output_file}")
        time.sleep(15)  # throttle to stay under 5 requests/minute

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        logging.warning(f"{ticker} exception: {e}")

def fetch_crypto_data(tickers, output_folder="crypto_data"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for ticker in tickers:
        fetch_polygon_crypto_data(ticker, output_folder)

if __name__ == "__main__":
    logging.basicConfig(
        filename="failed_crypto_tickers.log",
        level=logging.WARNING,
        format="%(asctime)s - %(message)s"
    )

    ticker_file = "coins.txt"
    tickers = read_tickers_from_file(ticker_file)

    if tickers:
        fetch_crypto_data(tickers)
    else:
        print("No valid tickers found in the file.")
