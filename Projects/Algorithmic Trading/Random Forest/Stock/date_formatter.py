import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Path to your stock data directory
data_folder = "stock_data"

def process_csv(file_path):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)

        # Identify the date column
        date_col = next((col for col in df.columns if col.strip().lower() == "date"), None)

        if date_col:
            # Convert to datetime with strict parsing from ambiguous formats
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False)
            df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')

            df.to_csv(file_path, index=False)
            return f"Updated {filename}"
        else:
            return f"No date column found in {filename}, skipping."

    except Exception as e:
        return f"Error processing {filename}: {e}"

def main():
    csv_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith(".csv")
    ]

    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_csv, path): path for path in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            results.append(future.result())

    for line in results:
        print(line)

if __name__ == "__main__":
    main()
