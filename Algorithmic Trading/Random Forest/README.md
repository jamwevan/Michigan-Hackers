# Machine Learning Powered Trading Algorithm

This repository contains a **well-structured, step-by-step stock prediction system**, where each script handles a specific task in the workflow. The pipeline fetches raw data from an external API, formats it, extracts features, trains a predictive model (Random Forest), and uses that model to simulate future trading days. Additional tools are included for evaluating prediction performance, visualizing results, and backtesting hypothetical trades.

The model used for forecasting is a **Random Forest Regressor**, chosen for two main reasons:

- **Limited data**: Most stock histories in the dataset aren’t long enough to effectively train RNNs or LSTMs.  
- **Nonlinear relationships**: Random Forest is a tree-based ensemble method that handles nonlinear patterns well without assuming a specific functional form. This works well for financial data.

---

## 🏆 Competition Win

This model **won the Michigan Hackers Algorithmic Trading Competition**, producing a **+476.51% return over a 6-month trading window**.

---

## Tools

- **`data_pipeline.py`**  
  Downloads historical OHLCV data for each ticker listed in `tickers.txt` using the external API. It stores the data in the `stock_data/` directory. This is the first step. Run this before anything else.

- **`date_formatter.py`**  
  Standardizes all `Date` fields in the CSVs to `MM/DD/YYYY` format (e.g., `4/2/2023`) to prevent parsing issues later in the pipeline.

- **`feature_engineering.py`**  
  Adds technical indicators like SMA, EMA, RSI, MACD, and Bollinger Bands to each ticker’s historical price data. Outputs go to `processed_data/`.

- **`date_extracter.py`**  
  Extracts valid start and end date ranges per ticker from the processed data and stores them in `ticker_date_ranges.txt`. This helps set the training window in `model.py`.

- **`model.py`**  
  Trains a Random Forest on each stock’s feature set and uses it to generate forward predictions one day at a time. Results are saved in `predictions/`.

- **`percent_change.py`**  
  Calculates the percent change in predicted price for each stock over the forecasting window. Reports any missing predictions and ranks all stocks from worst to best.

- **`backtest.py`**  
  Simulates a simple long-only strategy. You give it a stock, date range, and balance, and it computes the number of shares bought, final portfolio value, and percent return.

- **`graph.py`** *(optional)*  
  Plots actual vs. predicted price paths for individual stocks. Helpful for visualization and debugging.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `requests`
  - `tqdm`
  - `os`
  - `time`
  - `logging`
  - `datetime`
  - `concurrent.futures`
  - `calendar`
  
### Installation

```bash
git clone https://github.com/your_username/stock-predictor.git
cd stock-predictor
pip install -r requirements.txt
```

---

## Usage

### Full pipeline:

```bash
# Step 1: Download raw stock data
python data_pipeline.py

# Step 2: Standardize date formats (e.g., 4/2/2023)
python date_formatter.py

# Step 3: Add technical indicators
python feature_engineering.py

# Step 4: Extract valid training date ranges
python date_extracter.py

# Step 5: Train model and generate predictions
python model.py

# Step 6: View top/bottom stocks by percent gain
python percent_change.py

# Step 7: Run a buy-and-hold backtest
python backtest.py

# Optional: Plot predicted vs. actual
python graph.py
```

---

## Contact

**James Evans**  
📧 jamwevan@umich.edu
