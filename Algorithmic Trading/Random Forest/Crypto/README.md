# 🧠 Machine Learning Powered Crypto Trading Algorithm

This repository contains a modular, step-by-step cryptocurrency forecasting system, where each script handles a specific stage in the pipeline. The process fetches raw OHLCV data for selected cryptocurrencies, applies technical analysis, trains a predictive model (Random Forest), and simulates forward trading days to forecast future price movement. Additional tools allow for percent change ranking, performance evaluation, and historical backtesting.

The forecasting model is a Random Forest Regressor, selected due to:  
• Limited data availability: Many free crypto APIs impose data limits that make it impractical to train deep learning models like LSTMs or RNNs.  
• Nonlinear relationships: Tree-based models like Random Forests excel in capturing non-linear interactions in market data without requiring strong parametric assumptions.

---

## 🔧 Tools

• `data_pipeline.py`  
Downloads historical OHLCV data for each coin listed in `coins.txt`. Raw data is saved to the `crypto_data/` directory. Run this first.

• `feature_engineering.py`  
Enhances each coin’s price history with technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands). Outputs are stored in `processed_data/`.

• `date_extracter.py`  
Extracts the start and end date range for each coin from the processed features and saves them in `coin_data_ranges.txt`.

• `model.py`  
Trains a Random Forest model for each coin using historical indicators and generates future price predictions one day at a time. Results go to `predictions/`.

• `percent_change.py`  
Computes percent change in predicted price for each coin, identifies missing prediction files, and ranks all cryptocurrencies from worst to best.

• `backtest.py`  
Simulates a basic long-only backtest. You input a coin, date range, and initial balance, and it reports the final value and return had you bought and held.

---

## ⚙️ Usage

```bash
# Step 1: Fetch raw crypto OHLCV data
python data_pipeline.py

# Step 2: Compute technical indicators
python feature_engineering.py

# Step 3: Extract valid date ranges
python date_extracter.py

# Step 4: Train models and generate future price predictions
python model.py

# Step 5: Rank coins by predicted percent gain/loss
python percent_change.py

# Step 6: Simulate buy-and-hold backtest
python backtest.py
```

---

## 📬 Contact

**James Evans**📧 [jamwevan@umich.edu](mailto\:jamwevan@umich.edu)
