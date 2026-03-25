import yfinance as yf
import pandas as pd

# Test fetching Apple stock data
df = yf.download("AAPL", period="3mo")
print("Shape:", df.shape)
print(df.tail())
print("\nColumns:", df.columns.tolist())

# Save to raw data folder
df.to_csv("data/raw/AAPL_prices.csv")
print("\nSaved to data/raw/AAPL_prices.csv")