import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings("ignore")

def read_yfinance_csv(ticker):
    path = f"data/raw/{ticker}_prices.csv"
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["date","Open","High","Low","Close","Volume"]
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[["Open","High","Low","Close","Volume"]] = df[
        ["Open","High","Low","Close","Volume"]
    ].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["date","Close"]).reset_index(drop=True)
    return df

ticker = "AAPL"
print("=== STEP 1: Raw prices ===")
prices = read_yfinance_csv(ticker)
print(f"Shape: {prices.shape}")
print(prices.head(3))
print(prices.dtypes)

print("\n=== STEP 2: Sentiment ===")
sentiment = pd.read_csv(f"data/processed/{ticker}_daily_sentiment.csv")
sentiment["date"] = pd.to_datetime(sentiment["date"])
print(f"Shape: {sentiment.shape}")
print(sentiment.head(3))
print(f"Sentiment date range: {sentiment['date'].min()} to {sentiment['date'].max()}")
print(f"Prices date range:    {prices['date'].min()} to {prices['date'].max()}")

print("\n=== STEP 3: Merged ===")
merged = pd.merge(prices, sentiment, on="date", how="left")
merged = merged.sort_values("date").reset_index(drop=True)
sentiment_cols = ["avg_sentiment","max_sentiment","min_sentiment",
                  "sentiment_std","news_count","pos_ratio","neg_ratio","neu_ratio"]
merged[sentiment_cols] = merged[sentiment_cols].ffill().fillna(0)
print(f"Shape: {merged.shape}")
print(f"NaN count per column:\n{merged.isnull().sum()[merged.isnull().sum()>0]}")

print("\n=== STEP 4: After technical indicators ===")
close = merged["Close"]
merged["rsi"]      = ta.momentum.RSIIndicator(close, window=14).rsi()
merged["macd"]     = ta.trend.MACD(close).macd()
merged["ema_20"]   = ta.trend.EMAIndicator(close, window=20).ema_indicator()
merged["volume_ma7"] = merged["Volume"].rolling(7).mean()
print(f"NaN count after indicators:\n{merged.isnull().sum()[merged.isnull().sum()>0]}")
print(f"\nFirst non-NaN row index: {merged.dropna().index[0] if len(merged.dropna())>0 else 'ALL NaN'}")
print(f"Rows after dropna: {len(merged.dropna())}")

print("\n=== STEP 5: Sample of Close column ===")
print(merged[["date","Close","rsi","ema_20"]].head(30).to_string())