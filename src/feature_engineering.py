import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings("ignore")

# ── Robust CSV loader for yfinance output ─────────
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

# ── Load and merge with forward-fill ─────────────
def load_and_merge(ticker):
    prices = read_yfinance_csv(ticker)

    sentiment = pd.read_csv(f"data/processed/{ticker}_daily_sentiment.csv")
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    # Drop the ticker column from sentiment — causes 249 NaN after merge
    if "ticker" in sentiment.columns:
        sentiment = sentiment.drop(columns=["ticker"])

    # Left merge — keep ALL 251 trading days
    merged = pd.merge(prices, sentiment, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)

    sentiment_cols = ["avg_sentiment","max_sentiment","min_sentiment",
                      "sentiment_std","news_count","pos_ratio",
                      "neg_ratio","neu_ratio"]

    # Forward-fill: carry last known sentiment forward
    merged[sentiment_cols] = merged[sentiment_cols].ffill()
    # Fill remaining NaN at start with 0 (neutral — no news yet)
    merged[sentiment_cols] = merged[sentiment_cols].fillna(0)

    print(f"  Merged shape: {merged.shape}")
    print(f"  Days with real news : {(merged['news_count'] > 0).sum()}")
    print(f"  Days forward-filled : {(merged['news_count'] == 0).sum()}")
    return merged

# ── Technical indicators ──────────────────────────
def add_technical_indicators(df):
    close = df["Close"]
    df["rsi"]         = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["macd"]        = ta.trend.MACD(close).macd()
    df["macd_signal"] = ta.trend.MACD(close).macd_signal()
    df["bb_high"]     = ta.volatility.BollingerBands(close).bollinger_hband()
    df["bb_low"]      = ta.volatility.BollingerBands(close).bollinger_lband()
    df["bb_width"]    = ta.volatility.BollingerBands(close).bollinger_wband()
    df["ema_20"]      = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["atr"]         = ta.volatility.AverageTrueRange(
                            df["High"], df["Low"], close).average_true_range()
    df["volume_ma7"]  = df["Volume"].rolling(7).mean()
    df["volume_ratio"]= df["Volume"] / df["volume_ma7"]
    return df

# ── Lag + rolling features ────────────────────────
def add_lag_features(df):
    df["sentiment_lag1"]  = df["avg_sentiment"].shift(1)
    df["sentiment_lag2"]  = df["avg_sentiment"].shift(2)
    df["sentiment_lag3"]  = df["avg_sentiment"].shift(3)
    df["sentiment_roll3"] = df["avg_sentiment"].rolling(3).mean()
    df["sentiment_roll7"] = df["avg_sentiment"].rolling(7).mean()
    df["price_change_1d"] = df["Close"].pct_change(1)
    df["price_change_3d"] = df["Close"].pct_change(3)
    df["price_change_5d"] = df["Close"].pct_change(5)
    df["high_low_ratio"]  = df["High"] / df["Low"]
    return df

# ── Target variable ───────────────────────────────
def create_target(df):
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df

# ── Full pipeline per ticker ──────────────────────
def build_features(ticker):
    print(f"\nBuilding features for {ticker}...")
    df = load_and_merge(ticker)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = create_target(df)

    # Only drop rows where CORE features are NaN
    # (first ~25 rows while indicators warm up, last 1 row for target)
    feature_cols = ["rsi","macd","ema_20","atr","volume_ma7",
                    "sentiment_lag1","price_change_1d","target"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    df["ticker"] = ticker
    print(f"  Final shape: {df.shape[0]} rows x {df.shape[1]} cols")
    return df

# ── Run for all tickers ───────────────────────────
if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "MSFT"]
    all_data = []

    for ticker in tickers:
        df = build_features(ticker)
        df.to_csv(f"data/processed/{ticker}_features.csv", index=False)
        all_data.append(df)

    master = pd.concat(all_data, ignore_index=True)
    master.to_csv("data/processed/master_features.csv", index=False)

    print(f"\n{'='*45}")
    print(f"Master dataset  : {master.shape[0]} rows x {master.shape[1]} cols")
    print(f"Features        : {master.shape[1] - 3} (excl. date, ticker, target)")
    print(f"Target distribution:\n{master['target'].value_counts()}")
    print(f"{'='*45}")
    print("All feature files saved to data/processed/")