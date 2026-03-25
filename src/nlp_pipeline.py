from transformers import pipeline
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

print("Loading FinBERT model... (first run downloads ~400MB, be patient)")
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    return_all_scores=True,
    device=-1  # use CPU; change to 0 if you have GPU
)
print("FinBERT loaded successfully!")

# ── Score a single headline ───────────────────────
def get_sentiment(headline):
    try:
        headline = str(headline).strip()
        if len(headline) < 5:
            return 0.33, 0.33, 0.33
        result = finbert(headline[:512])[0]
        scores = {r["label"]: r["score"] for r in result}
        pos = scores.get("positive", 0)
        neg = scores.get("negative", 0)
        neu = scores.get("neutral",  0)
        return pos, neg, neu
    except Exception as e:
        print(f"  Warning: {e}")
        return 0, 0, 1
    
# ── Score all headlines in a news CSV ────────────
def score_news_file(ticker):
    path = f"data/raw/{ticker}_news.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    print(f"\nScoring {len(df)} headlines for {ticker}...")
    results = []
    for i, row in df.iterrows():
        pos, neg, neu = get_sentiment(row["headline"])
        results.append({
            "date":      row["date"],
            "headline":  row["headline"],
            "pos":       pos,
            "neg":       neg,
            "neu":       neu,
            "sentiment_score": pos - neg  # net sentiment
        })
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(df)} headlines...")

    scored_df = pd.DataFrame(results)
    return scored_df

# ── Aggregate to daily sentiment ─────────────────
def aggregate_daily(scored_df, ticker):
    daily = scored_df.groupby("date").agg(
        avg_sentiment    = ("sentiment_score", "mean"),
        max_sentiment    = ("sentiment_score", "max"),
        min_sentiment    = ("sentiment_score", "min"),
        sentiment_std    = ("sentiment_score", "std"),
        news_count       = ("headline",        "count"),
        pos_ratio        = ("pos",             "mean"),
        neg_ratio        = ("neg",             "mean"),
        neu_ratio        = ("neu",             "mean"),
    ).reset_index()
    daily["ticker"] = ticker
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    return daily

# ── Run for all tickers ──────────────────────────
if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "MSFT"]
    os.makedirs("data/processed", exist_ok=True)

    for ticker in tickers:
        print(f"\n{'='*40}")
        print(f"Processing {ticker}")
        print(f"{'='*40}")

        scored   = score_news_file(ticker)
        daily    = aggregate_daily(scored, ticker)

        # Save both detailed and daily
        scored.to_csv(f"data/processed/{ticker}_scored_news.csv",  index=False)
        daily.to_csv(f"data/processed/{ticker}_daily_sentiment.csv", index=False)

        print(f"\n{ticker} Results:")
        print(f"  Total headlines scored : {len(scored)}")
        print(f"  Trading days with news : {len(daily)}")
        print(f"  Avg sentiment score    : {daily['avg_sentiment'].mean():.4f}")
        print(f"  Most positive day      : {daily.loc[daily['avg_sentiment'].idxmax(), 'date'].date()}")
        print(f"  Most negative day      : {daily.loc[daily['avg_sentiment'].idxmin(), 'date'].date()}")

    print("\n\nAll sentiment scores saved to data/processed/")