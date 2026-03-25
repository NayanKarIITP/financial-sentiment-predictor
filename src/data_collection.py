from newsapi import NewsApiClient
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
# ── News collection ──────────────────────────────
def fetch_news(query="Apple stock", days=30):
    api = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
    articles = api.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=100
    )
    records = []
    for a in articles["articles"]:
        records.append({
            "date": a["publishedAt"][:10],
            "headline": a["title"] or "",
            "source": a["source"]["name"]
        })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["headline"] != ""]
    return df
# ── Stock price collection ────────────────────────
def fetch_stock(ticker="AAPL", period="1y"):
    df = yf.download(ticker, period=period)
    df = df[["Open","High","Low","Close","Volume"]]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df

# ── Run both and save ────────────────────────────
if __name__ == "__main__":
    tickers = {"AAPL": "Apple stock",
               "TSLA": "Tesla stock",
               "MSFT": "Microsoft stock"}

    for ticker, query in tickers.items():
        print(f"\nFetching data for {ticker}...")
        news = fetch_news(query=query)
        stock = fetch_stock(ticker=ticker)
        news.to_csv(f"data/raw/{ticker}_news.csv", index=False)
        stock.to_csv(f"data/raw/{ticker}_prices.csv")
        print(f"  News: {len(news)} articles")
        print(f"  Stock: {len(stock)} trading days")
    print("\nAll data saved to data/raw/")
