# from fastapi import FastAPI
# import xgboost as xgb
# import pandas as pd

# app = FastAPI()
# model = xgb.XGBClassifier()
# model.load_model("models/xgb_model.json")

# @app.post("/predict")
# def predict(ticker: str, sentiment_score: float, rsi: float):
#     features = pd.DataFrame([{
#         "avg_sentiment": sentiment_score,
#         "rsi": rsi,
#         # ... other features with defaults
#     }])
#     prob = model.predict_proba(features)[0][1]
#     direction = "UP" if prob > 0.5 else "DOWN"
#     return {"ticker": ticker, "direction": direction,
#             "confidence": round(prob, 3)}




from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Financial Sentiment Stock Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ────────────────────
model = xgb.XGBClassifier()
model.load_model("models/xgb_best_model.json")

FEATURES = [
    "avg_sentiment","max_sentiment","min_sentiment",
    "sentiment_std","pos_ratio","neg_ratio",
    "sentiment_lag1","sentiment_lag2","sentiment_lag3",
    "sentiment_roll3","sentiment_roll7",
    "rsi","macd","macd_signal",
    "bb_high","bb_low","bb_width",
    "ema_20","atr","volume_ratio",
    "price_change_1d","price_change_3d","price_change_5d",
    "high_low_ratio"
]

# ── Request schema ────────────────────────────────
class PredictRequest(BaseModel):
    ticker: str
    avg_sentiment: float = 0.0
    pos_ratio: float = 0.33
    neg_ratio: float = 0.33

# ── Helper: get latest technical features ─────────
def get_technical_features(ticker: str):
    df = yf.download(ticker, period="3mo", progress=False)
    if df.empty:
        return None
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    rsi         = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd        = ta.trend.MACD(close).macd().iloc[-1]
    macd_signal = ta.trend.MACD(close).macd_signal().iloc[-1]
    bb_high     = ta.volatility.BollingerBands(close).bollinger_hband().iloc[-1]
    bb_low      = ta.volatility.BollingerBands(close).bollinger_lband().iloc[-1]
    bb_width    = ta.volatility.BollingerBands(close).bollinger_wband().iloc[-1]
    ema_20      = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    atr         = ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1]
    vol_ma7     = vol.rolling(7).mean().iloc[-1]
    vol_ratio   = float(vol.iloc[-1]) / float(vol_ma7) if vol_ma7 else 1.0
    pc1d        = close.pct_change(1).iloc[-1]
    pc3d        = close.pct_change(3).iloc[-1]
    pc5d        = close.pct_change(5).iloc[-1]
    hl_ratio    = float(high.iloc[-1]) / float(low.iloc[-1])

    return {
        "rsi": float(rsi), "macd": float(macd),
        "macd_signal": float(macd_signal),
        "bb_high": float(bb_high), "bb_low": float(bb_low),
        "bb_width": float(bb_width), "ema_20": float(ema_20),
        "atr": float(atr), "volume_ratio": float(vol_ratio),
        "price_change_1d": float(pc1d),
        "price_change_3d": float(pc3d),
        "price_change_5d": float(pc5d),
        "high_low_ratio":  float(hl_ratio),
        "current_price":   float(close.iloc[-1]),
    }

# ── Routes ────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Financial Sentiment Stock Predictor API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "xgb_best_model.json"}

@app.post("/predict")
def predict(req: PredictRequest):
    tech = get_technical_features(req.ticker)
    if tech is None:
        return {"error": f"Could not fetch data for {req.ticker}"}

    features = {
        "avg_sentiment":   req.avg_sentiment,
        "max_sentiment":   req.avg_sentiment + 0.1,
        "min_sentiment":   req.avg_sentiment - 0.1,
        "sentiment_std":   0.2,
        "pos_ratio":       req.pos_ratio,
        "neg_ratio":       req.neg_ratio,
        "sentiment_lag1":  req.avg_sentiment,
        "sentiment_lag2":  req.avg_sentiment,
        "sentiment_lag3":  req.avg_sentiment,
        "sentiment_roll3": req.avg_sentiment,
        "sentiment_roll7": req.avg_sentiment,
        **{k: v for k, v in tech.items() if k != "current_price"}
    }

    X   = pd.DataFrame([features])[FEATURES]
    prob = float(model.predict_proba(X)[0][1])
    direction = "UP" if prob >= 0.5 else "DOWN"

    return {
        "ticker":        req.ticker.upper(),
        "direction":     direction,
        "confidence":    round(prob, 4),
        "sentiment":     round(req.avg_sentiment, 4),
        "current_price": round(tech["current_price"], 2),
        "rsi":           round(tech["rsi"], 2),
        "signal":        "BUY" if direction == "UP" and prob > 0.55
                         else "SELL" if direction == "DOWN" and prob < 0.45
                         else "HOLD"
    }

@app.get("/technical/{ticker}")
def technical(ticker: str):
    tech = get_technical_features(ticker)
    if tech is None:
        return {"error": f"Could not fetch data for {ticker}"}
    return {"ticker": ticker.upper(), **tech}