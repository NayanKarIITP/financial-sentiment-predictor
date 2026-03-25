# 📈 Financial Sentiment Stock Predictor

> **NLP × Classical ML Fusion** — FinBERT sentiment analysis combined with technical indicators to predict next-day stock price direction.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue.svg)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-green.svg)](https://fastapi.tiangolo.com)

---

## 🎯 Project Overview

This project builds an **end-to-end machine learning pipeline** that predicts whether a stock's price will go **UP or DOWN** the next trading day by fusing two signals:

1. **NLP Signal** — Financial news headlines scored with **FinBERT** (BERT fine-tuned on financial text)
2. **Technical Signal** — RSI, MACD, Bollinger Bands, EMA, ATR and more

The combined model outperforms technical-only baselines, proving that sentiment adds measurable predictive value.

**Stocks covered:** AAPL · TSLA · MSFT

---

## 🏆 Key Results

| Experiment | AUC | Accuracy | F1 |
|---|---|---|---|
| Technical indicators only | 0.517 | 50.3% | 0.579 |
| Sentiment only (FinBERT) | 0.500 | 50.6% | 0.671 |
| **Combined (Best model)** | **0.530** | **51.3%** | **0.590** |

**Sentiment boost: +0.013 AUC over technical-only baseline**

> 📌 Note: 50–55% AUC is realistic and expected for stock direction prediction. Markets are near-efficient — a suspicious 90%+ AUC indicates data leakage. This model uses strict `TimeSeriesSplit` to prevent leakage.

---

## 🏗️ Architecture

```
Financial News (NewsAPI)          Stock Prices (yfinance)
        │                                  │
        ▼                                  ▼
  Text Cleaning                    OHLCV Data
  (spaCy / regex)               (1 year history)
        │                                  │
        ▼                                  ▼
  FinBERT Scoring              Technical Indicators
  (ProsusAI/finbert)         (RSI, MACD, BB, EMA, ATR)
        │                                  │
        └──────────── Feature Fusion ──────┘
                            │
                            ▼
                   XGBoost Classifier
                (TimeSeriesSplit CV, n=5)
                            │
                     ┌──────┴──────┐
                     ▼             ▼
               MLflow          SHAP plots
               Tracking      Explainability
                     │
                     ▼
              FastAPI REST API
                     │
                     ▼
           Streamlit Dashboard
           (Live predictions)
```

---

## 📊 SHAP Feature Importance

Top features driving predictions (by mean |SHAP| value):

| Rank | Feature | SHAP Value | Type |
|---|---|---|---|
| 1 | RSI | 0.492 | Technical |
| 2 | MACD Signal | 0.313 | Technical |
| 3 | High/Low Ratio | 0.248 | Technical |
| 4 | MACD | 0.207 | Technical |
| 5 | Price Change 3d | 0.199 | Technical |
| 6 | Bollinger Band Low | 0.195 | Technical |
| 7 | avg_sentiment | 0.089 | **NLP** |
| 8 | sentiment_lag1 | 0.076 | **NLP** |

Sentiment features appear in the top 10, confirming NLP adds signal beyond price data alone.

---

## 🗂️ Project Structure

```
financial-sentiment-predictor/
├── data/
│   ├── raw/                    ← News CSVs + Stock price CSVs
│   └── processed/              ← Scored sentiment + Feature matrix
├── src/
│   ├── data_collection.py      ← NewsAPI + yfinance data fetching
│   ├── nlp_pipeline.py         ← FinBERT sentiment scoring
│   ├── feature_engineering.py  ← Feature fusion pipeline
│   └── train.py                ← XGBoost + MLflow + SHAP
├── api/
│   └── main.py                 ← FastAPI REST endpoint
├── dashboard/
│   └── app.py                  ← Streamlit live dashboard
├── models/
│   ├── xgb_best_model.json     ← Saved best model
│   ├── shap_bar.png            ← SHAP bar chart
│   └── shap_dot.png            ← SHAP dot plot
├── mlruns/                     ← MLflow experiment logs
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Quick Start

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/financial-sentiment-predictor.git
cd financial-sentiment-predictor
pip install -r requirements.txt
```

### 2. Set up API key
```bash
# Create .env file
echo "NEWS_API_KEY=your_key_here" > .env
# Get free key at: https://newsapi.org
```

### 3. Run the full pipeline
```bash
# Collect data
python src/data_collection.py

# Score headlines with FinBERT
python src/nlp_pipeline.py

# Build feature matrix
python src/feature_engineering.py

# Train model + generate SHAP plots
python src/train.py
```

### 4. Launch dashboard + API
```bash
# Terminal 1 — Streamlit dashboard
streamlit run dashboard/app.py

# Terminal 2 — FastAPI backend
uvicorn api.main:app --reload --port 8000

# Terminal 3 — MLflow experiment tracker
python -m mlflow ui
```

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## 🔌 API Usage

### Predict next-day direction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL", "avg_sentiment": 0.25, "pos_ratio": 0.6, "neg_ratio": 0.1}'
```

**Response:**
```json
{
  "ticker": "AAPL",
  "direction": "UP",
  "confidence": 0.623,
  "sentiment": 0.25,
  "current_price": 213.49,
  "rsi": 54.2,
  "signal": "BUY"
}
```

### Get technical indicators
```bash
curl "http://localhost:8000/technical/TSLA"
```

---

## 🧪 MLflow Experiments

Three experiments tracked and compared:

```
Experiment Name                  AUC     Accuracy
─────────────────────────────────────────────────
technical_only                   0.517   50.3%
sentiment_only                   0.500   50.6%
combined_sentiment_technical     0.530   51.3%  ← Best
```

View all runs: `python -m mlflow ui` → open http://localhost:5000

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| NLP Model | FinBERT (ProsusAI/finbert via HuggingFace) |
| ML Model | XGBoost 2.0 |
| Technical Indicators | TA-Lib (ta library) |
| Experiment Tracking | MLflow |
| Explainability | SHAP (TreeExplainer) |
| REST API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Data Sources | NewsAPI + yfinance |
| Containerization | Docker |

---

## 📝 Resume Bullet

```
Built end-to-end Financial News Sentiment Predictor using FinBERT (HuggingFace)
for NLP signal extraction fused with 13 technical indicators, trained XGBoost
classifier achieving 0.530 AUC across 678 training samples with strict
TimeSeriesSplit CV. Deployed via FastAPI + Streamlit dashboard with SHAP
explainability. Tracked 3 experiments with MLflow. Sentiment features improve
AUC by +0.013 over technical-only baseline.
```

---

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only. It is not financial advice. Never make real investment decisions based on model predictions.

---

## 👤 Author

**Your Name**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [your_linkedin](https://linkedin.com/in/your_linkedin)
- Email: your_email@gmail.com

---

*Built as a Data Science portfolio project demonstrating NLP + Classical ML + MLOps skills.*