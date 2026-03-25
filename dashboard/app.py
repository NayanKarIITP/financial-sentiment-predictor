import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import shap
import ta
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Load model ────────────────────────────────────
@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model("models/xgb_best_model.json")
    return m

@st.cache_data(ttl=300)
def get_stock_data(ticker, period="6mo"):
    df = yf.download(ticker, period=period, progress=False)
    return df

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

def compute_features(df, sentiment):
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
    pc1d        = float(close.pct_change(1).iloc[-1])
    pc3d        = float(close.pct_change(3).iloc[-1])
    pc5d        = float(close.pct_change(5).iloc[-1])
    hl_ratio    = float(high.iloc[-1]) / float(low.iloc[-1])
    return {
        "avg_sentiment": sentiment, "max_sentiment": sentiment+0.1,
        "min_sentiment": sentiment-0.1, "sentiment_std": 0.2,
        "pos_ratio": max(0, sentiment*0.5+0.33),
        "neg_ratio": max(0, -sentiment*0.5+0.33),
        "sentiment_lag1": sentiment, "sentiment_lag2": sentiment,
        "sentiment_lag3": sentiment, "sentiment_roll3": sentiment,
        "sentiment_roll7": sentiment,
        "rsi": float(rsi), "macd": float(macd),
        "macd_signal": float(macd_signal),
        "bb_high": float(bb_high), "bb_low": float(bb_low),
        "bb_width": float(bb_width), "ema_20": float(ema_20),
        "atr": float(atr), "volume_ratio": float(vol_ratio),
        "price_change_1d": pc1d, "price_change_3d": pc3d,
        "price_change_5d": pc5d, "high_low_ratio": hl_ratio,
    }

# ── UI ────────────────────────────────────────────
st.title("📈 Financial Sentiment Stock Predictor")
st.caption("XGBoost + FinBERT | NLP × Classical ML Fusion Project")

model = load_model()

# Sidebar
with st.sidebar:
    st.header("Settings")
    ticker    = st.selectbox("Stock ticker", ["AAPL","TSLA","MSFT","GOOGL","AMZN"])
    period    = st.selectbox("Chart period", ["3mo","6mo","1y"], index=1)
    sentiment = st.slider("Today's sentiment score", -1.0, 1.0, 0.0, 0.01,
                          help="-1 = very negative, 0 = neutral, +1 = very positive")
    st.markdown("---")
    st.caption("Sentiment score comes from FinBERT analysis of financial news headlines.")

# Fetch data
df = get_stock_data(ticker, period)

if df.empty:
    st.error(f"Could not fetch data for {ticker}")
    st.stop()

close_vals = df["Close"].squeeze()
current_price = float(close_vals.iloc[-1])
prev_price    = float(close_vals.iloc[-2])
price_change  = current_price - prev_price
price_pct     = price_change / prev_price * 100

# Predict
feat_dict = compute_features(df, sentiment)
X         = pd.DataFrame([feat_dict])[FEATURES]
prob      = float(model.predict_proba(X)[0][1])
direction = "UP ↑" if prob >= 0.5 else "DOWN ↓"
signal    = "BUY"  if prob > 0.55 else "SELL" if prob < 0.45 else "HOLD"
sig_color = {"BUY":"#1D9E75","SELL":"#E24B4A","HOLD":"#BA7517"}[signal]
rsi_val   = float(ta.momentum.RSIIndicator(close_vals, window=14).rsi().iloc[-1])

# ── Metrics row ───────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price",  f"${current_price:.2f}", f"{price_pct:+.2f}%")
c2.metric("Prediction",     direction,               f"{prob*100:.1f}% confidence")
c3.metric("Signal",         signal)
c4.metric("RSI",            f"{rsi_val:.1f}",        "Overbought" if rsi_val>70 else "Oversold" if rsi_val<30 else "Neutral")
c5.metric("Sentiment",      f"{sentiment:+.2f}",     "Positive" if sentiment>0.1 else "Negative" if sentiment<-0.1 else "Neutral")

st.markdown("---")

# ── Charts ────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{ticker} Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=close_vals,
        name=ticker,
        increasing_line_color="#1D9E75",
        decreasing_line_color="#E24B4A"
    ))
    # Add EMA line
    ema = ta.trend.EMAIndicator(close_vals, window=20).ema_indicator()
    fig.add_trace(go.Scatter(x=df.index, y=ema, name="EMA 20",
                             line=dict(color="#378ADD", width=1.5)))
    fig.update_layout(
        height=400, margin=dict(l=0,r=0,t=30,b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Prediction Confidence")
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": f"P(UP) = {prob*100:.1f}%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#1D9E75" if prob>=0.5 else "#E24B4A"},
            "steps":[
                {"range":[0,45],  "color":"#FCEBEB"},
                {"range":[45,55], "color":"#FAEEDA"},
                {"range":[55,100],"color":"#E1F5EE"},
            ],
            "threshold":{"line":{"color":"black","width":3},"value":50}
        }
    ))
    fig2.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20),
                       paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"""
    <div style='text-align:center;padding:10px;border-radius:8px;
    background:{sig_color}22;border:1px solid {sig_color};'>
    <span style='font-size:22px;font-weight:600;color:{sig_color}'>{signal}</span>
    </div>
    """, unsafe_allow_html=True)

# ── SHAP feature importance ───────────────────────
st.markdown("---")
st.subheader("Feature Importance (SHAP)")

try:
    shap_bar = open("models/shap_bar.png", "rb").read()
    shap_dot = open("models/shap_dot.png", "rb").read()
    sc1, sc2 = st.columns(2)
    with sc1:
        st.image(shap_bar, caption="Top features by SHAP magnitude", use_container_width=True)
    with sc2:
        st.image(shap_dot, caption="SHAP impact direction", use_container_width=True)
except:
    st.info("Run src/train.py first to generate SHAP plots.")

# ── Experiment results ────────────────────────────
st.markdown("---")
st.subheader("Model Experiment Results")
results_df = pd.DataFrame({
    "Experiment":   ["Technical Only", "Sentiment Only", "Combined (Best)"],
    "AUC":          [0.517, 0.500, 0.5296],
    "Accuracy":     [0.5027, 0.5062, 0.5133],
    "F1 Score":     [0.5789, 0.6711, 0.5903],
})
fig3 = px.bar(results_df, x="Experiment", y="AUC",
              color="Experiment",
              color_discrete_sequence=["#9FE1CB","#AFA9EC","#1D9E75"],
              title="AUC Score by Experiment — Combined model wins")
fig3.add_hline(y=0.5, line_dash="dash", line_color="gray",
               annotation_text="Random baseline (0.50)")
fig3.update_layout(showlegend=False, height=300,
                   plot_bgcolor="rgba(0,0,0,0)",
                   paper_bgcolor="rgba(0,0,0,0)",
                   margin=dict(t=40,b=0,l=0,r=0))
st.plotly_chart(fig3, use_container_width=True)
st.dataframe(results_df.set_index("Experiment"), use_container_width=True)

st.markdown("---")
st.caption("Built with XGBoost + FinBERT (HuggingFace) + MLflow + Streamlit | Financial Sentiment Predictor")