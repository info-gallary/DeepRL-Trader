# 🧭 DeepRL + Agentic Trader & Investment Advisor

A robust RL trading dashboard with a LangGraph-style AI agent, sentiment analysis, and plain-English profit plans powered by Groq's LLaMA-3.3-70B.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 Groq API Key (FREE)
1. Go to https://console.groq.com
2. Sign up → API Keys → Create
3. Paste into the sidebar

## 🧠 Architecture

```
User Input
    │
    ▼
RL Training (PPO)
    │
    ▼
Backtest Engine ──────────────────────────┐
    │                                     │
    ▼                                     ▼
LangGraph Agent (3 Nodes)           Metrics & Charts
    ├─ Node 1: Sentiment Analysis
    ├─ Node 2: Risk Assessment
    └─ Node 3: Action Plan Generation
         │
         ▼
    Beginner-Friendly Report
```

## ✨ Features
- **PPO RL Agent** with Sharpe-shaped reward, normalized observations, MACD+BB features
- **LangGraph-style 3-node agent** orchestrating Groq API calls
- **Sentiment Analysis** from price action, RSI, MACD, fundamentals
- **Risk Assessment** with VaR, volatility, beginner suitability
- **Action Plan** with DOs/DON'Ts, stop-loss, take-profit, position sizing
- **Candlestick + Bollinger Bands** chart with trade markers
- **Model persistence** — saved models reloaded automatically
- **Paper Trading** mode with any saved model
- **CSV export** of full trade logs
