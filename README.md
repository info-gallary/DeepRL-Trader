# 🚀 DeepRL-Trader-Pro

A reinforcement learning-based trading system built using PPO (actor–critic) with a custom Gym environment, incorporating transaction costs, portfolio dynamics, and risk-aware metrics such as Sharpe ratio and maximum drawdown.

---

## 📌 Overview

This project models stock trading as a **Markov Decision Process (MDP)** and trains a PPO agent to learn optimal **Buy / Sell / Hold** strategies. It includes an interactive Streamlit dashboard for visualization, live logs, and performance analysis.

## 🚀 Why PPO (and not DQN)?

### 🔹 Why PPO is Used

This project uses **Proximal Policy Optimization (PPO)**, a state-of-the-art policy gradient algorithm widely adopted in real-world reinforcement learning systems.

PPO is chosen because:

- It is **stable and reliable** compared to older policy gradient methods  
- Uses **clipped objective function** to prevent destructive policy updates  
- Works well with **continuous and noisy environments** like financial markets  
- Supports **actor–critic architecture**, improving learning efficiency  
- Requires **less hyperparameter tuning** than many RL algorithms  

In trading, where environments are **non-stationary and stochastic**, PPO provides better convergence and robustness.

---

### 🔹 Why Not DQN?

Although DQN is popular, it is **not ideal for trading systems**:

- DQN is **value-based**, not policy-based  
- Struggles with **non-stationary environments** (markets constantly change)  
- Requires **experience replay**, which may break temporal dependencies in financial data  
- Can be **unstable with noisy rewards**  
- Limited flexibility for **complex action spaces and portfolio dynamics**  

In contrast, PPO directly learns a **policy**, making it more suitable for sequential decision-making tasks like trading.

---

## 🧠 Why This Approach is Near State-of-the-Art

This system incorporates several elements used in modern RL-based trading research:

- **Actor–Critic PPO framework**
- **Custom MDP formulation for financial markets**
- **Reward shaping based on portfolio value**
- **Transaction cost modeling (realistic constraint)**
- **Risk-aware metrics (Sharpe Ratio, Max Drawdown)**
- **Backtesting + Paper Trading pipeline**
- **Live decision logging and visualization**

These components align with methodologies used in:

- Quantitative finance research  
- Algorithmic trading systems  
- Deep RL financial modeling papers  




---

## 🧠 Features

- PPO-based deep reinforcement learning agent  
- Custom trading environment (Buy / Sell / Hold actions)  
- Transaction cost penalty  
- Portfolio balance and cash tracking  
- Long and short positions  
- Sharpe ratio and max drawdown metrics  
- Candlestick chart visualization (Plotly)  
- Date-based train/test split  
- Saved model persistence  
- Paper trading mode  
- Live decision logs  

---

## 📂 Project Structure

DeepRL-Trader-Pro/
- app.py  
- saved_models/  
- requirements.txt  
- README.md  

---

## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/your-username/DeepRL-Trader-Pro.git  
cd DeepRL-Trader-Pro  

---

### 2. Create virtual environment

#### Windows

python -m venv venv  
venv\Scripts\activate  

#### macOS / Linux

python3 -m venv venv  
source venv/bin/activate  

---
### 🔹 Python Dependencies

### 3. Install dependencies

pip install streamlit pandas numpy yfinance gymnasium stable-baselines3 plotly matplotlib  

---

## ▶️ Running the Application

streamlit run app.py  

Open in browser:  
http://localhost:8501  

---

## 🧪 How It Works

### 1. Data Processing
- Fetches stock data using Yahoo Finance  
- Computes indicators:
  - RSI  
  - SMA  
  - EMA  
  - Volatility  

### 2. RL Formulation

- State → Market features over time window  
- Actions:
  - 0 → Hold  
  - 1 → Buy (Long)  
  - 2 → Sell (Short)  
- Reward → Change in portfolio value  

### 3. Training

- PPO optimizes policy using:
  - policy gradients  
  - clipped objective  
  - advantage estimation  

### 4. Evaluation

- Backtesting on unseen data  
- Comparison with buy-and-hold  
- Risk metric computation  

---

## 📊 Metrics

- Strategy Return → total portfolio return  
- Sharpe Ratio → risk-adjusted performance  
- Max Drawdown → worst loss from peak  
- Equity Curve → portfolio growth over time  

---

## 📈 Usage Workflow

1. Enter stock ticker (e.g., TSLA, AAPL)  
2. Select mode:
   - Train + Backtest  
   - Paper Trading  
3. Configure parameters:
   - training steps  
   - transaction cost  
   - observation window  
4. Choose train/test split date  
5. Run the agent  

---

## 💾 Saved Models

Trained models are automatically stored in:

saved_models/  

Used for:
- avoiding retraining  
- fast paper trading  

---

## 🧪 Paper Trading Mode

- Uses trained PPO model  
- Simulates trading on unseen data  
- Logs decisions step-by-step  
- No retraining required  

---

## 🧰 Tech Stack

- Python  
- Streamlit  
- Stable-Baselines3 (PPO)  
- Gymnasium  
- Pandas / NumPy  
- Plotly  
- Yahoo Finance  

---

## ⚠️ Disclaimer

- This project is for **educational and research purposes only**  
- It is **not financial advice**  
- Performance in backtesting does not guarantee real-world results  

---

## 🔮 Future Improvements

- Multi-asset portfolio  
- Stop-loss / take-profit strategies  
- Advanced reward engineering  
- Hyperparameter tuning  
- Live trading integration  

---

## 📜 License

MIT License
