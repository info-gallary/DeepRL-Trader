# 🚀 DeepRL-Trader-Pro

A reinforcement learning-based trading system built using PPO (actor–critic) with a custom Gym environment, incorporating transaction costs, portfolio dynamics, and risk-aware metrics such as Sharpe ratio and maximum drawdown.

---

## 📌 Overview

This project models stock trading as a **Markov Decision Process (MDP)** and trains a PPO agent to learn optimal **Buy / Sell / Hold** strategies. It includes a Streamlit dashboard for visualization, live logs, and performance analysis.

---

## 🧠 Features

- PPO-based deep reinforcement learning agent  
- Custom trading environment (Buy / Sell / Hold)  
- Transaction cost penalty  
- Portfolio balance and cash tracking  
- Long and short positions  
- Sharpe ratio and max drawdown metrics  
- Candlestick chart visualization  
- Date-based train/test split  
- Saved model support  
- Paper trading mode  
- Live decision logs  

---

## 📂 Project Structure

```bash
DeepRL-Trader-Pro/
│── app.py
│── saved_models/
│── requirements.txt
│── README.md
