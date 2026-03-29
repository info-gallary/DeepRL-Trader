# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import gymnasium as gym
# from gym_anytrading.envs import StocksEnv
# from stable_baselines3 import PPO
# import matplotlib.pyplot as plt

# # --- 1. Enhanced Environment with Logging Capability ---
# class LoggingTradingEnv(StocksEnv):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.logs = []

#     def _calculate_reward(self, action):
#         current_tick = self._current_tick
#         current_price = self.prices[current_tick]
#         last_price = self.prices[current_tick - 1]
#         price_diff = current_price - last_price
        
#         # Action Map: 0 = Sell/Short, 1 = Buy/Long
#         act_name = "LONG" if action == 1 else "SHORT"
        
#         # Basic Reward Logic
#         step_reward = price_diff if action == 1 else -price_diff
        
#         # Log the activity
#         log_entry = f"Tick {current_tick}: {act_name} | Price: {current_price:.2f} | Rew: {step_reward:.4f}"
#         self.logs.append(log_entry)
        
#         return step_reward

#     def _process_data(self):
#         start = self.frame_bound[0] - self.window_size
#         end = self.frame_bound[1]
#         prices = self.df.loc[:, 'Close'].to_numpy()[start:end].flatten().astype(np.float32)
#         signal_features = self.df.loc[:, ['Close', 'RSI', 'SMA']].to_numpy()[start:end].astype(np.float32)
#         return prices, signal_features

# # --- 2. Streamlit UI ---
# st.set_page_config(page_title="Pro AI Trader", layout="wide")

# # Sidebar for Config and Logs
# st.sidebar.header("🛠️ Control Panel")
# ticker = st.sidebar.text_input("Ticker Symbol", value="TSLA")
# iters = st.sidebar.slider("Training Steps", 5000, 30000, 10000)

# st.sidebar.markdown("---")
# st.sidebar.header("📜 Live Activity Log")
# log_container = st.sidebar.empty() # Placeholder for the log text

# # --- 3. Logic ---
# @st.cache_data
# def get_data(symbol):
#     data = yf.download(symbol, period="2y", interval="1d")
#     df = data.copy()
#     if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
#     df['SMA'] = df['Close'].rolling(window=15).mean()
#     delta = df['Close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
#     return df.dropna().astype(np.float32)

# if st.sidebar.button("Launch Agent"):
#     df = get_data(ticker)
#     split = int(len(df) * 0.8)
    
#     # Training
#     env_train = LoggingTradingEnv(df=df, window_size=12, frame_bound=(12, split))
#     model = PPO("MlpPolicy", env_train, verbose=0)
    
#     with st.spinner("Brain training in progress..."):
#         model.learn(total_timesteps=iters)

#     # Simulation with Sidebar Logging
#     env_test = LoggingTradingEnv(df=df, window_size=12, frame_bound=(split, len(df)-1))
#     obs, info = env_test.reset()
    
#     done = False
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env_test.step(action)
#         done = terminated or truncated
        
#         # Update the Sidebar Log Live
#         log_text = "\n".join(env_test.logs[-15:]) # Show last 15 actions
#         log_container.code(log_text, language="text")

#     # --- 4. Main Display ---
#     st.title(f"Analysis: {ticker}")
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         fig, ax = plt.subplots(figsize=(12, 5))
#         env_test.render_all()
#         st.pyplot(fig)
        
#     with col2:
#         st.subheader("Performance Metrics")
#         ai_ret = (info['total_profit'] - 1)
#         st.metric("Strategy Return", f"{ai_ret:.2%}")
#         st.metric("Total Reward", f"{info['total_reward']:.2f}")
        
#         st.info("The sidebar log shows the step-by-step decisions made by the Neural Network during the test phase.")
import os
import hashlib
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import plotly.graph_objects as go

st.set_page_config(page_title="RL", layout="wide")

MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(show_spinner=False)
def get_data(symbol, period="5y", interval="1d"):
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Ticker symbol cannot be empty.")
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data found for ticker '{symbol}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy().reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["SMA_15"] = df["Close"].rolling(15).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    df["Returns"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Returns"].rolling(10).std()
    df.dropna(inplace=True)
    if len(df) < 120:
        raise ValueError("Not enough processed rows. Try another ticker or longer period.")
    return df

def annualized_sharpe(returns, periods_per_year=252):
    returns = pd.Series(returns).dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / (returns.std() + 1e-12))

def max_drawdown(equity_curve):
    eq = pd.Series(equity_curve).astype(float)
    roll_max = eq.cummax()
    dd = (eq - roll_max) / (roll_max + 1e-12)
    return float(dd.min())

def list_saved_models():
    return sorted([p.name for p in MODELS_DIR.glob("*.zip")])

def model_filename(ticker, window_size, tx_cost, train_steps, train_end_date):
    key = f"{ticker}_{window_size}_{tx_cost}_{train_steps}_{train_end_date}"
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    return MODELS_DIR / f"{ticker}_{train_end_date}_{digest}"

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=20, initial_cash=10000, transaction_cost_pct=0.001):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.window_size = int(window_size)
        self.initial_cash = float(initial_cash)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.feature_columns = ["Close", "SMA_15", "EMA_10", "RSI_14", "Returns", "Volatility_10", "Volume"]
        self.action_space = spaces.Discrete(3)
        obs_shape = (self.window_size * len(self.feature_columns),)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.reset()

    def _get_observation(self):
        frame = self.df.loc[self.current_step - self.window_size:self.current_step - 1, self.feature_columns].copy()
        arr = frame.to_numpy(dtype=np.float32)
        return arr.flatten()

    def _portfolio_value(self, price):
        return self.cash + self.shares * price

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares = 0
        self.position = 0
        self.logs = []
        self.trades = []
        self.equity_curve = [self.initial_cash]
        self.portfolio_returns = []
        self.last_portfolio_value = self.initial_cash
        return self._get_observation(), {}

    def step(self, action):
        terminated = False
        truncated = False

        row = self.df.iloc[self.current_step]
        price = float(row["Close"])
        date = pd.to_datetime(row["Date"])
        prev_value = self._portfolio_value(price)

        target_position = self.position
        if action == 1:
            target_position = 1
        elif action == 2:
            target_position = -1

        trade_executed = False
        units_changed = abs(target_position - self.position)

        if target_position != self.position:
            trade_cost = price * units_changed * self.transaction_cost_pct

            if self.position == 1:
                self.cash += price
                self.shares -= 1
            elif self.position == -1:
                self.cash -= price
                self.shares += 1

            if target_position == 1:
                if self.cash >= price + trade_cost:
                    self.cash -= (price + trade_cost)
                    self.shares += 1
                    self.position = 1
                    trade_executed = True
                else:
                    target_position = 0
                    self.position = 0
                    self.cash -= trade_cost
            elif target_position == -1:
                self.cash += (price - trade_cost)
                self.shares -= 1
                self.position = -1
                trade_executed = True
            else:
                self.cash -= trade_cost
                self.position = 0
                trade_executed = True

        current_value = self._portfolio_value(price)
        reward = (current_value - prev_value) / max(self.initial_cash, 1.0)

        self.equity_curve.append(current_value)
        step_ret = (current_value - self.last_portfolio_value) / max(self.last_portfolio_value, 1e-12)
        self.portfolio_returns.append(step_ret)
        self.last_portfolio_value = current_value

        action_name = {0: "HOLD", 1: "BUY/LONG", 2: "SELL/SHORT"}[int(action)]
        log_entry = {
            "Date": date.strftime("%Y-%m-%d"),
            "Action": action_name,
            "Price": round(price, 2),
            "Position": int(self.position),
            "Shares": int(self.shares),
            "Cash": round(self.cash, 2),
            "Portfolio": round(current_value, 2),
            "Reward": round(float(reward), 6)
        }
        self.logs.append(log_entry)

        if trade_executed or action == 0:
            self.trades.append({
                "Date": date,
                "Action": action_name,
                "Price": price,
                "Position": int(self.position),
                "Portfolio": current_value
            })

        self.current_step += 1
        if self.current_step >= len(self.df):
            terminated = True

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "portfolio_value": current_value,
            "cash": self.cash,
            "shares": self.shares,
            "position": self.position,
            "max_drawdown": max_drawdown(self.equity_curve),
            "sharpe_ratio": annualized_sharpe(self.portfolio_returns)
        }
        return obs, float(reward), terminated, truncated, info

def evaluate_model_live(model, env, right_log_placeholder):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        right_log_placeholder.dataframe(pd.DataFrame(env.logs).tail(50), use_container_width=True, height=700)
    return info, pd.DataFrame(env.logs), pd.DataFrame(env.trades)

def paper_trade_live(model, env, right_log_placeholder):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        right_log_placeholder.dataframe(pd.DataFrame(env.logs).tail(50), use_container_width=True, height=700)
    return info, pd.DataFrame(env.logs), pd.DataFrame(env.trades)

def plot_candlestick(df_slice, trades_df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_slice["Date"],
        open=df_slice["Open"],
        high=df_slice["High"],
        low=df_slice["Low"],
        close=df_slice["Close"],
        name="Candles"
    ))
    if trades_df is not None and not trades_df.empty:
        buy_df = trades_df[trades_df["Action"] == "BUY/LONG"]
        sell_df = trades_df[trades_df["Action"] == "SELL/SHORT"]
        hold_df = trades_df[trades_df["Action"] == "HOLD"]

        if not buy_df.empty:
            fig.add_trace(go.Scatter(
                x=buy_df["Date"],
                y=buy_df["Price"],
                mode="markers",
                name="Buy/Long",
                marker=dict(size=10, symbol="triangle-up")
            ))
        if not sell_df.empty:
            fig.add_trace(go.Scatter(
                x=sell_df["Date"],
                y=sell_df["Price"],
                mode="markers",
                name="Sell/Short",
                marker=dict(size=10, symbol="triangle-down")
            ))
        if not hold_df.empty:
            fig.add_trace(go.Scatter(
                x=hold_df["Date"],
                y=hold_df["Price"],
                mode="markers",
                name="Hold",
                marker=dict(size=6, symbol="circle")
            ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650
    )
    return fig

st.title("RL INNOVATIVE")
# st.caption("Educational RL trading dashboard with hold action, transaction costs, saved models, paper trading, live logs, risk metrics, and date-based splits.")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Train + Backtest", "Paper Trading"])
    ticker = st.text_input("Ticker Symbol", value="TSLA").strip().upper()
    period = st.selectbox("Data Period", ["1y", "2y", "5y", "10y"], index=2)
    window_size = st.slider("Observation Window", 5, 60, 20)
    initial_cash = st.number_input("Initial Cash", min_value=1000.0, value=10000.0, step=1000.0)
    tx_cost = st.number_input("Transaction Cost %", min_value=0.0, max_value=5.0, value=0.10, step=0.01) / 100.0
    train_steps = st.slider("Training Steps", 2000, 50000, 10000, step=1000)
    available_models = list_saved_models()
    selected_model_name = st.selectbox("Saved Model", [""] + available_models)
    run_button = st.button("Run")

if not run_button:
    st.stop()

try:
    df = get_data(ticker, period=period)
except Exception as e:
    st.error(str(e))
    st.stop()

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

if mode == "Train + Backtest":
    default_train_end = df["Date"].iloc[int(len(df) * 0.8)].date()
    train_end_date = st.date_input("Train End Date", value=default_train_end, min_value=min_date, max_value=max_date)
    train_df = df[df["Date"].dt.date <= train_end_date].copy()
    test_df = df[df["Date"].dt.date > train_end_date].copy()

    if len(train_df) < window_size + 30 or len(test_df) < window_size + 10:
        st.error("Train or test split is too small. Choose a different train end date.")
        st.stop()

    left, right = st.columns([3.2, 1.4])

    with right:
        st.subheader("Live Logs")
        right_log_placeholder = st.empty()

    with left:
        st.subheader("Training / Backtest Output")
        train_env = TradingEnv(train_df, window_size=window_size, initial_cash=initial_cash, transaction_cost_pct=tx_cost)
        model_path = model_filename(ticker, window_size, tx_cost, train_steps, str(train_end_date))
        status = st.empty()
        progress = st.progress(0)

        if model_path.with_suffix(".zip").exists():
            status.info("Loading saved trained model...")
            model = PPO.load(str(model_path), env=train_env)
            progress.progress(35)
        else:
            status.info("Training model...")
            model = PPO("MlpPolicy", train_env, verbose=0)
            progress.progress(20)
            model.learn(total_timesteps=train_steps)
            progress.progress(75)
            model.save(str(model_path))
            status.success(f"Model saved: {model_path.name}.zip")

        test_env = TradingEnv(test_df, window_size=window_size, initial_cash=initial_cash, transaction_cost_pct=tx_cost)
        status.info("Running live backtest...")
        info, logs_df, trades_df = evaluate_model_live(model, test_env, right_log_placeholder)
        progress.progress(100)

        test_prices = test_df["Close"].reset_index(drop=True)
        if len(test_prices) >= 2:
            buy_hold_return = float((test_prices.iloc[-1] / test_prices.iloc[0]) - 1.0)
        else:
            buy_hold_return = 0.0

        strategy_return = float((info["portfolio_value"] / initial_cash) - 1.0)
        mdd = info["max_drawdown"]
        sharpe = info["sharpe_ratio"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final Portfolio", f"${info['portfolio_value']:,.2f}")
        c2.metric("Strategy Return", f"{strategy_return:.2%}")
        c3.metric("Buy & Hold", f"{buy_hold_return:.2%}")
        c4.metric("Max Drawdown", f"{mdd:.2%}")
        c5.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.plotly_chart(plot_candlestick(test_df, trades_df, f"{ticker} Backtest Candlestick"), use_container_width=True)

        eq_df = pd.DataFrame({
            "Step": range(len(test_env.equity_curve)),
            "Equity": test_env.equity_curve
        })
        st.line_chart(eq_df.set_index("Step"))

        st.subheader("Trade Log")
        st.dataframe(logs_df, use_container_width=True, height=350)

        st.download_button(
            "Download Trade Log CSV",
            logs_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_backtest_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    if not selected_model_name:
        st.error("Select a saved model for paper trading.")
        st.stop()

    paper_start_default = max(min_date, df["Date"].iloc[max(window_size + 1, len(df) - 90)].date())
    paper_start_date = st.date_input("Paper Trade Start Date", value=paper_start_default, min_value=min_date, max_value=max_date)
    paper_df = df[df["Date"].dt.date >= paper_start_date].copy()

    if len(paper_df) < window_size + 10:
        st.error("Paper trading range is too small.")
        st.stop()

    left, right = st.columns([3.2, 1.4])

    with right:
        st.subheader("Live Logs")
        right_log_placeholder = st.empty()

    with left:
        st.subheader("Paper Trading Output")
        env = TradingEnv(paper_df, window_size=window_size, initial_cash=initial_cash, transaction_cost_pct=tx_cost)
        model_file = MODELS_DIR / selected_model_name
        model = PPO.load(str(model_file), env=env)

        status = st.empty()
        status.info("Running paper trading simulation...")
        info, logs_df, trades_df = paper_trade_live(model, env, right_log_placeholder)

        strategy_return = float((info["portfolio_value"] / initial_cash) - 1.0)
        mdd = info["max_drawdown"]
        sharpe = info["sharpe_ratio"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Portfolio", f"${info['portfolio_value']:,.2f}")
        c2.metric("Paper Return", f"{strategy_return:.2%}")
        c3.metric("Max Drawdown", f"{mdd:.2%}")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.plotly_chart(plot_candlestick(paper_df, trades_df, f"{ticker} Paper Trading Candlestick"), use_container_width=True)

        eq_df = pd.DataFrame({
            "Step": range(len(env.equity_curve)),
            "Equity": env.equity_curve
        })
        st.line_chart(eq_df.set_index("Step"))

        st.subheader("Paper Trading Log")
        st.dataframe(logs_df, use_container_width=True, height=350)

        st.download_button(
            "Download Paper Trading Log CSV",
            logs_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_paper_trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
