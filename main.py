import os
import json
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
from stable_baselines3.common.callbacks import BaseCallback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RL Trader Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark premium theme */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1629 0%, #161d35 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080c18 !important;
    border-right: 1px solid #1e2d4a;
}

[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    color: #94a3b8 !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-weight: 700;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(14,165,233,0.4) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1e2d4a;
}

/* Title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    margin-bottom: 0;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 32px;
}

/* Status tags */
.status-ok {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.05em;
}

.status-warn {
    display: inline-block;
    background: rgba(245,158,11,0.15);
    color: #f59e0b;
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
}

.model-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #94a3b8;
}

.model-card strong {
    color: #38bdf8;
    font-size: 12px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants & Directories
# ─────────────────────────────────────────────
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = ["Close", "SMA_15", "EMA_10", "RSI_14", "Returns", "Volatility_10", "Volume"]

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


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


def calmar_ratio(returns, equity_curve, periods_per_year=252):
    mdd = abs(max_drawdown(equity_curve))
    if mdd < 1e-9:
        return 0.0
    ann_return = float(np.mean(returns) * periods_per_year)
    return ann_return / mdd


def win_rate(logs_df):
    trades = logs_df[logs_df["Action"] != "HOLD"]
    if trades.empty:
        return 0.0
    wins = (trades["Reward"] > 0).sum()
    return float(wins / len(trades))


# ─────────────────────────────────────────────
# Model metadata helpers  ← KEY FIX
# ─────────────────────────────────────────────
def meta_path(model_path: Path) -> Path:
    """Return the .json metadata file path for a given model .zip path."""
    return model_path.with_suffix(".json")


def save_model_meta(model_path: Path, meta: dict):
    with open(meta_path(model_path), "w") as f:
        json.dump(meta, f, indent=2)


def load_model_meta(model_stem: str) -> dict | None:
    """Load metadata for a model given its stem (no extension)."""
    p = MODELS_DIR / (model_stem + ".json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def list_saved_models():
    """Return list of .zip model stems that also have matching .json metadata."""
    zips = {p.stem for p in MODELS_DIR.glob("*.zip")}
    jsons = {p.stem for p in MODELS_DIR.glob("*.json")}
    valid = sorted(zips & jsons)          # only show models with metadata
    orphans = sorted(zips - jsons)        # old models without metadata
    return valid, orphans


def model_filename(ticker, window_size, tx_cost, train_steps, train_end_date):
    key = f"{ticker}_{window_size}_{tx_cost}_{train_steps}_{train_end_date}"
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    return MODELS_DIR / f"{ticker}_{train_end_date}_{digest}"


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(symbol, period="5y"):
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Ticker symbol cannot be empty.")
    df = yf.download(symbol, period=period, interval="1d",
                     auto_adjust=False, progress=False)
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

    # Features
    df["SMA_15"] = df["Close"].rolling(15).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    df["Returns"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Returns"].rolling(10).std()
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    if len(df) < 120:
        raise ValueError("Not enough processed rows. Try another ticker or longer period.")
    return df


# ─────────────────────────────────────────────
# Training progress callback
# ─────────────────────────────────────────────
class ProgressCallback(BaseCallback):
    def __init__(self, total_steps, progress_bar, status_text):
        super().__init__()
        self.total_steps = total_steps
        self.progress_bar = progress_bar
        self.status_text = status_text

    def _on_step(self) -> bool:
        pct = min(int(self.num_timesteps / self.total_steps * 100), 100)
        self.progress_bar.progress(pct)
        self.status_text.markdown(
            f'<p style="font-family:Space Mono,monospace;font-size:12px;color:#64748b;">'
            f'Training… step {self.num_timesteps:,} / {self.total_steps:,}</p>',
            unsafe_allow_html=True,
        )
        return True


# ─────────────────────────────────────────────
# Trading Environment
# ─────────────────────────────────────────────
class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=20, initial_cash=10_000,
                 transaction_cost_pct=0.001, feature_columns=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.window_size = int(window_size)
        self.initial_cash = float(initial_cash)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.feature_columns = feature_columns or FEATURE_COLUMNS

        self.action_space = spaces.Discrete(3)  # 0=HOLD, 1=LONG, 2=SHORT
        obs_dim = self.window_size * len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.reset()

    # ── obs ──
    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step          # exclusive
        frame = self.df.loc[start:end - 1, self.feature_columns].copy()
        arr = frame.to_numpy(dtype=np.float32)
        # Normalise each column independently for stability
        col_std = arr.std(axis=0, keepdims=True) + 1e-8
        col_mean = arr.mean(axis=0, keepdims=True)
        arr = (arr - col_mean) / col_std
        return arr.flatten()

    def _portfolio_value(self, price):
        return self.cash + self.shares * price

    # ── reset ──
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

    # ── step ──
    def step(self, action):
        terminated = False
        truncated = False

        row = self.df.iloc[self.current_step]
        price = float(row["Close"])
        date = pd.to_datetime(row["Date"])
        prev_value = self._portfolio_value(price)

        target_position = int(action) - 1   # 0→-1(hold→0) mapping: action 0=hold,1=long,2=short
        # Re-map cleanly: 0=hold(keep), 1=long, 2=short
        action_int = int(action)
        if action_int == 0:
            target_position = self.position   # stay
        elif action_int == 1:
            target_position = 1
        else:
            target_position = -1

        trade_executed = False

        if target_position != self.position:
            # Close existing position first
            if self.position == 1:
                self.cash += price * (1 - self.transaction_cost_pct)
                self.shares -= 1
            elif self.position == -1:
                cost = price * (1 + self.transaction_cost_pct)
                self.cash -= cost
                self.shares += 1

            # Open new position
            if target_position == 1:
                cost = price * (1 + self.transaction_cost_pct)
                if self.cash >= cost:
                    self.cash -= cost
                    self.shares += 1
                    self.position = 1
                    trade_executed = True
                else:
                    self.position = 0   # can't afford — go flat
            elif target_position == -1:
                proceeds = price * (1 - self.transaction_cost_pct)
                self.cash += proceeds
                self.shares -= 1
                self.position = -1
                trade_executed = True
            else:
                self.position = 0
                trade_executed = True

        current_value = self._portfolio_value(price)
        # Reward: risk-adjusted return to encourage consistent gains
        step_ret = (current_value - self.last_portfolio_value) / max(self.last_portfolio_value, 1e-9)
        reward = step_ret * 10   # scale reward signal

        self.equity_curve.append(current_value)
        self.portfolio_returns.append(step_ret)
        self.last_portfolio_value = current_value

        action_name = {0: "HOLD", 1: "BUY/LONG", 2: "SELL/SHORT"}[action_int]
        log_entry = {
            "Date": date.strftime("%Y-%m-%d"),
            "Action": action_name,
            "Price": round(price, 4),
            "Position": int(self.position),
            "Shares": int(self.shares),
            "Cash": round(self.cash, 2),
            "Portfolio": round(current_value, 2),
            "Reward": round(float(reward), 6),
        }
        self.logs.append(log_entry)

        if trade_executed:
            self.trades.append({
                "Date": date,
                "Action": action_name,
                "Price": price,
                "Position": int(self.position),
                "Portfolio": current_value,
            })

        self.current_step += 1
        if self.current_step >= len(self.df):
            terminated = True

        obs = (self._get_observation() if not terminated
               else np.zeros(self.observation_space.shape, dtype=np.float32))
        info = {
            "portfolio_value": current_value,
            "cash": self.cash,
            "shares": self.shares,
            "position": self.position,
            "max_drawdown": max_drawdown(self.equity_curve),
            "sharpe_ratio": annualized_sharpe(self.portfolio_returns),
            "calmar_ratio": calmar_ratio(self.portfolio_returns, self.equity_curve),
        }
        return obs, float(reward), terminated, truncated, info


# ─────────────────────────────────────────────
# Run env to completion
# ─────────────────────────────────────────────
def run_env(model, env, log_placeholder):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        if log_placeholder is not None and len(env.logs) % 5 == 0:
            log_placeholder.dataframe(
                pd.DataFrame(env.logs).tail(40),
                use_container_width=True,
                height=660,
            )
    if log_placeholder is not None:
        log_placeholder.dataframe(
            pd.DataFrame(env.logs).tail(40),
            use_container_width=True,
            height=660,
        )
    return info, pd.DataFrame(env.logs), pd.DataFrame(env.trades)


# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,26,0.8)",
    font=dict(family="Space Mono, monospace", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e2d4a", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2d4a", showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d4a", borderwidth=1),
)


def plot_candlestick(df_slice, trades_df, title):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )
    # Candles
    fig.add_trace(go.Candlestick(
        x=df_slice["Date"],
        open=df_slice["Open"], high=df_slice["High"],
        low=df_slice["Low"], close=df_slice["Close"],
        name="Price",
        increasing_line_color="#10b981",
        decreasing_line_color="#f43f5e",
    ), row=1, col=1)

    # Volume bars
    colors = ["#10b981" if c >= o else "#f43f5e"
              for c, o in zip(df_slice["Close"], df_slice["Open"])]
    fig.add_trace(go.Bar(
        x=df_slice["Date"], y=df_slice["Volume"],
        name="Volume", marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    # Trade markers
    if trades_df is not None and not trades_df.empty:
        action_cfg = {
            "BUY/LONG":   dict(symbol="triangle-up",   color="#38bdf8", size=12),
            "SELL/SHORT": dict(symbol="triangle-down",  color="#f472b6", size=12),
        }
        for action, cfg in action_cfg.items():
            sub = trades_df[trades_df["Action"] == action]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["Date"], y=sub["Price"],
                    mode="markers", name=action,
                    marker=dict(size=cfg["size"], symbol=cfg["symbol"],
                                color=cfg["color"],
                                line=dict(width=1, color="white")),
                ), row=1, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne, sans-serif", size=18, color="#e2e8f0")),
        height=700,
        xaxis_rangeslider_visible=False,
        **CHART_TEMPLATE,
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def plot_equity(equity_curve, initial_cash, title="Equity Curve"):
    eq = pd.Series(equity_curve)
    bh_line = [initial_cash] * len(eq)   # flat baseline

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(eq))), y=eq,
        mode="lines", name="Portfolio",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(eq))), y=bh_line,
        mode="lines", name="Initial Capital",
        line=dict(color="#475569", width=1, dash="dot"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne, sans-serif", size=16, color="#e2e8f0")),
        height=320,
        xaxis_title="Step",
        yaxis_title="Portfolio Value ($)",
        **CHART_TEMPLATE,
    )
    return fig


def plot_returns_dist(returns, title="Return Distribution"):
    returns = pd.Series(returns).dropna() * 100
    fig = go.Figure(go.Histogram(
        x=returns, nbinsx=60,
        marker_color="#818cf8", opacity=0.8,
        name="Daily Return %",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#f43f5e", line_dash="dash")
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne, sans-serif", size=16, color="#e2e8f0")),
        height=300,
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        **CHART_TEMPLATE,
    )
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">⚙ Configuration</p>', unsafe_allow_html=True)

    mode = st.radio("Mode", ["Train + Backtest", "Paper Trading"], index=0)

    st.markdown("---")
    st.markdown('<p class="section-header">Market Data</p>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker Symbol", value="TSLA").strip().upper()
    period = st.selectbox("Data Period", ["1y", "2y", "5y", "10y"], index=2)

    st.markdown("---")
    st.markdown('<p class="section-header">Environment</p>', unsafe_allow_html=True)
    window_size = st.slider("Observation Window (days)", 5, 60, 20)
    initial_cash = st.number_input("Initial Cash ($)", min_value=1000.0,
                                   value=10_000.0, step=1000.0)
    tx_cost = st.number_input("Transaction Cost (%)", min_value=0.0,
                              max_value=5.0, value=0.10, step=0.01) / 100.0

    st.markdown("---")
    st.markdown('<p class="section-header">Training</p>', unsafe_allow_html=True)
    train_steps = st.slider("Training Steps", 2_000, 100_000, 20_000, step=1_000)

    st.markdown("---")
    st.markdown('<p class="section-header">Saved Models</p>', unsafe_allow_html=True)
    valid_models, orphan_models = list_saved_models()
    all_model_options = valid_models + ([f"[legacy] {m}" for m in orphan_models])
    selected_model_name = st.selectbox("Load Model", [""] + all_model_options)

    if orphan_models:
        st.caption(f"⚠️ {len(orphan_models)} legacy model(s) without metadata — may fail to load.")

    # Show metadata of selected model
    if selected_model_name and not selected_model_name.startswith("[legacy]"):
        meta = load_model_meta(selected_model_name)
        if meta:
            st.markdown(f"""
            <div class="model-card">
              <strong>{meta.get('ticker','?')}</strong> &nbsp;·&nbsp; {meta.get('train_end_date','?')}<br>
              Window: {meta.get('window_size','?')} &nbsp;|&nbsp; Steps: {meta.get('train_steps','?'):,}<br>
              Tx cost: {float(meta.get('tx_cost', 0))*100:.2f}%
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    run_button = st.button("▶  Run", use_container_width=True)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">RL Trader Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Reinforcement Learning · PPO · Educational Trading Dashboard</p>',
            unsafe_allow_html=True)

if not run_button:
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color:#1e2d4a;">
      <div style="font-size:5rem;">📈</div>
      <p style="font-family:Space Mono,monospace;font-size:14px;color:#334155;margin-top:16px;">
        Configure parameters in the sidebar and click <strong>Run</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
with st.spinner(f"Fetching market data for **{ticker}**…"):
    try:
        df = get_data(ticker, period=period)
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

st.success(f"✅ Loaded **{len(df):,}** trading days for **{ticker}**  "
           f"({df['Date'].min().date()} → {df['Date'].max().date()})")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

# ══════════════════════════════════════════════
# MODE: Train + Backtest
# ══════════════════════════════════════════════
if mode == "Train + Backtest":

    default_train_end = df["Date"].iloc[int(len(df) * 0.8)].date()
    train_end_date = st.date_input(
        "Train / Test Split Date",
        value=default_train_end,
        min_value=min_date,
        max_value=max_date,
        help="Data before this date is used for training; after for backtesting.",
    )

    train_df = df[df["Date"].dt.date <= train_end_date].copy()
    test_df  = df[df["Date"].dt.date > train_end_date].copy()

    if len(train_df) < window_size + 30:
        st.error("Training slice too small — move the split date earlier.")
        st.stop()
    if len(test_df) < window_size + 10:
        st.error("Test slice too small — move the split date earlier.")
        st.stop()

    st.markdown(f"**Train:** {len(train_df):,} rows  |  **Test:** {len(test_df):,} rows")

    left, right = st.columns([3, 1.2])

    with right:
        st.markdown('<p class="section-header">Live Trade Log</p>', unsafe_allow_html=True)
        right_log_placeholder = st.empty()

    with left:
        # ── Build / load model ──
        model_path = model_filename(ticker, window_size, tx_cost, train_steps, str(train_end_date))
        zip_path = model_path.with_suffix(".zip")
        train_env = TradingEnv(train_df, window_size=window_size,
                               initial_cash=initial_cash,
                               transaction_cost_pct=tx_cost,
                               feature_columns=FEATURE_COLUMNS)

        status_text = st.empty()
        progress_bar = st.progress(0)

        if zip_path.exists() and meta_path(model_path).exists():
            status_text.info("♻️ Loading cached model…")
            model = PPO.load(str(model_path), env=train_env)
            progress_bar.progress(50)
            status_text.success("✅ Cached model loaded.")
        else:
            status_text.info("🧠 Training PPO model…")
            model = PPO(
                "MlpPolicy", train_env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
            )
            cb = ProgressCallback(train_steps, progress_bar, status_text)
            model.learn(total_timesteps=train_steps, callback=cb)
            progress_bar.progress(90)
            model.save(str(model_path))
            # ← Save metadata so Paper Trading mode can reconstruct env correctly
            meta = {
                "ticker": ticker,
                "period": period,
                "window_size": window_size,
                "train_steps": train_steps,
                "tx_cost": tx_cost,
                "train_end_date": str(train_end_date),
                "feature_columns": FEATURE_COLUMNS,
                "obs_shape": list(train_env.observation_space.shape),
                "saved_at": datetime.now().isoformat(),
            }
            save_model_meta(model_path, meta)
            progress_bar.progress(100)
            status_text.success(f"✅ Model trained & saved: `{model_path.name}.zip`")

        # ── Backtest ──
        test_env = TradingEnv(test_df, window_size=window_size,
                              initial_cash=initial_cash,
                              transaction_cost_pct=tx_cost,
                              feature_columns=FEATURE_COLUMNS)
        with st.spinner("Running backtest…"):
            info, logs_df, trades_df = run_env(model, test_env, right_log_placeholder)

        # ── Metrics ──
        test_prices = test_df["Close"].reset_index(drop=True)
        buy_hold_return = float((test_prices.iloc[-1] / test_prices.iloc[0]) - 1.0) \
            if len(test_prices) >= 2 else 0.0
        strategy_return = float((info["portfolio_value"] / initial_cash) - 1.0)
        mdd = info["max_drawdown"]
        sharpe = info["sharpe_ratio"]
        calmar = info["calmar_ratio"]
        wr = win_rate(logs_df) if not logs_df.empty else 0.0
        alpha = strategy_return - buy_hold_return

        st.markdown("---")
        st.markdown('<p class="section-header">Performance Metrics</p>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Final Portfolio",   f"${info['portfolio_value']:,.2f}")
        c2.metric("Strategy Return",   f"{strategy_return:.2%}",
                  delta=f"α {alpha:+.2%} vs B&H")
        c3.metric("Buy & Hold",        f"{buy_hold_return:.2%}")
        c4.metric("Max Drawdown",      f"{mdd:.2%}")
        c5.metric("Sharpe Ratio",      f"{sharpe:.2f}")
        c6.metric("Win Rate",          f"{wr:.1%}")

        st.markdown("---")
        # Charts
        st.plotly_chart(
            plot_candlestick(test_df, trades_df, f"{ticker} — Backtest"),
            use_container_width=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_equity(test_env.equity_curve, initial_cash, "Equity Curve"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                plot_returns_dist(test_env.portfolio_returns, "Return Distribution"),
                use_container_width=True,
            )

        # Drawdown chart
        eq = pd.Series(test_env.equity_curve)
        dd_series = ((eq - eq.cummax()) / (eq.cummax() + 1e-12)) * 100
        dd_fig = go.Figure(go.Scatter(
            x=list(range(len(dd_series))), y=dd_series,
            mode="lines", fill="tozeroy",
            line=dict(color="#f43f5e", width=1.5),
            fillcolor="rgba(244,63,94,0.12)",
            name="Drawdown %",
        ))
        dd_fig.update_layout(
            title=dict(text="Drawdown (%)", font=dict(family="Syne,sans-serif", size=16,
                                                       color="#e2e8f0")),
            height=260, xaxis_title="Step", yaxis_title="Drawdown (%)",
            **CHART_TEMPLATE,
        )
        st.plotly_chart(dd_fig, use_container_width=True)

        # Trade log table
        st.markdown("---")
        st.markdown('<p class="section-header">Full Trade Log</p>', unsafe_allow_html=True)
        st.dataframe(logs_df, use_container_width=True, height=360)

        st.download_button(
            "⬇ Download Trade Log CSV",
            logs_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════
# MODE: Paper Trading  ← MAIN BUG FIX HERE
# ══════════════════════════════════════════════
else:
    if not selected_model_name:
        st.warning("⚠️ Select a saved model in the sidebar to run paper trading.")
        st.stop()

    is_legacy = selected_model_name.startswith("[legacy]")
    clean_name = selected_model_name.replace("[legacy] ", "")

    # ── Load metadata to reconstruct env with CORRECT parameters ──
    meta = load_model_meta(clean_name) if not is_legacy else None

    if meta is None and not is_legacy:
        st.error("Metadata file missing for this model. Please retrain it.")
        st.stop()

    if meta:
        # Use the EXACT parameters the model was trained with
        model_window_size    = int(meta["window_size"])
        model_tx_cost        = float(meta["tx_cost"])
        model_feature_cols   = meta.get("feature_columns", FEATURE_COLUMNS)
        model_ticker         = meta.get("ticker", ticker)
        expected_obs_shape   = tuple(meta["obs_shape"])
        st.info(
            f"🔗 Model trained on **{model_ticker}** | "
            f"window={model_window_size} | "
            f"tx={model_tx_cost*100:.2f}% | "
            f"obs shape={expected_obs_shape}"
        )
    else:
        # Legacy model — use sidebar params (may fail)
        model_window_size  = window_size
        model_tx_cost      = tx_cost
        model_feature_cols = FEATURE_COLUMNS
        st.warning("⚠️ Legacy model — using sidebar params. May fail if window size differs.")

    paper_start_default = max(
        min_date,
        df["Date"].iloc[max(model_window_size + 1, len(df) - 90)].date(),
    )
    paper_start_date = st.date_input(
        "Paper Trade Start Date",
        value=paper_start_default,
        min_value=min_date,
        max_value=max_date,
    )

    paper_df = df[df["Date"].dt.date >= paper_start_date].copy()

    if len(paper_df) < model_window_size + 10:
        st.error("Paper trading date range is too short.")
        st.stop()

    left, right = st.columns([3, 1.2])

    with right:
        st.markdown('<p class="section-header">Live Trade Log</p>', unsafe_allow_html=True)
        right_log_placeholder = st.empty()

    with left:
        # Build env with EXACTLY the same parameters as training
        env = TradingEnv(
            paper_df,
            window_size=model_window_size,
            initial_cash=initial_cash,
            transaction_cost_pct=model_tx_cost,
            feature_columns=model_feature_cols,
        )

        model_file = MODELS_DIR / clean_name
        try:
            model = PPO.load(str(model_file), env=env)
        except ValueError as e:
            st.error(
                f"❌ Observation space mismatch: {e}\n\n"
                "This usually means the model was saved with different settings. "
                "Please retrain with the current parameters."
            )
            st.stop()

        with st.spinner("Running paper trading simulation…"):
            info, logs_df, trades_df = run_env(model, env, right_log_placeholder)

        strategy_return = float((info["portfolio_value"] / initial_cash) - 1.0)
        mdd = info["max_drawdown"]
        sharpe = info["sharpe_ratio"]
        wr = win_rate(logs_df) if not logs_df.empty else 0.0

        # Metrics
        st.markdown("---")
        st.markdown('<p class="section-header">Paper Trading Metrics</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Portfolio", f"${info['portfolio_value']:,.2f}")
        c2.metric("Paper Return",    f"{strategy_return:.2%}")
        c3.metric("Max Drawdown",    f"{mdd:.2%}")
        c4.metric("Sharpe Ratio",    f"{sharpe:.2f}")

        st.markdown("---")
        st.plotly_chart(
            plot_candlestick(paper_df, trades_df, f"{ticker} — Paper Trading"),
            use_container_width=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_equity(env.equity_curve, initial_cash, "Equity Curve"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                plot_returns_dist(env.portfolio_returns, "Return Distribution"),
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown('<p class="section-header">Paper Trading Log</p>', unsafe_allow_html=True)
        st.dataframe(logs_df, use_container_width=True, height=360)

        st.download_button(
            "⬇ Download Paper Trading Log CSV",
            logs_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
