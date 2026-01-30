import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import shutil
import json
from datetime import datetime
import pytz
import numpy as np

# ==========================================
# 1. 初始化設定與狀態管理
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

# 狀態持久化
if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None
if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心功能函數 (資料、行情與財務計算)
# ==========================================

def load_data(user):
    path = f"portfolio_{user}.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    source_path = f"portfolio_{user}.csv"
    if os.path.exists(source_path):
        now = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y%m%d_%H%M%S")
        shutil.copy2(source_path, os.path.join(BACKUP_DIR, f"backup_{user}_{now}.csv"))
    df.to_csv(source_path, index=False)

def load_financial_config(user):
    path = f"financial_config_{user}.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except: pass
    return {
        "cash_res": 500000.0,
        "l1_p": 3000000.0, "l1_r": 2.65, "l1_y": 30, "l1_m": 12,
        "l2_p": 0.0, "l2_r": 3.5, "l2_y": 7, "l2_m": 0,
        "pledge_loan": 0.0, "pledge_targets": []
    }

def save_financial_config(user, config):
    path = f"financial_config_{user}.json"
    with open(path, "w") as f:
        json.dump(config, f)

@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        rate = yf.Ticker("USDTWD=X").fast_info.last_price
        return float(rate) if rate else 32.5
    except: return 32.5

@st.cache_data(ttl=300)
def get_latest_quotes(symbols):
    if not symbols: return {}
    quotes = {}
    try:
        tickers = yf.Tickers(" ".join(symbols))
        for s in symbols:
            try:
                price = tickers.tickers[s].fast_info.last_price
                if price is None or np.isnan(price):
                    price = tickers.tickers[s].history(period="1d")['Close'].iloc[-1]
                quotes[s] = float(price)
            except: quotes[s] = 0.0
        return quotes
    except: return {s: 0.0 for s in symbols}

def calculate_remaining_loan(principal, annual_rate, years, months_passed):
    if principal <= 0 or annual_rate <= 0 or years <= 0: return 0.0
    r = annual_rate / 12 / 100
    n = years * 12
    if months_passed >= n: return 0.0
    remaining = principal * ((1 + r)**n - (1 + r)**months_passed) / ((1 + r)**n - 1)
    return float(remaining)

# --- 技術指標計算 ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean(); exp2 = series.ewm(span=26, adjust=False).mean()
    m = exp1 - exp2; sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m - sig

def calculate_bb(series, window=20):
    ma = series.rolling(window=window).mean(); std = series.rolling(window=window).std()
    return ma + (std * 2), ma, ma - (std * 2)

# ==========================================
# 3. MPT 模擬引擎
# ==========================================

def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "標的不足。"
    try:
        data = yf.download(symbols, period="3y", interval="1d", auto_adjust=True)
        close = data['Close'] if len(symbols) > 1 else data['Close'].to_frame(name=symbols[0])
        rets = close.ffill().pct_change().dropna()
        m_rets = rets.mean() * 252; c_mat = rets.cov() * 252
        res = np.zeros((3, 2000)); w_rec = []
        for i in range(2000):
            w = np.random.random(len(symbols)); w /= np.sum(w); w_rec.append(w)
            p_r = np.sum(w * m_rets); p_s = np.sqrt(np.dot(w.T, np.dot(c_mat, w)))
            res[0,i] = p_r; res[1,i] = p_s; res[2,i] = (p_r - 0.02) / p_s
        idx = np.argmax(res[2]); curr_w = portfolio_df["現值_TWD"].values / portfolio_df["現值_TWD"].sum()
        comp = pd.DataFrame({"股票代號": symbols, "目前權重 (%)": curr_w * 100, "建議權重 (%)": w_rec[idx] * 100})
        return {"sim_df": pd.DataFrame({'Return': res[0], 'Volatility': res[1], 'Sharpe': res[2]}), 
                "comparison": comp, "max_sharpe": (res[0, idx], res[1, idx]), "corr": rets.corr()}, None
    except Exception as e: return None, str(e)

# ==========================================
# 4. 介面表格組件 (具排序功能)
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, title, currency, usd_rate, user):
    st.subheader(title)
    h_map = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本"), ("現值", "現值"), ("獲利", "獲利"), ("報酬率", "獲利率(%)")]
    h_cols = st.columns(COLS_RATIO)
    for i, (label, col_name) in enumerate(h_map):
        arrow = " ▲" if st.session_state.sort_col == col_name and st.session_state.sort_asc else " ▼" if st.session_state.sort_col == col_name else ""
        if h_cols[i].button(f"{label}{arrow}", key=f"h_{currency}_{col_name}_{user}"):
            if st.session_state.sort_col == col_name: st.session_state.sort_asc = not st.session_state.sort_asc
            else: st.session_state.sort_col, st.session_state.sort_asc = col_name, False
            st.rerun()
    h_cols[8].write("**管理**")
    for _, row in df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc).iterrows():
        r = st.columns(COLS_RATIO); fmt = "{:,.0f}" if currency == "TWD" else "{:
