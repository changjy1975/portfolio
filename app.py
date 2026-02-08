import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import shutil
from datetime import datetime
import numpy as np

# ==========================================
# 1. 初始化設定與全域配置
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

# 初始化 Session State
if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None
if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 數據核心函數
# ==========================================

def load_data(user):
    path = f"portfolio_{user}.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    source_path = f"portfolio_{user}.csv"
    if os.path.exists(source_path):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(source_path, os.path.join(BACKUP_DIR, f"backup_{user}_{now}.csv"))
    df.to_csv(source_path, index=False)

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

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# --- 技術指標與精確訊號計算 ---
def calculate_indicators(df):
    # 移動平均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # 新增 EMA 指標
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI 指標
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    # MACD 指標
    e1, e2 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = e1 - e2
    df['MACD_S'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_H'] = df['MACD'] - df['MACD_S']
    
    # KD 指標
    l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
    rsv = (df['Close'] - l9) / (h9 - l9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean(); df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    
    # ATR 波動率
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

def get_refined_signals(df):
    """精準訊號濾鏡：趨勢與動能雙重確認"""
    m_gold = (df['MACD'] > df['MACD_S']) & (df['MACD'].shift(1) <= df['MACD_S'].shift(1))
    m_dead = (df['MACD'] < df['MACD_S']) & (df['MACD'].shift(1) >= df['MACD_S'].shift(1))
    k_gold = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    buy = ( (df['Close'] > df['MA20']) & (df['MA20'] > df['MA60']) & (m_gold | (k_gold & (df['K'] < 40))) )
    sell = ( (df['Close'] < df['MA5']) & m_dead ) | (df['RSI'] > 78) | ( (df['Close'].shift(1) > df['MA20']) & (df['Close'] < df['MA20']) )
    return buy, sell

# --- 歷史回測與 MPT 引擎 ---
@st.cache_data(ttl=3600
