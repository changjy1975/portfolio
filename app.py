import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
from datetime import datetime
import pytz

# --- 設定檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="台美股投資戰情室", layout="wide")
st.title("📈 智能投資組合戰情室")

# ==========================================
# 狀態初始化
# ==========================================
if "sort_col" not in st.session_state:
    st.session_state.sort_col = "獲利(原幣)"
if "sort_asc" not in st.session_state:
    st.session_state.sort_asc = False
if "last_updated" not in st.session_state:
    st.session_state.last_updated = "尚未更新"

# ==========================================
# 核心功能函數
# ==========================================

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def remove_stock(symbol):
    df = load_data()
    df = df[df["股票代號"] != symbol]
    save_data(df)

def get_exchange_rate():
    try:
        ticker = yf.Ticker("USDTWD=X")
        rate = ticker.fast_info.last_price
        if rate is None or pd.isna(rate):
             rate = ticker.history(period="1d")['Close'].iloc[-1]
        return rate
    except:
        return 32.5

def get_current_prices(symbols):
    if not symbols: return {}
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            price = None
            try:
                price = ticker.fast_info.last_price
            except:
                price = None

            if price is None or pd.isna(price):
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
            
            if price is None or pd.isna(price):
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPreviousClose') or info.get('previousClose')
            
            prices[symbol] = price
        except:
            prices[symbol] = None
    return prices

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# ==========================================
# 技術分析邏輯
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stock_technical(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", interval="1wk")
        if df.empty: return None, "無法獲取歷史資料"
        df_recent = df.tail(26) 
        current_price = df['Close'].iloc[-1]
        high_6m = df_recent['High'].max()
        low_6m = df_recent['Low'].min()
        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        rsi_series = calculate_rsi(df['Close'], 14)
        rsi_curr = rsi_series.iloc[-1]
        trend = "多頭排列 🐂" if current_price > ma_20 else "空頭/整理 🐻"
        entry_price = max(low_6m * 1.02, ma_20)
        exit_price = high_6m * 0.98
        
        if rsi_curr > 70: advice, color = "過熱，建議分批獲利", "red"
        elif rsi_curr < 30: advice, color = "超賣，可考慮分批佈局", "green"
        elif current_price > ma_20: advice, color = "趨勢向上，持股續抱", "orange"
        else: advice, color = "趨勢偏弱，觀望或區間操作", "gray"

        return {
            "current_price": current_price, "high_6m": high_6m, "low_6m": low_6m,
            "ma_20": ma_20, "rsi": rsi_curr, "trend": trend,
            "entry_target": entry_price, "exit_target": exit_price,
            "advice": advice, "advice_color": color, "history_df": df_recent
        }, None
    except Exception as e:
        return None, str(e)

# ==========================================
# 介面顯示組件
# ==========================================

COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def update_sort(column_name):
    if st.session_state.sort_col == column_name:
        st.session_state.sort_asc = not st.session_state.sort_asc
    else:
        st.session_state.sort_col = column_name
        st.session_state.sort_asc = False

def get_header_label(label, col_name):
    if st.session_state.sort_col == col_name:
        arrow = "▲" if st.session_state.sort_asc else "▼"
        return f"{label} {arrow}"
    return label

def display_headers(key_suffix):
    st.markdown("<div style='padding-right: 15px;'>", unsafe_allow_html=True) 
    cols = st.columns(COLS_RATIO)
    headers_map = [
        ("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), 
        ("現價", "最新股價"), ("總成本", "總投入成本(原幣)"), 
        ("現值", "現值(原幣)"), ("獲利", "獲利(原幣)"), ("報酬率%", "獲利率(%)")
    ]
    for col, (label, col_name) in zip(cols[:-1], headers_map):
        if col.button(get_header_label(label, col_name), key=f"btn_head_{col_name}_{key_suffix}"):
            update_sort(col_name)
            st.rerun()
