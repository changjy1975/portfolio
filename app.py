import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
from datetime import datetime
import numpy as np

# --- 檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="個人投資組合戰情室", layout="wide")
st.title("📈 智能投資組合戰情室")

# ==========================================
# 1. 數據與報價工具
# ==========================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def analyze_stock_technical(symbol):
    """AI 技術診斷邏輯"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", interval="1d")
        if df.empty: return None, "無法獲取歷史資料"
        
        curr_p = float(df['Close'].iloc[-1])
        ma_20 = float(df['Close'].rolling(window=20).mean().iloc[-1])
        rsi_curr = float(calculate_rsi(df['Close'], 14).iloc[-1])
        df_6m = df.tail(126)
        h_6m, l_6m = float(df_6m['High'].max()), float(df_6m['Low'].min())
        
        trend = "多頭排列 🐂" if curr_p > ma_20 else "空頭/整理 🐻"
        if rsi_curr > 70: advice, color = "過熱，建議減碼", "red"
        elif rsi_curr < 30: advice, color = "超賣，建議佈局", "green"
        else: advice, color = "趨勢持平", "orange"

        return {
            "current_price": curr_p, "high_6m": h_6m, "low_6m": l_6m,
            "ma_20": ma_20, "rsi": rsi_curr, "trend": trend,
            "entry_target": l_6m * 1.05, "exit_target": h_6m * 0.95,
            "advice": advice, "advice_color": color, "df": df.tail(100)
        }, None
    except Exception as e: return None, str(e)

def get_current_prices(symbols):
    prices = {}
    if not symbols: return prices
    for symbol in symbols:
        try:
            t = yf.Ticker(symbol)
            p = t.fast_info.last_price
            if p is None or pd.isna(p) or p <= 0:
                hist = t.history(period="1d")
                p = hist['Close'].iloc[-1] if not hist.empty else 0.0
            prices[symbol] = float(p)
        except: prices[symbol] = 0.0
    return prices

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["股票代號"] = df["股票代號"].astype(str)
        df["股數"] = pd.to_numeric(df["股數"], errors='coerce').fillna(0)
        df["持有成本單價"] = pd.to_numeric(df["持有成本單價"], errors='coerce').fillna(0)
        return df
    return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df): df.to_csv(DATA_FILE, index=False)

def get_exchange_rate():
    try:
        rate = yf.Ticker("USDTWD=X").fast_info.last_price
        return float(rate) if rate and not pd.isna(rate) else 32.5
    except: return 32.5

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# ==========================================
# 2. UI 渲染組件
# ==========================================

COLS_RATIO = [1.3, 0.8, 0.9, 0.9, 1.2, 1.2, 1.2, 0.9, 0.6]

def display_headers():
    cols = st.columns(COLS_RATIO)
    labels = ["代號", "股數", "均價", "現價", "成本(原)", "現值(原)", "獲利(原)", "報酬率", "管理"]
    for col, label in zip(cols, labels): col.markdown(f"**{label}**")
    st.markdown("---")

def display_stock_rows(df):
    for _, row in df.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        sym = str(row["股票代號"])
        prof, roi = float(row["獲利(原幣)"]), float(row["獲利率(%)"])
        color = "red" if prof > 0 else "green"
        fmt = "{:,.0f}" if row["幣別"] == "TWD" else "{:,.2f}"
        
        c1.write(f"**{sym}**")
        c2.write(f"{float(row['股數']):.2f}")
        c3.write(f"{float(row['平均持有單價']):.2f}")
        c4.write(f"{float(row['最新股價']):.2f}")
        c5.write(fmt.format(float(row["總投入成本(原幣)"])))
        c6.write(fmt.format(float(row["現值(原幣)"])))
        c7.markdown(f":{color}[{fmt.format(prof)}]")
        c8.markdown(f":{color}[{roi:.2f}%]")
        if c9.button("🗑️", key=f"del_{sym}"):
            df_old = load_data(); df_old = df_old[df_old["股票代號"] != sym]; save_data(df_old); st.rerun()

def display_subtotal_row(df, label):
    if df.empty: return
    t_cost, t_val = float(df["總投入成本(原幣)"].sum()), float(df["現值(原幣)"].sum())
    t_prof = t_val - t_cost
    t_roi = (t_prof / t_cost * 100) if t_cost != 0 else 0
    fmt = "{:,.0f}" if df["幣別"].iloc[0] == "TWD" else "{:,.2f}"
    st.markdown("---")
    c1, _, _, _, c5, c6, c7, c8, _ = st.columns(COLS_RATIO)
    c1.markdown(f"**🔹 {label}**")
    c5.markdown(f"**{fmt.format(t_cost)}**")
    c6.markdown(f"**{fmt.format(t_val)}**")
    c7.markdown(f":{'red' if t_prof > 0 else 'green'}[**{fmt.format(t_prof)}**]")
    c8.markdown(f":{'red' if t_prof > 0 else 'green'}[**{t_roi:.2f}%**]")

# ==========================================
# 3. 主程式邏輯
# ==========================================

tab1, tab2 = st.tabs(["📊 庫存配置", "🧠 AI 持股診斷"])
df_raw = load_data()

# 資料預處理
if not df_raw.empty:
    usd_rate = get_exchange_rate()
    df_raw["單筆成本"] = df_raw["股數"] * df_raw["持有成本單價"]
    portfolio = df_raw.groupby("股票代號").agg({"股數":"sum", "單筆成本":"sum"}).reset_index()
    portfolio["平均持有單價"] = portfolio["單筆成本"] / portfolio["股數"]
    portfolio.rename(columns={"單筆成本": "總投入成本(原幣)"}, inplace=True)
    
    prices = get_current_prices(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(prices).astype(float)
    portfolio["幣別"] = portfolio["股票代號"].apply(identify_currency)
    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    portfolio["獲利率(%)"] = portfolio.apply(lambda r: (r["獲利(原幣)"]/r["總投入成本(原幣)"]*100) if r["總投入成本(原幣)"] != 0 else 0, axis=1)
    
    # 統一換算台幣
    portfolio["現值(TWD)"] = portfolio.apply(lambda r: r["現值(原幣)"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)
    portfolio["總成本(TWD)"] = portfolio.apply(lambda r: r["總投入成本(原幣)"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)
    portfolio["獲利(TWD)"] = portfolio["現值(TWD)"] - portfolio["總成本(TWD)"]

# --- Tab 1: 庫存配置與儀表板 ---
with tab1:
    with st.sidebar:
        st.header("📝 新增投資紀錄")
        with st.form("add_form", clear_on_submit=True):
            s_in = st.text_input("代號 (如: 2330.TW, TSLA)", "").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, value=0.0)
            c_in = st.number_input("買入單價", min_value=0.0, value=0.0)
            if st.form_submit_button("新增標的"):
                if s_in and q_in > 0:
                    save_data(pd.concat([load_data(), pd.DataFrame([{"股票代號":s_in, "股數":q_in, "持有成本單價":c_in}])], ignore_index=True)); st.rerun()

    if df_raw.empty:
        st.info("尚無持股資料。")
    else:
        # A. 核心儀表板
        total_val = float(portfolio['現值(TWD)'].sum())
        total_profit = float(portfolio['獲利(TWD)'].sum())
        total_cost = float(portfolio['總成本(TWD)'].sum())
        total_roi = (total_profit / total_cost * 100) if total_cost != 0 else 0
        
        

[Image of stock market dashboard metrics]


        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 總資產 (TWD)", f"${total_val:,.0f}")
        m2.metric("📈 總獲利 (TWD)", f"${total_profit:,.0f}", f"{total_profit:,.0f}")
        m3.metric("📊 總報酬率", f"{total_roi:.2f}%")
        m4.metric("💱 最新匯率 (USD/TWD)", f"{usd_rate:.2f}")
        
        st.divider()
        
        # B. 圓餅圖區
        st.subheader("📊 資產分佈分析")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 🔹 台美股配置比例")
            # 依幣別加總
            category_df = portfolio.groupby("幣別")["現值(TWD)"].sum().reset_index()
            category_df["類別"] = category_df["幣別"].map({"TWD": "台股 (TWD)", "USD": "美股 (USD)"})
            fig_cat = px.pie(category_df, values="現值(TWD)", names="類別", hole=0.4,
                             color_discrete_sequence=["#1f77b4", "#ff7f0e"])
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_chart2:
            st.markdown("#### 🔹 個股權重佔比")
            chart_view = st.selectbox("圖表過濾", ["全部資產", "僅限台股", "僅限美股"], label_visibility="collapsed")
            df_plt = portfolio if chart_view == "全部資產" else (portfolio[portfolio["幣別"]=="TWD"] if chart_view=="僅限台股" else portfolio[portfolio["幣別"]=="USD"])
            if not df_plt.empty:
                fig_pie = px.pie(df_plt, values="現值(TWD)", names="股票代號", hole=0.4)
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.write("無資料")
        
        
        
        # C. 分區明細
        st.divider()
        df_tw, df_us = portfolio[portfolio["幣別"]=="TWD"], portfolio[portfolio["幣別"]=="USD"]
        if not df_tw.empty:
            st.subheader("🇹🇼 台股明細")
            display_headers(); display_stock_rows(df_tw); display_subtotal_row(df_tw, "台股小計")

        if not df_us.empty:
            st.subheader("🇺🇸 美股明細")
            display_headers(); display_stock_rows(df_us); display_subtotal_row(df_us, "美股小計")

# --- Tab 2: AI 持股健診 ---
with tab2:
    if df_raw.empty:
        st.info("請先新增標的。")
    else:
        st.subheader("🧠 AI 持股技術面診斷")
        sel_s = st.selectbox("選擇要健診的股票：", portfolio["股票代號"].tolist())
        if st.button("🚀 啟動深度診斷"):
            res, err = analyze_stock_technical(sel_s)
            if err: st.error(err)
            else:
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("目前價格", f"${res['current_price']:.2f}")
                c2.metric("半年高 (壓力)", f"${res['high_6m']:.2f}")
                c3.metric("半年低 (支撐)", f"${res['low_6m']:.2f}")
                c4.metric("RSI 指標", f"{res['rsi']:.1f}")
                st.markdown(f"### 💡 診斷建議：:{res['advice_color']}[{res['advice']}]")
                st.info(f"趨勢：{res['trend']} | 進場參考：${res['entry_target']:.2f} | 減碼參考：${res['exit_target']:.2f}")
                st.line_chart(res['df']['Close'])
