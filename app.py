import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
from datetime import datetime
import pytz
import numpy as np
from scipy.optimize import minimize

# --- 設定檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="個人投資組合戰情室", layout="wide")
st.title("📈 智能投資組合戰情室")

# ==========================================
# 核心數學模型 (Modern Portfolio Theory)
# ==========================================

def calculate_mpt_optimization(returns_df):
    """執行 MPT 優化計算：最小波動與最高夏普比率"""
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    num_assets = len(mean_returns)
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def neg_sharpe_ratio(weights, risk_free_rate=0.02):
        p_ret = np.sum(mean_returns * weights)
        p_vol = portfolio_volatility(weights)
        return -(p_ret - risk_free_rate) / (p_vol + 1e-9)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    min_vol_res = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    max_sharpe_res = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return {
        "symbols": list(returns_df.columns),
        "min_vol_weights": min_vol_res.x,
        "max_sharpe_weights": max_sharpe_res.x
    }

# ==========================================
# 基礎數據處理函數
# ==========================================

def load_data():
    if os.path.exists(DATA_FILE): return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df): df.to_csv(DATA_FILE, index=False)

def get_exchange_rate():
    try:
        rate = yf.Ticker("USDTWD=X").fast_info.last_price
        return rate if rate and not pd.isna(rate) else 32.5
    except: return 32.5

def get_current_prices(symbols):
    prices = {}
    if not symbols: return prices
    for symbol in symbols:
        try:
            t = yf.Ticker(symbol)
            p = t.fast_info.last_price
            if p is None or pd.isna(p):
                p = t.history(period="1d")['Close'].iloc[-1]
            prices[symbol] = p
        except: prices[symbol] = None
    return prices

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# ==========================================
# 技術分析邏輯 (Tab 2)
# ==========================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def analyze_stock_technical(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", interval="1wk")
        if df.empty: return None, "無法獲取歷史資料"
        df_recent = df.tail(26) 
        current_price = df['Close'].iloc[-1]
        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        rsi_series = calculate_rsi(df['Close'], 14)
        rsi_curr = rsi_series.iloc[-1]
        
        trend = "多頭排列 🐂" if current_price > ma_20 else "空頭/整理 🐻"
        if rsi_curr > 70: advice, color = "過熱，建議分批獲利", "red"
        elif rsi_curr < 30: advice, color = "超賣，可考慮分批佈局", "green"
        else: advice, color = "趨勢持平，觀望或持股續抱", "orange"

        return {
            "current_price": current_price, "high_6m": df_recent['High'].max(), "low_6m": df_recent['Low'].min(),
            "ma_20": ma_20, "rsi": rsi_curr, "trend": trend,
            "entry_target": df_recent['Low'].min() * 1.02, "exit_target": df_recent['High'].max() * 0.98,
            "advice": advice, "advice_color": color, "history_df": df_recent
        }, None
    except Exception as e: return None, str(e)

# ==========================================
# 介面渲染與排版
# ==========================================

COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def display_stock_rows(df, currency_type):
    for _, row in df.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        sym = row["股票代號"]
        prof, roi = row["獲利(原幣)"], row["獲利率(%)"]
        color = "red" if prof > 0 else "green"
        fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"

        c1.write(f"**{sym}**")
        c2.write(f"{row['股數']:.2f}")
        c3.write(f"{row['平均持有單價']:.2f}")
        c4.write(f"{row['最新股價']:.2f}")
        c5.write(fmt.format(row["總投入成本(原幣)"]))
        c6.write(fmt.format(row["現值(原幣)"]))
        c7.markdown(f":{color}[{fmt.format(prof)}]")
        c8.markdown(f":{color}[{roi:.2f}%]")
        if c9.button("🗑️", key=f"del_{sym}"):
            df_old = load_data()
            df_old = df_old[df_old["股票代號"] != sym]
            save_data(df_old); st.rerun()

# ==========================================
# 主程式邏輯
# ==========================================

tab1, tab2, tab3 = st.tabs(["📊 庫存資產配置", "🧠 AI 持股健診", "⚖️ MPT 數學模擬器"])
df_record = load_data()

with tab1:
    with st.sidebar:
        st.header("📝 新增投資")
        with st.form("add_form"):
            s_in = st.text_input("股票代號", "2330.TW").upper().strip()
            q_in = st.number_input("股數", value=100.0)
            c_in = st.number_input("成本單價", value=600.0)
            if st.form_submit_button("新增"):
                new_df = pd.concat([load_data(), pd.DataFrame([{"股票代號":s_in, "股數":q_in, "持有成本單價":c_in}])])
                save_data(new_df); st.rerun()

    if df_record.empty:
        st.info("請先新增股票。")
    else:
        usd_rate = get_exchange_rate()
        unique_syms = df_record["股票代號"].unique().tolist()
        prices = get_current_prices(unique_syms)
        
        # 關鍵處理：整合資料並修正 KeyError 欄位名稱
        df_record["總投入成本(原幣)"] = df_record["股數"] * df_record["持有成本單價"]
        portfolio = df_record.groupby("股票代號").agg({"股數": "sum", "總投入成本(原幣)": "sum"}).reset_index()
        portfolio["平均持有單價"] = portfolio["總投入成本(原幣)"] / portfolio["股數"]
        portfolio["最新股價"] = portfolio["股票代號"].map(prices)
        portfolio = portfolio.dropna(subset=["最新股價"])
        portfolio["幣別"] = portfolio["股票代號"].apply(identify_currency)
        portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
        portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
        portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
        portfolio["現值(TWD)"] = portfolio.apply(lambda r: r["現值(原幣)"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

        # 頂部看板
        t_val = portfolio["現值(TWD)"].sum()
        st.metric("💰 總資產 (TWD)", f"${t_val:,.0f}")
        
        st.subheader("📦 詳細持股清單")
        display_stock_rows(portfolio, "MIX")

with tab2:
    if not df_record.empty:
        sel_sym = st.selectbox("選擇分析標的", portfolio["股票代號"].tolist())
        if st.button("🚀 開始診斷"):
            res, err = analyze_stock_technical(sel_sym)
            if err: st.error(err)
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("目前價格", f"{res['current_price']:.2f}")
                c2.metric("RSI 指標", f"{res['rsi']:.1f}")
                c3.metric("半年高點", f"{res['high_6m']:.2f}")
                st.success(f"**綜合點評**：:{res['advice_color']}[{res['advice']}]")
                st.line_chart(res['history_df']['Close'])

with tab3:
    st.subheader("⚖️ MPT 權重優化模擬")
    if not df_record.empty and len(portfolio) >= 2:
        if st.button("🚀 啟動模擬"):
            with st.spinner("計算中..."):
                symbols = portfolio["股票代號"].tolist()
                hist = yf.download(symbols, period="3y", interval="1d", auto_adjust=True)['Close']
                if isinstance(hist, pd.Series): hist = hist.to_frame(name=symbols[0])
                returns = hist.ffill().pct_change().dropna()
                mpt = calculate_mpt_optimization(returns)
                
                # 對比表格
                total_v = portfolio["現值(TWD)"].sum()
                curr_w = {row["股票代號"]: (row["現值(TWD)"]/total_v)*100 for _, row in portfolio.iterrows()}
                comp_df = pd.DataFrame({
                    "標的": mpt['symbols'],
                    "目前權重 (%)": [curr_w.get(s, 0) for s in mpt['symbols']],
                    "最小波動建議 (%)": mpt['min_vol_weights'] * 100,
                    "最高夏普建議 (%)": mpt['max_sharpe_weights'] * 100
                })
                st.dataframe(comp_df.style.format("{:.2f}%"), use_container_width=True, hide_index=True)
                
                
                
                st.markdown("### 🔗 相關性矩陣 (風險分散檢查)")
                st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
    else:
        st.warning("請至少加入 2 支股票進行分析。")
