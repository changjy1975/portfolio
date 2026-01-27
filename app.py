import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
from datetime import datetime
import numpy as np
from scipy.optimize import minimize

# --- 檔案儲存設定 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="個人投資組合戰情室", layout="wide")
st.title("📈 智能投資組合戰情室")

# ==========================================
# 1. 核心分析函數
# ==========================================

def calculate_mpt_optimization(returns_df):
    """執行 MPT 優化計算"""
    returns_df = returns_df.astype(float)
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    num_assets = len(mean_returns)
    
    if mean_returns.isnull().any() or cov_matrix.isnull().any().any():
        return None

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def neg_sharpe_ratio(weights, risk_free_rate=0.02):
        p_ret = np.sum(mean_returns * weights)
        p_vol = portfolio_volatility(weights)
        if p_vol < 1e-9: return 0
        return -(p_ret - risk_free_rate) / p_vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    try:
        min_vol_res = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        max_sharpe_res = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return {
            "symbols": list(returns_df.columns),
            "mean_returns": mean_returns,
            "cov_matrix": cov_matrix,
            "min_vol_weights": min_vol_res.x,
            "max_sharpe_weights": max_sharpe_res.x
        }
    except:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def analyze_stock_technical(symbol):
    """AI 技術診斷"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", interval="1d")
        if df.empty: return None, "無法獲取歷史資料"
        
        current_price = float(df['Close'].iloc[-1])
        ma_20 = float(df['Close'].rolling(window=20).mean().iloc[-1])
        rsi_curr = float(calculate_rsi(df['Close'], 14).iloc[-1])
        df_6m = df.tail(126)
        high_6m, low_6m = float(df_6m['High'].max()), float(df_6m['Low'].min())
        
        trend = "多頭排列 🐂" if current_price > ma_20 else "空頭/整理 🐻"
        if rsi_curr > 70: advice, color = "過熱，建議減碼", "red"
        elif rsi_curr < 30: advice, color = "超賣，建議佈局", "green"
        else: advice, color = "趨勢持平", "orange"

        return {
            "current_price": current_price, "high_6m": high_6m, "low_6m": low_6m,
            "ma_20": ma_20, "rsi": rsi_curr, "trend": trend,
            "entry_target": low_6m * 1.05, "exit_target": high_6m * 0.95,
            "advice": advice, "advice_color": color, "df": df.tail(100)
        }, None
    except Exception as e: return None, str(e)

# ==========================================
# 2. 報價與數據工具 (修復美股報價)
# ==========================================

def get_current_prices(symbols):
    """
    修復版：逐一獲取價格，確保美股在休市期間也能拿到最後收盤價
    """
    prices = {}
    if not symbols: return prices
    
    for symbol in symbols:
        try:
            t = yf.Ticker(symbol)
            # 優先嘗試快速獲取即時價
            p = t.fast_info.last_price
            
            # 如果 fast_info 失敗或回傳 NaN (常發生在美股非開盤時間)
            if p is None or pd.isna(p) or p <= 0:
                hist = t.history(period="1d")
                if not hist.empty:
                    p = hist['Close'].iloc[-1]
                else:
                    # 最後防線：使用 info (較慢但較全)
                    p = t.info.get('regularMarketPrice') or t.info.get('previousClose')
            
            prices[symbol] = float(p) if p else 0.0
        except:
            prices[symbol] = 0.0
    return prices

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["股票代號"] = df["股票代號"].astype(str)
        df["股數"] = pd.to_numeric(df["股數"], errors='coerce')
        df["持有成本單價"] = pd.to_numeric(df["持有成本單價"], errors='coerce')
        return df.dropna()
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
# 3. UI 元件
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
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
    c1.markdown(f"**🔹 {label}**")
    c5.markdown(f"**{fmt.format(t_cost)}**")
    c6.markdown(f"**{fmt.format(t_val)}**")
    c7.markdown(f":{'red' if t_prof > 0 else 'green'}[**{fmt.format(t_prof)}**]")
    c8.markdown(f":{'red' if t_prof > 0 else 'green'}[**{t_roi:.2f}%**]")

# ==========================================
# 4. 主程式
# ==========================================

tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 AI 技術診斷", "⚖️ MPT 數學模擬"])
df_raw = load_data()

if not df_raw.empty:
    usd_rate = get_exchange_rate()
    df_raw["單筆成本"] = df_raw["股數"] * df_raw["持有成本單價"]
    portfolio = df_raw.groupby("股票代號").agg({"股數":"sum", "單筆成本":"sum"}).reset_index()
    portfolio["平均持有單價"] = portfolio["單筆成本"] / portfolio["股數"]
    portfolio.rename(columns={"單筆成本": "總投入成本(原幣)"}, inplace=True)
    
    # 調用修復後的價格獲取函數
    prices = get_current_prices(portfolio["股票代號"].tolist())
    
    portfolio["最新股價"] = portfolio["股票代號"].map(prices).astype(float)
    portfolio["幣別"] = portfolio["股票代號"].apply(identify_currency)
    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    portfolio["獲利率(%)"] = portfolio.apply(lambda r: (r["獲利(原幣)"]/r["總投入成本(原幣)"]*100) if r["總投入成本(原幣)"] != 0 else 0, axis=1)
    portfolio["現值(TWD)"] = portfolio.apply(lambda r: r["現值(原幣)"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

# --- Tab 1 ---
with tab1:
    with st.sidebar:
        st.header("📝 新增投資")
        with st.form("add_form", clear_on_submit=True):
            s_in = st.text_input("代號 (如: 2330.TW, TSLA)", "").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, value=100.0)
            c_in = st.number_input("買入單價", min_value=0.0, value=100.0)
            if st.form_submit_button("新增"):
                if s_in and q_in > 0:
                    save_data(pd.concat([load_data(), pd.DataFrame([{"股票代號":s_in, "股數":q_in, "持有成本單價":c_in}])], ignore_index=True)); st.rerun()

    if df_raw.empty: st.info("尚無持股。")
    else:
        st.metric("💰 總資產 (TWD)", f"${float(portfolio['現值(TWD)'].sum()):,.0f}")
        st.divider()
        st.subheader("📊 資產佔比圖")
        c_view = st.selectbox("圖表範圍", ["全部資產", "僅限台股", "僅限美股"])
        df_p = portfolio if c_view == "全部資產" else (portfolio[portfolio["幣別"]=="TWD"] if c_view=="僅限台股" else portfolio[portfolio["幣別"]=="USD"])
        if not df_p.empty:
            fig = px.pie(df_p, values="現值(TWD)", names="股票代號", hole=0.4); st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        df_tw, df_us = portfolio[portfolio["幣別"]=="TWD"], portfolio[portfolio["幣別"]=="USD"]
        if not df_tw.empty: st.subheader("🇹🇼 台股明細"); display_headers(); display_stock_rows(df_tw); display_subtotal_row(df_tw, "台股小計")
        if not df_us.empty: st.subheader("🇺🇸 美股明細"); display_headers(); display_stock_rows(df_us); display_subtotal_row(df_us, "美股小計")

# --- Tab 2 ---
with tab2:
    if df_raw.empty: st.info("請先新增標的。")
    else:
        st.subheader("🧠 AI 持股技術健診")
        sel_s = st.selectbox("選擇分析股票：", portfolio["股票代號"].tolist())
        if st.button("🚀 啟動診斷"):
            res, err = analyze_stock_technical(sel_s)
            if err: st.error(err)
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("價格", f"${res['current_price']:.2f}"); c2.metric("半年高", f"${res['high_6m']:.2f}")
                c3.metric("半年低", f"${res['low_6m']:.2f}"); c4.metric("RSI", f"{res['rsi']:.1f}")
                st.markdown(f"### 💡 建議：:{res['advice_color']}[{res['advice']}]")
                st.line_chart(res['df']['Close'])

# --- Tab 3 ---
with tab3:
    st.subheader("⚖️ MPT 權重優化與報酬預測")
    if not df_raw.empty and len(portfolio) >= 2:
        if st.button("🚀 執行模擬"):
            try:
                hist = yf.download(portfolio["股票代號"].tolist(), period="3y")['Close'].ffill().dropna()
                mpt = calculate_mpt_optimization(hist.pct_change().dropna())
                if mpt:
                    total_twd = float(portfolio["現值(TWD)"].sum())
                    curr_w = np.array([float(portfolio[portfolio["股票代號"]==s]["現值(TWD)"].sum()/total_twd) for s in mpt['symbols']])
                    def get_perf(w):
                        r = float(np.sum(mpt['mean_returns'] * w) * 100)
                        v = float(np.sqrt(np.dot(w.T, np.dot(mpt['cov_matrix'], w))) * 100)
                        return r, v
                    r_now, v_now = get_perf(curr_w)
                    r_min, v_min = get_perf(mpt['min_vol_weights'])
                    r_max, v_max = get_perf(mpt['max_sharpe_weights'])
                    
                    st.table(pd.DataFrame({
                        "方案": ["目前配置", "最小波動", "最高夏普"],
                        "預期報酬": [r_now, r_min, r_max],
                        "預期波動": [v_now, v_min, v_max]
                    }).set_index("方案").style.format("{:.2f}%"))
                    
                    
                    
                    st.dataframe(pd.DataFrame({"標的": mpt['symbols'], "目前": curr_w*100, "最小建議": mpt['min_vol_weights']*100, "最高建議": mpt['max_sharpe_weights']*100}).style.format("{:.2f}%"), use_container_width=True, hide_index=True)
            except Exception as e: st.error(f"分析失敗: {e}")
