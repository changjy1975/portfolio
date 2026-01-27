import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize

# --- 檔案儲存設定 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="Pro 投資組合戰情室", layout="wide", initial_sidebar_state="expanded")

# 自定義 CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 核心數據獲取 (加入快取機制)
# ==========================================

@st.cache_data(ttl=3600)  # 快取 1 小時
def fetch_stock_data(symbols, period="1y"):
    if not symbols: return pd.DataFrame()
    data = yf.download(symbols, period=period)['Close']
    return data

@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        rate = yf.Ticker("USDTWD=X").fast_info.last_price
        return float(rate) if rate else 32.5
    except:
        return 32.5

# ==========================================
# 2. 核心數學與分析函數
# ==========================================

def calculate_mpt_optimization(returns_df):
    """執行 MPT 優化計算"""
    returns_df = returns_df.astype(float).dropna()
    if returns_df.empty: return None
    
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    num_assets = len(mean_returns)

    def portfolio_performance(weights):
        returns = np.sum(mean_returns * weights)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, volatility

    def neg_sharpe_ratio(weights, risk_free_rate=0.02):
        p_ret, p_vol = portfolio_performance(weights)
        return -(p_ret - risk_free_rate) / (p_vol + 1e-9)

    def volatility_only(weights):
        return portfolio_performance(weights)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    try:
        opt_sharpe = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_vol = minimize(volatility_only, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return {
            "symbols": list(returns_df.columns),
            "min_vol_weights": opt_vol.x,
            "max_sharpe_weights": opt_sharpe.x,
            "mean_returns": mean_returns,
            "cov_matrix": cov_matrix
        }
    except:
        return None

def calculate_rsi(series, period=14):
    """標準 RSI 計算 (使用 EMA)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period-1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period-1, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ==========================================
# 3. 數據處理工具
# ==========================================

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["股票代號"] = df["股票代號"].astype(str)
        return df.dropna()
    return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def identify_currency(symbol):
    return "TWD" if any(x in symbol.upper() for x in [".TW", ".TWO"]) else "USD"

# ==========================================
# 4. UI 元件優化
# ==========================================

def display_stock_table(df, title):
    st.subheader(title)
    # 設定欄位比例
    cols = st.columns([1.2, 1, 1, 1, 1.2, 1.2, 1.2, 1, 0.6])
    headers = ["代號", "股數", "均價", "現價", "成本(原)", "現值(原)", "獲利(原)", "報酬率", "管理"]
    for col, head in zip(cols, headers): col.write(f"**{head}**")
    
    for i, row in df.iterrows():
        c = st.columns([1.2, 1, 1, 1, 1.2, 1.2, 1.2, 1, 0.6])
        color = "red" if row["獲利(原幣)"] >= 0 else "green"
        fmt = "{:,.0f}" if row["幣別"] == "TWD" else "{:,.2f}"
        
        c[0].write(row["股票代號"])
        c[1].write(f"{row['股數']:.1f}")
        c[2].write(f"{row['平均持有單價']:.2f}")
        c[3].write(f"{row['最新股價']:.2f}")
        c[4].write(fmt.format(row["總投入成本(原幣)"]))
        c[5].write(fmt.format(row["現值(原幣)"]))
        c[6].markdown(f":{color}[{fmt.format(row['獲利(原幣)'])}]")
        c[7].markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        if c[8].button("🗑️", key=f"del_{row['股票代號']}_{i}"):
            full_df = load_data()
            full_df = full_df.drop(i)
            save_data(full_df)
            st.rerun()

# ==========================================
# 5. 主程式邏輯
# ==========================================

st.title("📈 智能投資組合戰情室")

tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 AI 技術診斷", "⚖️ MPT 數學模擬"])
df_raw = load_data()

# --- 資料處理流程 ---
if not df_raw.empty:
    usd_rate = get_exchange_rate()
    # 聚合相同代號
    portfolio = df_raw.groupby("股票代號").apply(
        lambda x: pd.Series({
            "股數": x["股數"].sum(),
            "總投入成本(原幣)": (x["股數"] * x["持有成本單價"]).sum()
        })
    ).reset_index()
    
    portfolio["平均持有單價"] = portfolio["總投入成本(原幣)"] / portfolio["股數"]
    unique_syms = portfolio["股票代號"].tolist()
    
    # 批次下載最新價
    with st.spinner("更新市場報價中..."):
        current_prices = fetch_stock_data(unique_syms, period="5d").iloc[-1]
    
    # 處理單一標的與多標的 yfinance 回傳格式差異
    if len(unique_syms) == 1:
        portfolio["最新股價"] = float(current_prices)
    else:
        portfolio["最新股價"] = portfolio["股票代號"].map(current_prices).astype(float)

    portfolio["幣別"] = portfolio["股票代號"].apply(identify_currency)
    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
    portfolio["現值(TWD)"] = portfolio.apply(lambda r: r["現值(原幣)"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

# --- Tab 1: 庫存配置 ---
with tab1:
    with st.sidebar:
        st.header("📝 新增投資標的")
        with st.form("add_form"):
            s_in = st.text_input("代號 (如: 2330.TW, VOO)", "").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, value=10.0, step=1.0)
            c_in = st.number_input("成交單價", min_value=0.0, value=100.0, step=0.1)
            if st.form_submit_button("確認新增"):
                if s_in:
                    new_data = pd.DataFrame([{"股票代號":s_in, "股數":q_in, "持有成本單價":c_in}])
                    save_data(pd.concat([df_raw, new_data], ignore_index=True))
                    st.rerun()

    if df_raw.empty:
        st.info("💡 目前庫存空空如也，請先從左側選單新增標的。")
    else:
        t1_col1, t1_col2 = st.columns([1, 1])
        with t1_col1:
            st.metric("💰 總資產 (TWD)", f"${portfolio['現值(TWD)'].sum():,.0f}")
        with t1_col2:
            st.metric("💵 今日匯率 (USD/TWD)", f"{usd_rate:.2f}")

        st.divider()
        
        # 圓餅圖
        fig_pie = px.pie(portfolio, values="現值(TWD)", names="股票代號", hole=0.4, title="資產權重分布")
        st.plotly_chart(fig_pie, use_container_width=True)

        # 明細表
        df_tw = portfolio[portfolio["幣別"]=="TWD"]
        df_us = portfolio[portfolio["幣別"]=="USD"]
        if not df_tw.empty: display_stock_table(df_tw, "🇹🇼 台股庫存")
        if not df_us.empty: display_stock_table(df_us, "🇺🇸 美股庫存")

# --- Tab 2: AI 技術診斷 ---
with tab2:
    if not df_raw.empty:
        target = st.selectbox("選擇分析標的", portfolio["股票代號"].unique())
        if st.button("🔍 執行深度診斷"):
            hist = yf.download(target, period="1y")
            if not hist.empty:
                # 計算指標
                curr_price = hist['Close'].iloc[-1]
                ma20 = hist['Close'].rolling(20).mean()
                rsi = calculate_rsi(hist['Close'])
                
                # 繪製專業圖表
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="股價"), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=ma20, name="MA20", line=dict(dash='dot')), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=rsi, name="RSI(14)", line=dict(color='orange')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                fig.update_layout(height=500, title_text=f"{target} 技術走勢")
                st.plotly_chart(fig, use_container_width=True)
                
                # 診斷內容
                last_rsi = rsi.iloc[-1]
                if last_rsi > 70: status, color = "超買過熱", "red"
                elif last_rsi < 30: status, color = "超跌低估", "green"
                else: status, color = "區間整理", "gray"
                
                st.info(f"**診斷結果：{target} 目前處於 :{color}[{status}] 狀態** (RSI: {last_rsi:.2f})")
    else:
        st.warning("請先新增標的以進行診斷。")

# --- Tab 3: MPT 優化 ---
with tab3:
    if not df_raw.empty and len(portfolio) >= 2:
        if st.button("📈 執行優化模擬"):
            with st.spinner("正在進行數萬次蒙地卡羅模擬運算..."):
                hist_data = fetch_stock_data(portfolio["股票代號"].tolist(), "3y")
                returns = hist_data.pct_change().dropna()
                res = calculate_mpt_optimization(returns)
                
                if res:
                    st.success("優化計算完成！")
                    total_val = portfolio["現值(TWD)"].sum()
                    curr_w = [portfolio[portfolio["股票代號"]==s]["現值(TWD)"].sum()/total_val for s in res['symbols']]
                    
                    comparison = pd.DataFrame({
                        "標的": res['symbols'],
                        "目前權重": [f"{w*100:.1f}%" for w in curr_w],
                        "最小風險建議": [f"{w*100:.1f}%" for w in res['min_vol_weights']],
                        "最高回報建議": [f"{w*100:.1f}%" for w in res['max_sharpe_weights']]
                    })
                    st.table(comparison)
                    
                    st.caption("註：優化結果基於過去 3 年歷史數據計算，僅供參考。")
    else:
        st.warning("MPT 優化至少需要 2 個以上的投資標的。")
