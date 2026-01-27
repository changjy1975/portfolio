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
# 核心數學模型 (MPT)
# ==========================================

def calculate_mpt_optimization(returns_df):
    """執行 MPT 優化計算：最小波動與最高夏普比率"""
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
        
        if not min_vol_res.success or not max_sharpe_res.success:
            return None

        return {
            "symbols": list(returns_df.columns),
            "min_vol_weights": min_vol_res.x,
            "max_sharpe_weights": max_sharpe_res.x
        }
    except:
        return None

# ==========================================
# 數據處理工具
# ==========================================

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # 強制轉換格式，避免讀取時產生型別錯誤
        df["股票代號"] = df["股票代號"].astype(str)
        df["股數"] = pd.to_numeric(df["股數"], errors='coerce')
        df["持有成本單價"] = pd.to_numeric(df["持有成本單價"], errors='coerce')
        return df.dropna()
    return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def get_exchange_rate():
    try:
        rate = yf.Ticker("USDTWD=X").fast_info.last_price
        return float(rate) if rate and not pd.isna(rate) else 32.5
    except:
        return 32.5

def get_current_prices(symbols):
    prices = {}
    if not symbols: return prices
    for symbol in symbols:
        try:
            t = yf.Ticker(symbol)
            p = t.fast_info.last_price
            if p is None or pd.isna(p):
                hist = t.history(period="1d")
                p = hist['Close'].iloc[-1] if not hist.empty else 0
            prices[symbol] = float(p)
        except:
            prices[symbol] = 0.0
    return prices

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# ==========================================
# UI 渲染函數
# ==========================================

COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def display_stock_rows(df):
    for _, row in df.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        sym = str(row["股票代號"])
        prof = float(row["獲利(原幣)"])
        roi = float(row["獲利率(%)"])
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
            df_old = load_data()
            df_old = df_old[df_old["股票代號"] != sym]
            save_data(df_old)
            st.rerun()

# ==========================================
# 主程式邏輯
# ==========================================

tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 AI 技術診斷", "⚖️ MPT 數學模擬"])

df_raw = load_data()

if not df_raw.empty:
    usd_rate = get_exchange_rate()
    # 確保數值型態
    df_raw["股數"] = pd.to_numeric(df_raw["股數"])
    df_raw["持有成本單價"] = pd.to_numeric(df_raw["持有成本單價"])
    df_raw["單筆成本"] = df_raw["股數"] * df_raw["持有成本單價"]
    
    portfolio = df_raw.groupby("股票代號").agg({"股數":"sum", "單筆成本":"sum"}).reset_index()
    portfolio["平均持有單價"] = portfolio["單筆成本"] / portfolio["股數"]
    portfolio.rename(columns={"單筆成本": "總投入成本(原幣)"}, inplace=True)
    
    prices = get_current_prices(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(prices).astype(float)
    portfolio["幣別"] = portfolio["股票代號"].apply(identify_currency)
    
    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    # 避免除以零
    portfolio["獲利率(%)"] = portfolio.apply(lambda r: (r["獲利(原幣)"]/r["總投入成本(原幣)"]*100) if r["總投入成本(原幣)"] != 0 else 0, axis=1)
    portfolio["現值(TWD)"] = portfolio.apply(lambda r: r["現值(原幣)"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

# --- Tab 1 ---
with tab1:
    with st.sidebar:
        st.header("📝 新增投資紀錄")
        with st.form("add_form", clear_on_submit=True):
            s_in = st.text_input("股票代號 (如: 2330.TW, NVDA)", "").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, value=100.0)
            c_in = st.number_input("買入單價", min_value=0.0, value=100.0)
            if st.form_submit_button("新增標的"):
                if s_in and q_in > 0:
                    new_entry = pd.DataFrame([{"股票代號":s_in, "股數":q_in, "持有成本單價":c_in}])
                    save_data(pd.concat([load_data(), new_entry], ignore_index=True))
                    st.rerun()

    if df_raw.empty:
        st.info("尚未有持股資料，請從左側新增。")
    else:
        total_asset = float(portfolio['現值(TWD)'].sum())
        st.metric("💰 總資產 (TWD)", f"${total_asset:,.0f}")
        display_stock_rows(portfolio)

# --- Tab 3: MPT 分析 ---
with tab3:
    st.subheader("⚖️ MPT 再平衡分析")
    if not df_raw.empty and len(portfolio) >= 2:
        if st.button("🚀 執行分析", type="primary"):
            try:
                syms = portfolio["股票代號"].tolist()
                hist = yf.download(syms, period="3y", interval="1d")['Close']
                if isinstance(hist, pd.Series): hist = hist.to_frame(name=syms[0])
                returns = hist.ffill().pct_change().dropna()
                
                mpt_results = calculate_mpt_optimization(returns)
                
                if mpt_results:
                    total_twd = float(portfolio["現值(TWD)"].sum())
                    current_weights = portfolio.set_index("股票代號")["現值(TWD)"] / total_twd
                    
                    res_df = pd.DataFrame({
                        "標的": mpt_results['symbols'],
                        "目前佔比 (%)": [float(current_weights.get(s, 0)*100) for s in mpt_results['symbols']],
                        "最小波動建議 (%)": [float(w*100) for w in mpt_results['min_vol_weights']],
                        "最高夏普建議 (%)": [float(w*100) for w in mpt_results['max_sharpe_weights']]
                    })
                    
                    st.markdown("### 1️⃣ 權重優化建議")
                    # 強制轉換為 float 避免 Style 報錯
                    st.dataframe(res_df.style.format({
                        "目前佔比 (%)": "{:.2f}%",
                        "最小波動建議 (%)": "{:.2f}%",
                        "最高夏普建議 (%)": "{:.2f}%"
                    }), use_container_width=True, hide_index=True)

                    
                    
                    st.markdown("### 2️⃣ 相關性矩陣")
                    st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale='RdBu_r'))
                else:
                    st.error("計算失敗，請確認標的歷史數據是否充足。")
            except Exception as e:
                st.error(f"分析過程發生錯誤: {e}")
    else:
        st.warning("至少需要 2 支標的。")
