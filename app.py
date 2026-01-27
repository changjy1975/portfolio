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
    """
    執行 MPT 優化計算：最小波動與最高夏普比率
    """
    mean_returns = returns_df.mean() * 252  # 年化報酬
    cov_matrix = returns_df.cov() * 252    # 年化共變異矩陣
    num_assets = len(mean_returns)
    
    # 目標函數 1: 投資組合波動度
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # 目標函數 2: 負夏普比率 (用於最大化)
    def neg_sharpe_ratio(weights, risk_free_rate=0.02):
        p_ret = np.sum(mean_returns * weights)
        p_vol = portfolio_volatility(weights)
        return -(p_ret - risk_free_rate) / p_vol

    # 設定限制條件：權重總和為 1，各股權重 0~1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    # 優化：最小波動 (Minimum Variance)
    min_vol_res = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # 優化：最大夏普 (Max Sharpe / Efficient Frontier)
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
        ticker = yf.Ticker("USDTWD=X")
        rate = ticker.fast_info.last_price
        return rate if rate and not pd.isna(rate) else 32.5
    except: return 32.5

def get_current_prices(symbols):
    prices = {}
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
# 分析組件邏輯
# ==========================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def perform_portfolio_analysis(portfolio_df):
    symbols = portfolio_df["股票代號"].unique().tolist()
    if len(symbols) < 2: return None, "分析需要至少兩支以上的股票資料。"
    
    try:
        # 抓取 3 年歷史資料進行模擬
        hist_data = yf.download(symbols, period="3y", interval="1d", auto_adjust=True)['Close']
        if isinstance(hist_data, pd.Series): hist_data = hist_data.to_frame(name=symbols[0])
        hist_data = hist_data.fillna(method='ffill').dropna()
        returns = hist_data.pct_change().dropna()
        
        # MPT 優化計算
        mpt_res = calculate_mpt_optimization(returns)
        
        # 個股指標計算
        perf_list = []
        for s in symbols:
            s_ret = returns[s]
            cagr = ((1 + s_ret.mean())**252 - 1) * 100
            vol = (s_ret.std() * np.sqrt(252)) * 100
            perf_list.append({
                "股票代號": s,
                "CAGR (%)": cagr,
                "年化波動率 (%)": vol,
                "Sharpe Ratio": (cagr/100 - 0.02) / (vol/100) if vol != 0 else 0
            })
            
        return {
            "corr_matrix": returns.corr(),
            "perf_df": pd.DataFrame(perf_list),
            "mpt": mpt_res
        }, None
    except Exception as e: return None, str(e)

# ==========================================
# 介面渲染組件
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
            save_data(df_old)
            st.rerun()

# ==========================================
# 主程式邏輯
# ==========================================

if "last_updated" not in st.session_state: st.session_state.last_updated = "尚未更新"

tab1, tab2, tab3 = st.tabs(["📊 庫存資產配置", "🧠 AI 持股健診", "⚖️ MPT 數學模擬器"])

df_record = load_data()

# --- Tab 1: 基礎資產管理 ---
with tab1:
    with st.sidebar:
        st.header("📝 新增標的")
        with st.form("add_form"):
            s_in = st.text_input("股票代號", "2330.TW").upper().strip()
            q_in = st.number_input("股數", value=100.0)
            c_in = st.number_input("成本單價", value=600.0)
            if st.form_submit_button("新增到庫存"):
                df_new = pd.concat([load_data(), pd.DataFrame([{"股票代號":s_in, "股數":q_in, "持有成本單價":c_in}])])
                save_data(df_new); st.rerun()
        if st.button("🚨 清空數據庫"): 
            if os.path.exists(DATA_FILE): os.remove(DATA_FILE); st.rerun()

    if df_record.empty:
        st.info("目前庫存空空如也，請先從側邊欄新增股票。")
    else:
        usd_rate = get_exchange_rate()
        unique_syms = df_record["股票代號"].unique().tolist()
        prices = get_current_prices(unique_syms)
        
        # 資料處理
        portfolio = df_record.groupby("股票代號").agg({"股數":"sum", "持有成本單價":"mean"}).reset_index()
        portfolio["最新股價"] = portfolio["股票代號"].map(prices)
        portfolio = portfolio.dropna(subset=["最新股價"])
        portfolio["幣別"] = portfolio["股票代號"].apply(identify_currency)
        portfolio["總投入成本(原幣)"] = portfolio["股數"] * portfolio["持有成本單價"]
        portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
        portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
        portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
        portfolio["現值(TWD)"] = portfolio.apply(lambda r: r["現值(原幣)"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

        # 看板與清單
        t_val = portfolio["現值(TWD)"].sum()
        t_prof = (portfolio["現值(TWD)"] - (portfolio["總投入成本(原幣)"] * portfolio.apply(lambda r: usd_rate if r["幣別"]=="USD" else 1, axis=1))).sum()
        
        c1, c2 = st.columns(2)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}")
        c2.metric("📈 總損益 (TWD)", f"${t_prof:,.0f}")
        
        st.divider()
        st.subheader("📦 詳細持股清單")
        display_stock_rows(portfolio, "MIX")

# --- Tab 3: MPT 數學模擬器 ---
with tab3:
    st.subheader("⚖️ 現代投資組合 (MPT) 權重優化模擬")
    st.caption("系統將抓取過去 3 年歷史資料，透過數學算法計算在「風險最低」與「報酬風險比最高」時的理想權重。")

    if not df_record.empty:
        if st.button("🚀 執行效率前緣模擬分析", type="primary"):
            with st.spinner("正在下載大數據並計算優化模型..."):
                res, err = perform_portfolio_analysis(portfolio)
                if err: st.error(err)
                else:
                    st.session_state['analysis_res'] = res
        
        if 'analysis_res' in st.session_state:
            res = st.session_state['analysis_res']
            mpt = res['mpt']
            
            # 計算目前權重
            total_val = portfolio["現值(TWD)"].sum()
            curr_w = {row["股票代號"]: (row["現值(TWD)"]/total_val)*100 for _, row in portfolio.iterrows()}
            
            # 對比表格
            mpt_comparison = pd.DataFrame({
                "標的": mpt['symbols'],
                "目前實際權重 (%)": [curr_w.get(s, 0) for s in mpt['symbols']],
                "最小波動建議 (%)": mpt['min_vol_weights'] * 100,
                "最優回報建議 (%)": mpt['max_sharpe_weights'] * 100
            })
            
            
            
            st.markdown("### 1️⃣ 權重分配對比表")
            st.dataframe(mpt_comparison.style.format("{:.2f}%"), use_container_width=True, hide_index=True)
            
            st.markdown("### 2️⃣ 再平衡建議圖表")
            fig_mpt = px.bar(mpt_comparison, x="標的", y=["目前實際權重 (%)", "最小波動建議 (%)", "最優回報建議 (%)"], 
                            barmode="group", title="權重分佈對比 (MPT vs. 現狀)")
            st.plotly_chart(fig_mpt, use_container_width=True)
            
            st.markdown("### 3️⃣ 相關性矩陣 (風險分散檢查)")
            fig_heat = px.imshow(res['corr_matrix'], text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("請先新增至少兩支股票以執行 MPT 分析。")
