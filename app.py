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
# 核心數學模型 (MPT) - 加強穩定性
# ==========================================

def calculate_mpt_optimization(returns_df):
    """
    執行 MPT 優化計算。
    returns_df 必須是清洗過、無 NaN 的每日報酬率。
    """
    # 年化轉換常數 (一年約 252 個交易日)
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    num_assets = len(mean_returns)
    
    # 檢查是否有無效數值
    if mean_returns.isnull().any() or cov_matrix.isnull().any().any():
        return None

    # 目標函數 1: 最小化投資組合波動度
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # 目標函數 2: 負夏普比率 (用於最大化)
    def neg_sharpe_ratio(weights, risk_free_rate=0.02):
        p_ret = np.sum(mean_returns * weights)
        p_vol = portfolio_volatility(weights)
        if p_vol == 0: return 0
        return -(p_ret - risk_free_rate) / p_vol

    # 限制條件：權重總和必須為 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # 限制條件：每支股票權重介於 0 與 1 之間 (不允許放空)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    try:
        # 優化：最小波動
        min_vol_res = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        # 優化：最大夏普
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
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def get_exchange_rate():
    try:
        # 獲取台幣匯率
        rate = yf.Ticker("USDTWD=X").fast_info.last_price
        return rate if rate and not pd.isna(rate) else 32.5
    except:
        return 32.5

def get_current_prices(symbols):
    prices = {}
    if not symbols: return prices
    # 這裡使用單個下載確保不會因為一支失敗而全部失敗
    for symbol in symbols:
        try:
            t = yf.Ticker(symbol)
            p = t.fast_info.last_price
            if p is None or pd.isna(p):
                # 備用方案：抓取最新歷史價格
                hist = t.history(period="1d")
                p = hist['Close'].iloc[-1] if not hist.empty else None
            prices[symbol] = p
        except:
            prices[symbol] = None
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
        sym = row["股票代號"]
        prof, roi = row["獲利(原幣)"], row["獲利率(%)"]
        color = "red" if prof > 0 else "green"
        fmt = "{:,.0f}" if row["幣別"] == "TWD" else "{:,.2f}"

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

tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 AI 技術診斷", "⚖️ MPT 數學模擬"])

df_raw = load_data()

# --- 資料預處理 (確保各 Tab 資料一致) ---
if not df_raw.empty:
    usd_rate = get_exchange_rate()
    # 修正：先計算單筆總成本再聚合，避免平均值的數學錯誤
    df_raw["單筆成本"] = df_raw["股數"] * df_raw["持有成本單價"]
    portfolio = df_raw.groupby("股票代號").agg({"股數":"sum", "單筆成本":"sum"}).reset_index()
    portfolio["平均持有單價"] = portfolio["單筆成本"] / portfolio["股數"]
    portfolio.rename(columns={"單筆成本": "總投入成本(原幣)"}, inplace=True)
    
    unique_syms = portfolio["股票代號"].tolist()
    prices = get_current_prices(unique_syms)
    
    portfolio["最新股價"] = portfolio["股票代號"].map(prices)
    portfolio["幣別"] = portfolio["股票代號"].apply(identify_currency)
    
    # 清理掉抓不到價格的股票
    portfolio = portfolio.dropna(subset=["最新股價"])
    
    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
    portfolio["現值(TWD)"] = portfolio.apply(lambda r: r["現值(原幣)"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

# --- Tab 1 ---
with tab1:
    with st.sidebar:
        st.header("📝 新增投資紀錄")
        with st.form("add_form", clear_on_submit=True):
            s_in = st.text_input("股票代號 (如: 2330.TW, NVDA)", "").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, value=0.0)
            c_in = st.number_input("平均買入成本單價", min_value=0.0, value=0.0)
            if st.form_submit_button("新增標的"):
                if s_in and q_in > 0:
                    new_entry = pd.DataFrame([{"股票代號":s_in, "股數":q_in, "持有成本單價":c_in}])
                    save_data(pd.concat([load_data(), new_entry], ignore_index=True))
                    st.rerun()

    if df_raw.empty:
        st.info("尚未有持股資料，請從左側新增。")
    else:
        st.metric("💰 總資產 (TWD)", f"${portfolio['現值(TWD)'].sum():,.0f}")
        display_stock_rows(portfolio)

# --- Tab 3: MPT 分析 ---
with tab3:
    st.subheader("⚖️ 現代投資組合 (MPT) 再平衡分析")
    if not df_raw.empty and len(portfolio) >= 2:
        if st.button("🚀 執行數學優化模擬 (抓取 3 年歷史資料)", type="primary"):
            with st.spinner("正在下載大數據並計算優化模型..."):
                try:
                    # 獲取歷史資料
                    syms = portfolio["股票代號"].tolist()
                    hist = yf.download(syms, period="3y", interval="1d")['Close']
                    
                    # 處理單支股票與多支股票回傳格式不同的問題
                    if isinstance(hist, pd.Series):
                        hist = hist.to_frame(name=syms[0])
                    
                    # 關鍵：處理資料斷層 (例如：美股開市但台股休市)
                    hist = hist.ffill().dropna()
                    
                    if len(hist) < 30:
                        st.error("歷史數據量不足，無法進行分析。")
                    else:
                        returns = hist.pct_change().dropna()
                        mpt_results = calculate_mpt_optimization(returns)
                        
                        if mpt_results:
                            # 計算目前權重比
                            total_twd = portfolio["現值(TWD)"].sum()
                            current_weights = portfolio.set_index("股票代號")["現值(TWD)"] / total_twd
                            
                            res_df = pd.DataFrame({
                                "標的": mpt_results['symbols'],
                                "目前佔比 (%)": [current_weights.get(s, 0)*100 for s in mpt_results['symbols']],
                                "最小波動配置建議 (%)": mpt_results['min_vol_weights'] * 100,
                                "最高夏普配置建議 (%)": mpt_results['max_sharpe_weights'] * 100
                            })
                            
                            st.markdown("### 1️⃣ 權重優化對比")
                            st.dataframe(res_df.style.format("{:.2f}%"), use_container_width=True, hide_index=True)
                            
                            # 效率前緣示意圖
                            

                            st.markdown("### 2️⃣ 相關性矩陣 (分散投資檢查)")
                            st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale='RdBu_r'))
                        else:
                            st.error("優化計算失敗，可能是標的相關性過高或數據異常。")
                except Exception as e:
                    st.error(f"分析過程發生錯誤: {e}")
    else:
        st.warning("執行 MPT 分析至少需要庫存內有 2 支有效的標的。")
