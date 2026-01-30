import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import os
import shutil
from datetime import datetime
import pytz
import numpy as np

# ==========================================
# 1. 初始化設定與路徑
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心功能函數 (資料、備份與行情)
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

@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        rate = yf.Ticker("USDTWD=X").fast_info.last_price
        return float(rate) if rate else 32.5
    except: return 32.5

@st.cache_data(ttl=300)
def get_latest_quotes(symbols):
    """跨市場批次抓取最後成交價"""
    if not symbols: return {}
    quotes = {}
    try:
        # 使用多執行緒抓取 Tickers
        tickers = yf.Tickers(" ".join(symbols))
        for s in symbols:
            try:
                # 優先抓取即時最後價格
                price = tickers.tickers[s].fast_info.last_price
                if price is None or np.isnan(price):
                    # 備案：抓取昨日收盤
                    price = tickers.tickers[s].history(period="1d")['Close'].iloc[-1]
                quotes[s] = float(price)
            except:
                quotes[s] = 0.0
        return quotes
    except: return {s: 0.0 for s in symbols}

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

def calculate_rsi(series, period=14):
    """精確化 RSI (EMA)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 3. MPT 數學模擬引擎 (完整邏輯)
# ==========================================

def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "至少需要 2 支標的才能進行組合優化。"
    try:
        # 下載 3 年歷史數據
        data = yf.download(symbols, period="3y", interval="1d", auto_adjust=True)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=symbols[0])
        data = data.ffill().pct_change().dropna()
        
        # 計算年化報酬與共變異
        mean_returns = data.mean() * 252
        cov_matrix = data.cov() * 252
        
        # 蒙地卡羅模擬 2000 種組合
        num_portfolios = 2000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(len(symbols))
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            results[0,i] = portfolio_return
            results[1,i] = portfolio_std
            results[2,i] = (portfolio_return - 0.02) / portfolio_std # 假設無風險利率 2%
            
        max_sharpe_idx = np.argmax(results[2])
        min_vol_idx = np.argmin(results[1])
        
        # 目前配置權重
        current_weights_val = portfolio_df["現值_TWD"].values
        current_weights = current_weights_val / np.sum(current_weights_val)
        
        comparison = pd.DataFrame({
            "股票代號": symbols,
            "目前權重 (%)": current_weights * 100,
            "建議-高回報 (Max Sharpe) (%)": weights_record[max_sharpe_idx] * 100,
            "建議-低波動 (Min Vol) (%)": weights_record[min_vol_idx] * 100
        })

        return {
            "sim_df": pd.DataFrame({'Return': results[0], 'Volatility': results[1], 'Sharpe': results[2]}),
            "comparison": comparison,
            "max_sharpe": (results[0, max_sharpe_idx], results[1, max_sharpe_idx]),
            "min_vol": (results[0, min_vol_idx], results[1, min_vol_idx]),
            "corr": data.corr()
        }, None
    except Exception as e: return None, str(e)

# ==========================================
# 4. 主程式與介面
# ==========================================

if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add_form", clear_on_submit=True):
            st.subheader("📝 新增持股")
            s_in = st.text_input("代號 (如 2330.TW 或 NVDA)").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, step=1.0)
            c_in = st.number_input("成本", min_value=0.0, step=0.1)
            if st.form_submit_button("執行新增"):
                if s_in:
                    df = load_data(current_user)
                    save_data(pd.concat([df, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user)
                    st.rerun()

# 資料讀取
if current_user == "All":
    df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True)
else:
    df_record = load_data(current_user)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)"])

if not df_record.empty:
    usd_rate = get_exchange_rate()
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    
    # 彙整組合
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(
        lambda g: pd.Series({
            '股數': g['股數'].sum(),
            '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()
        }), include_groups=False
    ).reset_index()

    # 抓取報價
    price_map = get_latest_quotes(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    
    # 計算各項指標
    portfolio["總投入成本"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]) * 100
    
    # 台幣換算 (用於圓餅圖與總額)
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (usd_rate if r["幣別"] == "USD" else 1), axis=1)
    portfolio["獲利_TWD"] = portfolio.apply(lambda r: r["獲利"] * (usd_rate if r["幣別"] == "USD" else 1), axis=1)

    t_val = float(portfolio["現值_TWD"].sum())
    t_prof = float(portfolio["獲利_TWD"].sum())
    roi_pct = (t_prof / (t_val - t_prof) * 100) if (t_val - t_prof) != 0 else 0

    with tab1:
        # 刷新按鈕
        if st.button("🔄 刷新最新報價"):
            st.cache_data.clear()
            st.rerun()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}")
        c3.metric("📊 總報酬率", f"{roi_pct:.2f}%"); c4.metric("💱 匯率", f"{usd_rate:.2f}")
        
        st.divider()
        st.subheader("🎯 組合配置分析")
        cc1, cc2 = st.columns(2)
        # 修正圓餅圖：確保抓取的是計算後的 現值_TWD
        with cc1: 
            pie_data = portfolio.groupby("幣別")["現值_TWD"].sum().reset_index()
            st.plotly_chart(px.pie(pie_data, values="現值_TWD", names="幣別", title="市場配置 (TWD)", hole=0.4), use_container_width=True)
        with cc2:
            v_opt = st.selectbox("標的分佈：", ["全部", "台股", "美股"])
            pdf = portfolio[portfolio["幣別"] == "TWD"] if v_opt == "台股" else portfolio[portfolio["幣別"] == "USD"] if v_opt == "美股" else portfolio
            if not pdf.empty:
                st.plotly_chart(px.pie(pdf, values="現值_TWD", names="股票代號", title=f"{v_opt} 持股比例", hole=0.4), use_container_width=True)

        st.divider()
        # 清單列表 (略過複雜的自訂 Header 以確保穩定)
        st.subheader("📋 詳細持股清單")
        st.dataframe(portfolio[["股票代號", "幣別", "股數", "平均持有單價", "最新股價", "獲利", "獲利率(%)", "現值_TWD"]].style.format({
            "平均持有單價": "{:.2f}", "最新股價": "{:.2f}", "獲利": "{:,.2f}", "獲利率(%)": "{:.2f}%", "現值_TWD": "{:,.0f}"
        }), use_container_width=True)

    with tab2:
        target = st.selectbox("分析標的：", portfolio["股票代號"].tolist())
        hist = yf.Ticker(target).history(period="1y")
        if not hist.empty:
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            st.metric(f"{target} RSI (14D)", f"{rsi:.2f}")
            st.line_chart(hist['Close'])

    with tab3:
        st.subheader("⚖️ 投資組合優化模擬 (Modern Portfolio Theory)")
        st.write("本功能將根據過去 3 年的歷史數據，透過 2,000 次隨機模擬，找出風險與回報平衡的最佳路徑。")
        
        if st.button("🚀 開始計算最佳權重"):
            with st.spinner("正在進行大數據模擬..."):
                res, err = perform_mpt_simulation(portfolio)
                if err:
                    st.error(f"模擬失敗：{err}")
                else:
                    st.success("模擬完成！")
                    sc1, sc2 = st.columns([2, 1])
                    with sc1:
                        st.write("#### 1️⃣ 效率前緣分佈 (Efficient Frontier)")
                        fig = px.scatter(res['sim_df'], x='Volatility', y='Return', color='Sharpe', color_continuous_scale='Viridis', labels={'Volatility':'年化波動度','Return':'預期回報'})
                        fig.add_trace(go.Scatter(x=[res['max_sharpe'][1]], y=[res['max_sharpe'][0]], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Max Sharpe'))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with sc2:
                        st.write("#### 2️⃣ 建議配置建議")
                        st.dataframe(res['comparison'].set_index("股票代號").style.format("{:.2f}%"))
                    
                    st.divider()
                    st.write("#### 3️⃣ 資產相關性矩陣 (降低風險的關鍵)")
                    st.plotly_chart(px.imshow(res['corr'], text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1), use_container_width=True)
                    st.info("💡 相關係數越低（趨向藍色）的資產組合，越能達到避險效果。")
else:
    st.info("尚未發現任何持股資料，請從左側選單新增標的。")
