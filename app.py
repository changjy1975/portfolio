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
    """跨市場批次抓取最後成交價 (解決台美股時差導致消失的問題)"""
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

def calculate_rsi(series, period=14):
    """精確化 RSI (使用 EMA)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 3. MPT 數學模擬引擎
# ==========================================

def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "至少需要 2 支標的才能優化。"
    try:
        data = yf.download(symbols, period="3y", interval="1d", auto_adjust=True)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=symbols[0])
        data = data.ffill().pct_change().dropna()
        mean_returns = data.mean() * 252
        cov_matrix = data.cov() * 252
        
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
            results[2,i] = (portfolio_return - 0.02) / portfolio_std
            
        max_idx = np.argmax(results[2]); min_idx = np.argmin(results[1])
        current_weights = portfolio_df["現值_TWD"].values / portfolio_df["現值_TWD"].sum()
        
        comparison = pd.DataFrame({
            "股票代號": symbols,
            "目前權重 (%)": current_weights * 100,
            "高回報建議 (Max Sharpe) (%)": weights_record[max_idx] * 100,
            "低波動建議 (Min Vol) (%)": weights_record[min_idx] * 100
        })
        return {"sim_df": pd.DataFrame({'Return': results[0], 'Volatility': results[1], 'Sharpe': results[2]}),
                "comparison": comparison, "max_sharpe": (results[0, max_idx], results[1, max_idx]),
                "min_vol": (results[0, min_idx], results[1, min_idx]), "corr": data.corr()}, None
    except Exception as e: return None, str(e)

# ==========================================
# 4. 介面顯示組件 (表格與小計)
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, title, currency, usd_rate, current_user):
    st.subheader(title)
    cols = st.columns(COLS_RATIO)
    for col, h in zip(cols, ["代號", "股數", "均價", "現價", "總成本", "現值", "獲利", "報酬率", "管理"]):
        col.caption(f"**{h}**")
    
    for _, row in df.iterrows():
        r = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"
        color = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}"); r[4].write(fmt.format(row['總投入成本'])); r[5].write(fmt.format(row['現值'])); r[6].markdown(f":{color}[{fmt.format(row['獲利'])}]"); r[7].markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
            full = load_data(current_user)
            save_data(full[full["股票代號"] != row['股票代號']], current_user); st.rerun()

    # 小計
    s_cost, s_val, s_profit = df["總投入成本"].sum(), df["現值"].sum(), df["獲利"].sum()
    s_roi = (s_profit / s_cost * 100) if s_cost != 0 else 0
    st.markdown("---")
    sc = st.columns(COLS_RATIO)
    sc[0].markdown(f"**{currency} 小計**"); sc[4].markdown(f"**{fmt.format(s_cost)}**"); sc[5].markdown(f"**{fmt.format(s_val)}**"); sc[6].markdown(f":{'red' if s_profit>0 else 'green'}[**{fmt.format(s_profit)}**]"); sc[7].markdown(f":{'red' if s_profit>0 else 'green'}[**{s_roi:.2f}%**]")
    if currency == "USD":
        sc2 = st.columns(COLS_RATIO); sc2[0].caption("*(換算台幣)*"); sc2[4].caption(f"${(s_cost*usd_rate):,.0f}"); sc2[5].caption(f"${(s_val*usd_rate):,.0f}"); sc2[6].caption(f"${(s_profit*usd_rate):,.0f}")
    st.write("")

# ==========================================
# 5. 主程式邏輯
# ==========================================

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add_form", clear_on_submit=True):
            st.subheader("📝 新增持股")
            s_in = st.text_input("代號 (如 2330.TW 或 NVDA)").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, step=1.0); c_in = st.number_input("成本", min_value=0.0, step=0.1)
            if st.form_submit_button("執行新增"):
                if s_in:
                    df = load_data(current_user)
                    save_data(pd.concat([df, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user)
                    st.rerun()

# 數據加載與彙整
if current_user == "All":
    df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True)
else:
    df_record = load_data(current_user)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)"])

if not df_record.empty:
    usd_rate = get_exchange_rate()
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(
        lambda g: pd.Series({'股數': g['股數'].sum(), '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()}), include_groups=False
    ).reset_index()

    price_map = get_latest_quotes(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    portfolio["總投入成本"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]) * 100
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)
    portfolio["獲利_TWD"] = portfolio.apply(lambda r: r["獲利"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

    with tab1:
        # A. 工具列與概覽
        tc1, tc2 = st.columns([1, 4])
        if tc1.button("🔄 刷新報價"):
            st.cache_data.clear(); st.rerun()
        
        t_val = float(portfolio["現值_TWD"].sum()); t_prof = float(portfolio["獲利_TWD"].sum())
        roi = (t_prof / (t_val - t_prof) * 100) if (t_val - t_prof) != 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}"); c3.metric("📊 總報酬率", f"{roi:.2f}%"); c4.metric("💱 匯率", f"{usd_rate:.2f}")

        st.divider()
        
        # B. 圓餅圖區塊 (移至上方)
        st.subheader("🎯 投資組合配置分析")
        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="市場配置 (TWD)", hole=0.45), use_container_width=True)
        with pc2:
            st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="股票代號", title="個股配置 (TWD)", hole=0.45), use_container_width=True)

        st.divider()

        # C. 庫存狀況區塊 (移至下方)
        tw_df = portfolio[portfolio["幣別"] == "TWD"]
        if not tw_df.empty: display_market_table(tw_df, "🇹🇼 台股庫存明細", "TWD", usd_rate, current_user)
        
        us_df = portfolio[portfolio["幣別"] == "USD"]
        if not us_df.empty: display_market_table(us_df, "🇺🇸 美股庫存明細", "USD", usd_rate, current_user)

    with tab2:
        target = st.selectbox("分析標的：", portfolio["股票代號"].tolist())
        hist = yf.Ticker(target).history(period="1y")
        if not hist.empty:
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            st.metric(f"{target} RSI (14D)", f"{rsi:.2f}"); st.line_chart(hist['Close'])

    with tab3:
        st.subheader("⚖️ MPT 組合優化模擬")
        if st.button("🚀 開始計算最佳權重"):
            res, err = perform_mpt_simulation(portfolio)
            if err: st.error(err)
            else:
                st.success("模擬完成！")
                sc1, sc2 = st.columns([2, 1])
                with sc1:
                    fig = px.scatter(res['sim_df'], x='Volatility', y='Return', color='Sharpe', color_continuous_scale='Viridis', labels={'Volatility':'年化波動','Return':'預期回報'})
                    fig.add_trace(go.Scatter(x=[res['max_sharpe'][1]], y=[res['max_sharpe'][0]], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Max Sharpe'))
                    st.plotly_chart(fig, use_container_width=True)
                with sc2:
                    st.write("#### 建議權重對比")
                    st.dataframe(res['comparison'].set_index("股票代號").style.format("{:.2f}%"))
                st.divider()
                st.write("#### 資產相關性矩陣")
                st.plotly_chart(px.imshow(res['corr'], text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1), use_container_width=True)
else:
    st.info("尚未發現持股，請從側邊欄新增。")
