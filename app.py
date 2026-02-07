import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import shutil
from datetime import datetime, timedelta
import pytz
import numpy as np

# ==========================================
# 1. 初始化設定與路徑
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None
if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心功能函數 (效能與 MPT 引擎)
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
    if not symbols: return {}
    try:
        data = yf.download(symbols, period="1d", interval="1m", progress=False)['Close']
        if len(symbols) == 1:
            return {symbols[0]: float(data.iloc[-1])}
        return {s: float(data[s].iloc[-1]) for s in symbols}
    except:
        return {s: 0.0 for s in symbols}

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# --- 技術指標計算 ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean(); exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2; signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

# --- MPT 引擎 ---
def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "至少需要 2 支標的才能進行優化模擬。"
    try:
        data = yf.download(symbols, period="3y", interval="1d", progress=False)['Close']
        returns = data.ffill().pct_change().dropna()
        if returns.empty: return None, "數據樣本不足。"
        
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_portfolios = 2000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(len(symbols))
            weights /= np.sum(weights)
            weights_record.append(weights)
            p_ret = np.sum(weights * mean_returns)
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            results[0,i] = p_ret
            results[1,i] = p_std
            results[2,i] = (p_ret - 0.02) / p_std
            
        max_idx = np.argmax(results[2])
        min_idx = np.argmin(results[1])
        comparison = pd.DataFrame({
            "股票代號": symbols,
            "目前權重 (%)": (portfolio_df["現值_TWD"] / portfolio_df["現值_TWD"].sum() * 100).values,
            "Max Sharpe 建議 (%)": weights_record[max_idx] * 100,
            "Min Vol 建議 (%)": weights_record[min_idx] * 100
        })
        return {"sim_df": pd.DataFrame({'Return': results[0], 'Volatility': results[1], 'Sharpe': results[2]}),
                "comparison": comparison, "max_sharpe": (results[0, max_idx], results[1, max_idx]),
                "corr": returns.corr()}, None
    except Exception as e: return None, str(e)

# ==========================================
# 3. 介面組件 (表格顯示)
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, title, currency, usd_rate, current_user):
    st.subheader(title)
    h_map = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本"), ("現值", "現值"), ("獲利", "獲利"), ("報酬率", "獲利率(%)")]
    h_cols = st.columns(COLS_RATIO)
    for i, (label, col_name) in enumerate(h_map):
        arrow = " ▲" if st.session_state.sort_col == col_name and st.session_state.sort_asc else " ▼" if st.session_state.sort_col == col_name else ""
        if h_cols[i].button(f"{label}{arrow}", key=f"h_{currency}_{col_name}_{current_user}"):
            if st.session_state.sort_col == col_name: st.session_state.sort_asc = not st.session_state.sort_asc
            else: st.session_state.sort_col, st.session_state.sort_asc = col_name, False
            st.rerun()
    
    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in df_sorted.iterrows():
        r = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"
        color = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}")
        r[4].write(fmt.format(row['總投入成本'])); r[5].write(fmt.format(row['現值']))
        r[6].markdown(f":{color}[{fmt.format(row['獲利'])}]"); r[7].markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
            full = load_data(current_user); save_data(full[full["股票代號"] != row['股票代號']], current_user); st.rerun()

# ==========================================
# 4. 主程式邏輯
# ==========================================

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add_form", clear_on_submit=True):
            st.subheader("📝 新增持股")
            s_in = st.text_input("代號 (如 2330.TW)").upper().strip()
            q_in = st.number_input("股數", min_value=0.0); c_in = st.number_input("成本", min_value=0.0)
            if st.form_submit_button("執行新增"):
                if s_in:
                    df = load_data(current_user); save_data(pd.concat([df, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user); st.rerun()

df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3 = st.tabs(["📊 庫存績效", "🧠 技術健診", "⚖️ 組合優化 (MPT)"])

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

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        t_val = portfolio["現值_TWD"].sum(); t_cost = (portfolio["總投入成本"] * portfolio.apply(lambda r: usd_rate if r["幣別"]=="USD" else 1, axis=1)).sum()
        t_prof = t_val - t_cost
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}")
        c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}", f"{(t_prof/t_cost*100):.2f}%")
        c3.metric("💱 美金匯率", f"{usd_rate:.2f}")
        
        st.divider()
        for m, cur in [("🇹🇼 台股庫存", "TWD"), ("🇺🇸 美股庫存", "USD")]:
            m_df = portfolio[portfolio["幣別"] == cur]
            if not m_df.empty: display_market_table(m_df, m, cur, usd_rate, current_user)

    with tab2:
        target = st.selectbox("選擇分析標的：", portfolio["股票代號"].tolist())
        df_tech = yf.Ticker(target).history(period="1y")
        if not df_tech.empty:
            # 計算指標
            df_tech['MA20'] = df_tech['Close'].rolling(window=20).mean()
            df_tech['RSI'] = calculate_rsi(df_tech['Close'])
            df_tech['MACD'], df_tech['MACD_S'], df_tech['MACD_H'] = calculate_macd(df_tech['Close'])
            df_tech['Golden_Cross'] = (df_tech['MACD'] > df_tech['MACD_S']) & (df_tech['MACD'].shift(1) <= df_tech['MACD_S'].shift(1))
            
            # 共振檢查邏輯
            is_strong_buy = df_tech['Golden_Cross'].iloc[-1] and (df_tech['RSI'].iloc[-1] > 30 and df_tech['RSI'].shift(1).iloc[-1] <= 30) and (df_tech['Close'].iloc[-1] > df_tech['MA20'].iloc[-1])

            # 繪圖
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df_tech.index, open=df_tech['Open'], high=df_tech['High'], low=df_tech['Low'], close=df_tech['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech[df_tech['Golden_Cross']].index, y=df_tech[df_tech['Golden_Cross']]['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=15, color='gold'), name='MACD金叉'), row=1, col=1)
            fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_H'], name="MACD柱狀"), row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            if is_strong_buy:
                st.success("🔥 **強烈買入共振觸發！** 滿足 MACD金叉 + RSI低檔回升 + 站上月線。")
            else: st.info("目前技術指標處於觀察區。")

    with tab3:
        st.subheader("⚖️ MPT 組合優化")
        if st.button("🚀 啟動模擬計算"):
            with st.spinner("模擬中..."):
                res, err = perform_mpt_simulation(portfolio)
                if err: st.error(err)
                else:
                    st.session_state.mpt_results = res
        
        if st.session_state.mpt_results:
            res = st.session_state.mpt_results
            st.plotly_chart(px.scatter(res['sim_df'], x='Volatility', y='Return', color='Sharpe', title="效率前緣"), use_container_width=True)
            st.write("#### 建議配置比例")
            st.dataframe(res['comparison'].set_index("股票代號").style.format("{:.2f}%"))
else:
    st.info("尚無持股資料。")
