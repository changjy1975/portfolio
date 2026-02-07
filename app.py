import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import shutil
from datetime import datetime
import pytz
import numpy as np

# ==========================================
# 1. 初始化與全域設定
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None
if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心計算函數
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

# --- 技術指標與訊號 ---
def calculate_indicators(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    e1, e2 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = e1 - e2
    df['MACD_S'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_H'] = df['MACD'] - df['MACD_S']
    l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
    rsv = (df['Close'] - l9) / (h9 - l9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean(); df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

def get_ultra_signals(df):
    m_gold = (df['MACD'] > df['MACD_S']) & (df['MACD'].shift(1) <= df['MACD_S'].shift(1))
    m_dead = (df['MACD'] < df['MACD_S']) & (df['MACD'].shift(1) >= df['MACD_S'].shift(1))
    k_gold = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    buy = ( (df['Close'] > df['MA20']) & (m_gold | (k_gold & (df['K'] < 30))) )
    sell = ( (df['Close'] < df['MA5']) & (m_dead | (df['RSI'] > 75)) ) | ( (df['Close'].shift(1) > df['MA20']) & (df['Close'] < df['MA20']) )
    return buy, sell

# --- 歷史回測引擎 ---
@st.cache_data(ttl=3600)
def get_backtest_data(symbols):
    if not symbols: return pd.DataFrame()
    data = yf.download(symbols + ["USDTWD=X"], period="1y", interval="1d", progress=False)['Close']
    return data.ffill()

# ==========================================
# 3. 介面組件
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, title, currency, current_user):
    st.subheader(title)
    h_cols = st.columns(COLS_RATIO)
    labels = ["代號", "股數", "均價", "現價", "總成本", "現值", "獲利", "報酬率"]
    keys = ["股票代號", "股數", "平均持有單價", "最新股價", "總投入成本", "現值", "獲利", "獲利率(%)"]
    for i, (l, k) in enumerate(zip(labels, keys)):
        arrow = " ▲" if st.session_state.sort_col == k and st.session_state.sort_asc else " ▼" if st.session_state.sort_col == k else ""
        if h_cols[i].button(f"{l}{arrow}", key=f"h_{currency}_{k}_{current_user}"):
            if st.session_state.sort_col == k: st.session_state.sort_asc = not st.session_state.sort_asc
            else: st.session_state.sort_col, st.session_state.sort_asc = k, False
            st.rerun()
    
    s_cost, s_val, s_prof = df["總投入成本"].sum(), df["現值"].sum(), df["獲利"].sum()
    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in df_sorted.iterrows():
        r = st.columns(COLS_RATIO); fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"
        color = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}"); r[4].write(fmt.format(row['總投入成本'])); r[5].write(fmt.format(row['現值'])); r[6].markdown(f":{color}[{fmt.format(row['獲利'])}]"); r[7].markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
            full = load_data(current_user); save_data(full[full["股票代號"] != row['股票代號']], current_user); st.rerun()

    st.markdown("---")
    f_cols = st.columns(COLS_RATIO); f_fmt, f_c = ("{:,.0f}" if currency == "TWD" else "{:,.2f}"), ("red" if s_prof > 0 else "green")
    f_cols[0].write(f"**[{currency} 小計]**"); f_cols[4].write(f"**{f_fmt.format(s_cost)}**"); f_cols[5].write(f"**{f_fmt.format(s_val)}**"); f_cols[6].markdown(f"**:{f_c}[{f_fmt.format(s_prof)}]**"); f_cols[7].markdown(f"**:{f_c}[{(s_prof/s_cost*100 if s_cost!=0 else 0):.2f}%]**")

# ==========================================
# 4. 主程式
# ==========================================
with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("使用者", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add"):
            s_in = st.text_input("代號").upper().strip()
            q_in, c_in = st.number_input("股數", min_value=0.0), st.number_input("成本", min_value=0.0)
            if st.form_submit_button("新增"):
                if s_in:
                    d = load_data(current_user); save_data(pd.concat([d, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user); st.rerun()

df_raw = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3 = st.tabs(["📊 配置與績效", "🧠 技術診斷", "⚖️ 組合優化"])

if not df_raw.empty:
    rate = get_exchange_rate()
    df_raw['幣別'] = df_raw['股票代號'].apply(identify_currency)
    portfolio = df_raw.groupby(["股票代號", "幣別"]).apply(lambda g: pd.Series({'股數': g['股數'].sum(), '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()}), include_groups=False).reset_index()
    q_map = get_latest_quotes(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(q_map)
    portfolio["總投入成本"], portfolio["現值"] = portfolio["股數"] * portfolio["平均持有單價"], portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]) * 100
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (rate if r["幣別"]=="USD" else 1), axis=1)

    with tab1:
        if st.button("🔄 更新最新報價", use_container_width=True): st.cache_data.clear(); st.rerun()
        t_v = portfolio["現值_TWD"].sum(); t_p = portfolio.apply(lambda r: (r["獲利"] * rate) if r["幣別"]=="USD" else r["獲利"], axis=1).sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_v:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_p:,.0f}"); c3.metric("📊 總報酬率", f"{(t_p/(t_v-t_p)*100 if t_v!=t_p else 0):.2f}%"); c4.metric("💱 匯率", f"{rate:.2f}")

        # --- 新增：圓餅圖配置 ---
        st.divider(); st.subheader("🎯 投資組合配置分析")
        pc1, pc2 = st.columns([1, 2])
        with pc1:
            view_mode = st.selectbox("配置範圍：", ["全部", "台股", "美股"], key="pie_select")
        with pc2:
            chart_df = portfolio[portfolio["幣別"] == ("TWD" if view_mode == "台股" else "USD")] if view_mode != "全部" else portfolio
            if not chart_df.empty:
                st.plotly_chart(px.pie(chart_df, values="現值_TWD", names="股票代號", title=f"個股配置 ({view_mode})", hole=0.4), use_container_width=True)

        # --- 新增：1年歷史回測 ---
        st.divider(); st.subheader("📈 歷史淨值回測 (過去一年模擬)")
        hist_p = get_backtest_data(portfolio["股票代號"].tolist())
        if not hist_p.empty:
            equity_curve = pd.Series(0.0, index=hist_p.index)
            fx_hist = hist_p["USDTWD=X"].ffill()
            for _, row in portfolio.iterrows():
                p_hist = hist_p[row["股票代號"]].ffill()
                mult = fx_hist if row["幣別"] == "USD" else 1.0
                equity_curve += p_hist * row["股數"] * mult
            fig_h = go.Figure(data=go.Scatter(x=equity_curve.index, y=equity_curve, name="組合淨值", line=dict(color='#00D1FF', width=3)))
            fig_h.update_layout(height=400, template="plotly_dark", hovermode='x unified', margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_h, use_container_width=True)

        st.divider()
        for m, cur in [("🇹🇼 台股庫存", "TWD"), ("🇺🇸 美股庫存", "USD")]:
            m_df = portfolio[portfolio["幣別"] == cur]
            if not m_df.empty: display_market_table(m_df, m, cur, current_user)

    with tab2:
        # ... (保留穩定版技術診斷邏輯)
        pass

    with tab3:
        # ... (保留穩定版 MPT 引擎邏輯)
        pass
else:
    st.info("請先新增持股。")
