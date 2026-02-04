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
for d in [BACKUP_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# ==========================================
# 2. 核心功能函數 (資料處理、行情與快照)
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

def update_daily_snapshot(user, total_val, total_profit, rate):
    """每日資產快照邏輯：每天僅記錄一次最準確的收盤/現狀"""
    path = f"history_{user}.csv"
    today = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d")
    
    if os.path.exists(path):
        history_df = pd.read_csv(path)
        last_date = history_df['日期'].iloc[-1] if not history_df.empty else None
    else:
        history_df = pd.DataFrame(columns=["日期", "總資產", "總獲利", "匯率"])
        last_date = None

    if last_date != today:
        new_record = pd.DataFrame([{"日期": today, "總資產": total_val, "總獲利": total_profit, "匯率": rate}])
        history_df = pd.concat([history_df, new_record], ignore_index=True)
        history_df.to_csv(path, index=False)
    return history_df if os.path.exists(path) else None

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

@st.cache_data(ttl=3600)
def get_backtest_data(symbols):
    """抓取過去一年的歷史價格進行回測"""
    if not symbols: return pd.DataFrame()
    data = yf.download(symbols + ["USDTWD=X"], period="1y", interval="1d")['Close']
    return data.ffill()

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# --- 技術指標計算 ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean(); exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2; signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def calculate_bb(series, window=20):
    ma = series.rolling(window=window).mean(); std = series.rolling(window=window).std()
    return ma + (std * 2), ma, ma - (std * 2)

# ==========================================
# 3. 介面顯示與主邏輯
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
    h_cols[8].write("**管理**")

    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in df_sorted.iterrows():
        r = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"
        color = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}"); r[4].write(fmt.format(row['總投入成本'])); r[5].write(fmt.format(row['現值'])); r[6].markdown(f":{color}[{fmt.format(row['獲利'])}]"); r[7].markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
            full = load_data(current_user); save_data(full[full["股票代號"] != row['股票代號']], current_user); st.rerun()

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
tab1, tab2, tab3 = st.tabs(["📊 庫存配置與績效", "🧠 技術健診", "⚖️ 組合分析 (MPT)"])

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

    # 執行快照
    if current_user != "All":
        update_daily_snapshot(current_user, portfolio["現值_TWD"].sum(), portfolio["獲利_TWD"].sum(), usd_rate)

    with tab1:
        # A. 概覽數據
        t_val = float(portfolio["現值_TWD"].sum()); t_prof = float(portfolio["獲利_TWD"].sum())
        roi = (t_prof / (t_val - t_prof) * 100) if (t_val - t_prof) != 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}"); c3.metric("📊 總報酬率", f"{roi:.2f}%"); c4.metric("💱 匯率", f"{usd_rate:.2f}")

        # B. 歷史績效追蹤 (回測曲線)
        st.divider(); st.subheader("📈 歷史淨值回測 (過去一年模擬)")
        hist_prices = get_backtest_data(portfolio["股票代號"].tolist())
        if not hist_prices.empty:
            # 計算每日投資組合價值 (台幣)
            equity_curve = pd.Series(0.0, index=hist_prices.index)
            fx_hist = hist_prices["USDTWD=X"].ffill()
            for _, row in portfolio.iterrows():
                p_hist = hist_prices[row["股票代號"]].ffill()
                multiplier = fx_hist if row["幣別"] == "USD" else 1.0
                equity_curve += p_hist * row["股數"] * multiplier
            
            # 建立圖表
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name="組合淨值", line=dict(color='#00D1FF', width=3)))
            
            # 加入 2.65% 資金成本基準線 (假設起點為一年前的價值)
            start_val = equity_curve.iloc[0]
            days = (equity_curve.index - equity_curve.index[0]).days
            cost_line = start_val * (1 + 0.0265 * (days / 365))
            fig_hist.add_trace(go.Scatter(x=equity_curve.index, y=cost_line, name="借貸成本基準 (2.65%)", line=dict(color='gray', dash='dash')))
            
            fig_hist.update_layout(height=400, template="plotly_dark", hovermode='x unified', margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("註：基準線是以一年前投資組合價值為起點，按 2.65% 年利率計算之資金成本。")

        # C. 圓餅圖配置
        st.divider(); st.subheader("🎯 投資組合配置分析")
        pc1, pc2 = st.columns(2)
        with pc1: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="市場配置 (TWD)", hole=0.45), use_container_width=True)
        with pc2:
            view_mode = st.selectbox("選擇個股配置範圍：", ["全部", "台股", "美股"], key="pie_filter")
            chart_df = portfolio[portfolio["幣別"] == ( "TWD" if view_mode == "台股" else "USD" )] if view_mode != "全部" else portfolio
            if not chart_df.empty: st.plotly_chart(px.pie(chart_df, values="現值_TWD", names="股票代號", title=f"個股配置 ({view_mode})", hole=0.45), use_container_width=True)
            else: st.info(f"目前沒有 {view_mode} 的資料。")

        # D. 庫存列表
        st.divider()
        for m, cur in [("🇹🇼 台股庫存", "TWD"), ("🇺🇸 美股庫存", "USD")]:
            m_df = portfolio[portfolio["幣別"] == cur]
            if not m_df.empty: display_market_table(m_df, m, cur, usd_rate, current_user)

    with tab2:
        target = st.selectbox("選擇分析標的：", portfolio["股票代號"].tolist())
        df_tech = yf.Ticker(target).history(period="1y")
        if not df_tech.empty:
            df_tech['RSI'] = calculate_rsi(df_tech['Close']); df_tech['BB_U'], _, df_tech['BB_L'] = calculate_bb(df_tech['Close'])
            df_tech['MACD'], df_tech['MACD_S'], df_tech['MACD_H'] = calculate_macd(df_tech['Close'])
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['Close'], name="收盤價", line=dict(color='#00D1FF')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_U'], name="上軌", line=dict(dash='dot', color='rgba(255, 82, 82, 0.8)')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_L'], name="下軌", line=dict(dash='dot', color='rgba(76, 175, 80, 0.8)')), row=1, col=1)
            macd_colors = ['#FF5252' if val < 0 else '#4CAF50' for val in df_tech['MACD_H']]
            fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_H'], name="MACD", marker_color=macd_colors), row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("尚無持股資料，請從側邊欄新增。")
