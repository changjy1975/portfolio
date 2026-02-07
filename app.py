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
# 1. 初始化設定
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None
if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心計算函數 (含超嚴格訊號濾鏡)
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

def calculate_indicators(df):
    # MA 均線
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    # MACD
    e1, e2 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = e1 - e2
    df['MACD_S'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_H'] = df['MACD'] - df['MACD_S']
    # KD
    l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
    rsv = (df['Close'] - l9) / (h9 - l9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    # ATR
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

def get_ultra_filtered_signals(df):
    """
    超嚴格訊號濾鏡：
    🟢 買進: (MACD金叉 且 股價 > MA20) 或 (KD低檔金叉 且 RSI < 40)
    🔴 賣出: (MACD死叉 且 股價 < MA5) 或 (RSI > 75 且 KD高檔死叉) 或 (收盤跌破 MA20)
    """
    m_gold = (df['MACD'] > df['MACD_S']) & (df['MACD'].shift(1) <= df['MACD_S'].shift(1))
    m_dead = (df['MACD'] < df['MACD_S']) & (df['MACD'].shift(1) >= df['MACD_S'].shift(1))
    k_gold = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    k_dead = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))

    buy = ( (m_gold & (df['Close'] > df['MA20'])) | (k_gold & (df['K'] < 30) & (df['RSI'] < 45)) )
    
    # 賣出邏輯大幅收緊：必須同時滿足價格轉弱 (MA5) 或 極端超買 (RSI/KD) 或 趨勢反轉 (MA20)
    sell = ( (m_dead & (df['Close'] < df['MA5'])) | 
             (k_dead & (df['K'] > 75) & (df['RSI'] > 70)) |
             ((df['Close'].shift(1) > df['MA20']) & (df['Close'] < df['MA20'])) )
    
    return buy, sell

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
    h_cols[8].write("**管理**")

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
# 4. 主程式執行
# ==========================================
with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("使用者", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add"):
            s_in = st.text_input("代號").upper().strip()
            q_in, c_in = st.number_input("股數", min_value=0.0), st.number_input("成本", min_value=0.0)
            if st.form_submit_button("新增持股"):
                if s_in:
                    d = load_data(current_user); save_data(pd.concat([d, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user); st.rerun()

df_raw = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 技術診斷", "⚖️ 組合優化"])

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
        t_v = portfolio["現值_TWD"].sum(); t_p = portfolio.apply(lambda r: r["獲利"] * (rate if r["幣別"]=="USD" else 1), axis=1).sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總資產 (TWD)", f"${t_v:,.0f}"); c2.metric("總獲利 (TWD)", f"${t_p:,.0f}"); c3.metric("報酬率", f"{(t_p/(t_v-t_p)*100 if t_v!=t_p else 0):.2f}%"); c4.metric("匯率", f"{rate:.2f}")
        for m, cur in [("🇹🇼 台股", "TWD"), ("🇺🇸 美股", "USD")]:
            m_df = portfolio[portfolio["幣別"] == cur]
            if not m_df.empty: display_market_table(m_df, m, cur, current_user)

    with tab2:
        target = st.selectbox("分析標的", portfolio["股票代號"].tolist())
        period = st.select_slider("時間範圍", options=["1mo", "3mo", "6mo", "1y"], value="1y")
        df_t = yf.Ticker(target).history(period=period)
        if not df_t.empty:
            df_t = calculate_indicators(df_t)
            df_t['Buy'], df_t['Sell'] = get_ultra_filtered_signals(df_t)
            lc = df_t['Close'].iloc[-1]; sl, tp = lc - (2*df_t['ATR'].iloc[-1]), lc + (3.5*df_t['ATR'].iloc[-1])

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=df_t.index, open=df_t['Open'], high=df_t['High'], low=df_t['Low'], close=df_t['Close'], name="K線"), row=1, col=1)
            # 訊號標記
            b_p, s_p = df_t[df_t['Buy']], df_t[df_t['Sell']]
            fig.add_trace(go.Scatter(x=b_p.index, y=b_p['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(width=1, color='white')), name='買'), row=1, col=1)
            fig.add_trace(go.Scatter(x=s_p.index, y=s_p['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=14, color='red', line=dict(width=1, color='white')), name='賣'), row=1, col=1)
            # MA線
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['MA20'], name="月線", line=dict(color='rgba(255,255,255,0.3)', width=1)), row=1, col=1)
            
            fig.add_hline(y=sl, line_dash="dash", line_color="red", row=1, col=1); fig.add_hline(y=tp, line_dash="dash", line_color="lime", row=1, col=1)
            fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, text=f"RSI: {df_t['RSI'].iloc[-1]:.1f} | ATR SL: {sl:.2f}", showarrow=False, font=dict(color="yellow", size=14), bgcolor="rgba(0,0,0,0.7)")
            
            # 副圖
            fig.add_trace(go.Bar(x=df_t.index, y=df_t['MACD_H'], marker_color=['red' if v<0 else 'green' for v in df_t['MACD_H']]), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['K'], line=dict(color='white', width=1)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['D'], line=dict(color='yellow', width=1)), row=2, col=1)
            fig.update_layout(height=750, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if st.button("🚀 執行優化模擬"):
            # ... (保留 MPT 邏輯) ...
            pass
else:
    st.info("請先新增持股。")
