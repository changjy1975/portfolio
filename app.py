import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import shutil
from datetime import datetime
import numpy as np

# ==========================================
# 1. 初始化設定與全域配置
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None
if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 數據核心函數
# ==========================================

def load_data(user):
    path = f"portfolio_{user}.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    source_path = f"portfolio_{user}.csv"
    if os.path.exists(source_path):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
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

# --- 技術指標計算 ---
def calculate_indicators(df):
    # 均線族群
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
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

def get_refined_signals(df):
    """精準訊號濾鏡：改採事件觸發邏輯以減少重疊訊號"""
    # 交叉事件
    m_gold = (df['MACD'] > df['MACD_S']) & (df['MACD'].shift(1) <= df['MACD_S'].shift(1))
    m_dead = (df['MACD'] < df['MACD_S']) & (df['MACD'].shift(1) >= df['MACD_S'].shift(1))
    k_gold = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    
    # 買進：趨勢向上且 (MACD金叉 或 KD低階金叉)
    buy = ( (df['Close'] > df['MA20']) & (df['MA20'] > df['MA60']) & (m_gold | (k_gold & (df['K'] < 40))) )
    
    # 賣出：
    # 1. MACD死叉且破5日線
    s1 = (df['Close'] < df['MA5']) & m_dead
    # 2. RSI 剛進入超買區 (>78) - 改為事件判定
    s2 = (df['RSI'] > 78) & (df['RSI'].shift(1) <= 78)
    # 3. 收盤剛跌破 20 日線 - 事件判定
    s3 = (df['Close'].shift(1) > df['MA20']) & (df['Close'] < df['MA20'])
    
    sell = s1 | s2 | s3
    return buy, sell

# --- 歷史回測與 MPT ---
@st.cache_data(ttl=3600)
def fetch_backtest_data(symbols, period="1y"):
    if not symbols: return pd.DataFrame()
    data = yf.download(symbols + ["USDTWD=X"], period=period, interval="1d", progress=False)['Close']
    return data.ffill()

def perform_mpt_simulation(portfolio_df, symbols):
    try:
        data = yf.download(symbols, period="3y", interval="1d", progress=False)['Close'].ffill().dropna()
        returns = data.pct_change().dropna()
        mean_rets, cov_mat = returns.mean() * 252, returns.cov() * 252
        num_p = 2000; results = np.zeros((3, num_p)); w_rec = []
        for i in range(num_p):
            w = np.random.random(len(symbols)); w /= np.sum(w); w_rec.append(w)
            results[0,i] = np.sum(w * mean_rets)
            results[1,i] = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            results[2,i] = (results[0,i] - 0.02) / results[1,i]
        idx = np.argmax(results[2])
        comp = pd.DataFrame({"股票代號": symbols, "目前權重 (%)": (portfolio_df["現值_TWD"] / portfolio_df["現值_TWD"].sum() * 100).values, "建議權重 (%)": w_rec[idx] * 100})
        return {"sim_df": pd.DataFrame({'Return': results[0], 'Volatility': results[1], 'Sharpe': results[2]}), "comparison": comp, "max_s": (results[0, idx], results[1, idx]), "corr": returns.corr()}, None
    except Exception as e: return None, str(e)

# ==========================================
# 3. 介面呈現組件
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def render_table(df, currency, current_user):
    h_cols = st.columns(COLS_RATIO)
    labels = ["代號", "股數", "均價", "現價", "總成本", "現值", "獲利", "報酬率"]
    keys = ["股票代號", "股數", "平均持有單價", "最新股價", "總投入成本", "現值", "獲利", "獲利率(%)"]
    for i, (l, k) in enumerate(zip(labels, keys)):
        arrow = " ▲" if st.session_state.sort_col == k and st.session_state.sort_asc else " ▼" if st.session_state.sort_col == k else ""
        if h_cols[i].button(f"{l}{arrow}", key=f"h_{currency}_{k}_{current_user}"):
            st.session_state.sort_asc = not st.session_state.sort_asc if st.session_state.sort_col == k else False
            st.session_state.sort_col = k; st.rerun()
    
    s_cost, s_val, s_prof = df["總投入成本"].sum(), df["現值"].sum(), df["獲利"].sum()
    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in df_sorted.iterrows():
        r = st.columns(COLS_RATIO); fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"
        clr = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}"); r[4].write(fmt.format(row['總投入成本'])); r[5].write(fmt.format(row['現值'])); r[6].markdown(f":{clr}[{fmt.format(row['獲利'])}]"); r[7].markdown(f":{clr}[{row['獲利率(%)']:.2f}%]")
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
            d = load_data(current_user); save_data(d[d["股票代號"] != row['股票代號']], current_user); st.rerun()
    st.markdown("---")
    f_cols = st.columns(COLS_RATIO); f_fmt, f_c = ("{:,.0f}" if currency == "TWD" else "{:,.2f}"), ("red" if s_prof > 0 else "green")
    f_cols[0].write(f"**[{currency} 小計]**"); f_cols[4].write(f"**{f_fmt.format(s_cost)}**"); f_cols[5].write(f"**{f_fmt.format(s_val)}**"); f_cols[6].markdown(f"**:{f_c}[{f_fmt.format(s_prof)}]**"); f_cols[7].markdown(f"**:{f_c}[{(s_prof/s_cost*100 if s_cost!=0 else 0):.2f}%]**")

# ==========================================
# 4. 主程式執行邏輯
# ==========================================
with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add"):
            s_in = st.text_input("股票代號").upper().strip()
            q_in, c_in = st.number_input("股數", min_value=0.0), st.number_input("成本", min_value=0.0)
            if st.form_submit_button("新增持股"):
                if s_in:
                    d = load_data(current_user); save_data(pd.concat([d, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user); st.rerun()

df_raw = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3 = st.tabs(["📊 庫存配置與回測", "🧠 技術診斷", "⚖️ 組合優化"])

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
        st.button("🔄 更新最新報價", on_click=lambda: st.cache_data.clear(), use_container_width=True)
        t_v = portfolio["現值_TWD"].sum(); t_p = portfolio.apply(lambda r: (r["獲利"] * rate) if r["幣別"]=="USD" else r["獲利"], axis=1).sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_v:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_p:,.0f}"); c3.metric("📊 總報酬率", f"{(t_p/(t_v-t_p)*100 if t_v!=t_p else 0):.2f}%"); c4.metric("💱 匯率", f"{rate:.2f}")

        st.divider(); cp1, cp2 = st.columns([1, 1.5])
        with cp1:
            st.subheader("🌐 市場資產比例")
            m_dist = portfolio.groupby("幣別")["現值_TWD"].sum().reset_index()
            st.plotly_chart(px.pie(m_dist, values="現值_TWD", names="幣別", hole=0.5, color="幣別", color_discrete_map={"TWD": "#FF4B4B", "USD": "#00D1FF"}), use_container_width=True)
        with cp2:
            st.subheader("🎯 個股配置分析")
            v_mode = st.radio("範圍", ["全部", "台股", "美股"], horizontal=True, label_visibility="collapsed")
            p_df = portfolio[portfolio["幣別"] == ("TWD" if v_mode == "台股" else "USD")] if v_mode != "全部" else portfolio
            if not p_df.empty: 
                st.plotly_chart(px.pie(p_df, values="現值_TWD", names="股票代號", hole=0.4, color_discrete_sequence=px.colors.qualitative.Vivid), use_container_width=True)

        st.divider()
        tw_p = portfolio[portfolio["幣別"] == "TWD"]
        if not tw_p.empty: render_table(tw_p, "TWD", current_user)
        st.divider()
        us_p = portfolio[portfolio["幣別"] == "USD"]
        if not us_p.empty: render_table(us_p, "USD", current_user)

        st.divider(); st.subheader("📈 組合淨值 1 年歷史回測")
        h_df = fetch_backtest_data(portfolio["股票代號"].tolist())
        if not h_df.empty:
            eq = pd.Series(0.0, index=h_df.index); fx = h_df["USDTWD=X"].ffill()
            for _, r in portfolio.iterrows():
                eq += h_df[r["股票代號"]].ffill() * r["股數"] * (fx if r["幣別"]=="USD" else 1.0)
            st.plotly_chart(go.Figure(data=go.Scatter(x=eq.index, y=eq, line=dict(color='#00D1FF', width=3))).update_layout(height=400, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10)), use_container_width=True)

    with tab2:
        target = st.selectbox("分析標的", portfolio["股票代號"].tolist())
        period = st.select_slider("資料範圍", options=["1mo", "3mo", "6mo", "1y"], value="1y")
        df_t = yf.Ticker(target).history(period=period)
        if not df_t.empty:
            df_t = calculate_indicators(df_t); df_t['Buy'], df_t['Sell'] = get_refined_signals(df_t)
            lc = df_t['Close'].iloc[-1]; sl, tp = lc - (2*df_t['ATR'].iloc[-1]), lc + (3.5*df_t['ATR'].iloc[-1])
            
            # --- 四層式圖表 ---
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                                row_heights=[0.4, 0.2, 0.2, 0.2],
                                subplot_titles=("價格與 EMA 均線", "MACD 指標", "RSI 強弱勢", "KD 隨機指標"))

            # Row 1: K線 + EMA
            fig.add_trace(go.Candlestick(x=df_t.index, open=df_t['Open'], high=df_t['High'], low=df_t['Low'], close=df_t['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['EMA10'], line=dict(color='orange', width=1.5), name='EMA10'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['EMA20'], line=dict(color='cyan', width=1.5), name='EMA20'), row=1, col=1)
            
            # 買賣點與止盈止損
            b, s = df_t[df_t['Buy']], df_t[df_t['Sell']]
            fig.add_trace(go.Scatter(x=b.index, y=b['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=14, color='lime'), name='買入'), row=1, col=1)
            fig.add_trace(go.Scatter(x=s.index, y=s['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=14, color='red'), name='賣出'), row=1, col=1)
            fig.add_hline(y=sl, line_dash="dash", line_color="red", row=1, col=1); fig.add_hline(y=tp, line_dash="dash", line_color="lime", row=1, col=1)

            # Row 2: MACD
            fig.add_trace(go.Bar(x=df_t.index, y=df_t['MACD_H'], marker_color=['red' if v<0 else 'green' for v in df_t['MACD_H']], name='MACD柱'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['MACD'], line=dict(color='white', width=1.2), name='MACD快線'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['MACD_S'], line=dict(color='yellow', width=1), name='訊號線'), row=2, col=1)

            # Row 3: RSI
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['RSI'], line=dict(color='#E377C2', width=2), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", line_width=1, row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", line_width=1, row=3, col=1)

            # Row 4: KD
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['K'], line=dict(color='white', width=1.2), name='K值'), row=4, col=1)
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['D'], line=dict(color='yellow', width=1.2), name='D值'), row=4, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="gray", line_width=0.5, row=4, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="gray", line_width=0.5, row=4, col=1)

            fig.update_layout(height=1000, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if st.button("🚀 執行 MPT 優化"):
            with st.spinner("模擬計算中..."):
                res, err = perform_mpt_simulation(portfolio, portfolio["股票代號"].tolist())
                if err: st.error(err)
                else: st.session_state.mpt_results = res
        if st.session_state.mpt_results:
            r = st.session_state.mpt_results; ca, cb = st.columns([2, 1])
            with ca: st.plotly_chart(px.scatter(r['sim_df'], x='Volatility', y='Return', color='Sharpe', title="效率前緣").add_trace(go.Scatter(x=[r['max_s'][1]], y=[r['max_s'][0]], mode='markers', marker=dict(color='red', size=15, symbol='star'))), use_container_width=True)
            with cb: st.write("#### ⚖️ 配置建議"); st.dataframe(r['comparison'].set_index("股票代號").style.format("{:.2f}%"))
            st.divider(); st.write("#### 🔗 相關性矩陣"); st.plotly_chart(px.imshow(r['corr'], text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
else:
    st.info("請先新增持股資料。")
