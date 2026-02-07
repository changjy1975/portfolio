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

# --- 技術指標 ---
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

def calculate_bb(series, window=20):
    ma = series.rolling(window=window).mean(); std = series.rolling(window=window).std()
    return ma + (std * 2), ma, ma - (std * 2)

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift())
    low_cp = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# --- MPT 引擎修復版 ---
def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "至少需要 2 支標的。"
    try:
        # 下載歷史數據並統一處理多重索引
        raw_data = yf.download(symbols, period="3y", interval="1d")
        if 'Close' in raw_data.columns: data = raw_data['Close']
        else: data = raw_data
        
        # 確保資料格式一致
        if isinstance(data, pd.Series): data = data.to_frame()
        data = data.ffill().dropna() # 剔除無交集時段
        
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        num_portfolios = 2000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            w = np.random.random(len(symbols))
            w /= np.sum(w)
            weights_record.append(w)
            p_ret = np.sum(w * mean_returns)
            p_std = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            results[0,i] = p_ret
            results[1,i] = p_std
            results[2,i] = (p_ret - 0.02) / p_std # Sharpe
            
        max_idx = np.argmax(results[2]); min_idx = np.argmin(results[1])
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
# 3. 介面組件
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, title, currency, current_user):
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

    # 計算區域小計
    s_cost = df["總投入成本"].sum(); s_val = df["現值"].sum(); s_prof = df["獲利"].sum()
    s_roi = (s_prof / s_cost * 100) if s_cost != 0 else 0
    
    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in df_sorted.iterrows():
        r = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"
        color = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}"); r[4].write(fmt.format(row['總投入成本'])); r[5].write(fmt.format(row['現值'])); r[6].markdown(f":{color}[{fmt.format(row['獲利'])}]"); r[7].markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
            full = load_data(current_user); save_data(full[full["股票代號"] != row['股票代號']], current_user); st.rerun()

    st.markdown("---")
    f_cols = st.columns(COLS_RATIO)
    f_fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"
    f_color = "red" if s_prof > 0 else "green"
    f_cols[0].write(f"**[{currency} 小計]**"); f_cols[4].write(f"**{f_fmt.format(s_cost)}**"); f_cols[5].write(f"**{f_fmt.format(s_val)}**"); f_cols[6].markdown(f"**:{f_color}[{f_fmt.format(s_prof)}]**"); f_cols[7].markdown(f"**:{f_color}[{s_roi:.2f}%]**")

# ==========================================
# 4. 主頁面邏輯
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
tab1, tab2, tab3 = st.tabs(["📊 庫存配置與績效", "🧠 技術診斷", "⚖️ 組合分析 (MPT)"])

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
        if st.button("🔄 點擊更新最新報價", use_container_width=True):
            st.cache_data.clear(); st.rerun()

        t_val = float(portfolio["現值_TWD"].sum()); t_prof = portfolio.apply(lambda r: r["獲利"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1).sum()
        roi = (t_prof / (t_val - t_prof) * 100) if (t_val - t_prof) != 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}")
        c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}")
        c3.metric("📊 總報酬率", f"{roi:.2f}%")
        c4.metric("💱 匯率", f"{usd_rate:.2f}")

        st.divider()
        for m, cur in [("🇹🇼 台股庫存", "TWD"), ("🇺🇸 美股庫存", "USD")]:
            m_df = portfolio[portfolio["幣別"] == cur]
            if not m_df.empty: display_market_table(m_df, m, cur, current_user)

    with tab2:
        target = st.selectbox("選擇分析標的：", portfolio["股票代號"].tolist())
        df_tech = yf.Ticker(target).history(period="1y")
        if not df_tech.empty:
            df_tech['RSI'] = calculate_rsi(df_tech['Close'])
            df_tech['ATR'] = calculate_atr(df_tech)
            df_tech['BB_U'], df_tech['BB_M'], df_tech['BB_L'] = calculate_bb(df_tech['Close'])
            df_tech['MACD'], df_tech['MACD_S'], df_tech['MACD_H'] = calculate_macd(df_tech['Close'])
            last_c = df_tech['Close'].iloc[-1]; last_rsi = df_tech['RSI'].iloc[-1]
            sl = last_c - (2 * df_tech['ATR'].iloc[-1]); tp = last_c + (3 * df_tech['ATR'].iloc[-1])

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            # 主圖
            fig.add_trace(go.Candlestick(x=df_tech.index, open=df_tech['Open'], high=df_tech['High'], low=df_tech['Low'], close=df_tech['Close'], name="K線"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_U'], name="BB上", line=dict(color='rgba(173,216,230,0.3)', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_L'], name="BB下", line=dict(color='rgba(173,216,230,0.3)', dash='dot')), row=1, col=1)
            fig.add_hline(y=sl, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=tp, line_dash="dash", line_color="lime", row=1, col=1)
            # 圖內資訊標註
            fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.95, text=f"RSI: {last_rsi:.1f}", showarrow=False, font=dict(size=18, color="yellow"), bgcolor="rgba(0,0,0,0.6)")
            fig.add_annotation(x=df_tech.index[-1], y=sl, text=f" SL:{sl:.2f}", showarrow=False, align="left", font=dict(color="red"), xanchor="left", row=1, col=1)
            fig.add_annotation(x=df_tech.index[-1], y=tp, text=f" TP:{tp:.2f}", showarrow=False, align="left", font=dict(color="lime"), xanchor="left", row=1, col=1)
            # MACD
            m_clrs = ['red' if v < 0 else 'green' for v in df_tech['MACD_H']]
            fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_H'], marker_color=m_clrs), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], line=dict(color='white', width=1)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_S'], line=dict(color='yellow', width=1)), row=2, col=1)
            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if st.button("🚀 執行 MPT 模擬", type="primary"):
            with st.spinner("計算中..."):
                res, err = perform_mpt_simulation(portfolio)
                if err: st.error(err)
                else: st.session_state.mpt_results = res
        if st.session_state.mpt_results:
            r = st.session_state.mpt_results
            ca, cb = st.columns([2, 1])
            with ca:
                f_mpt = px.scatter(r['sim_df'], x='Volatility', y='Return', color='Sharpe', title="效率前緣雲圖")
                f_mpt.add_trace(go.Scatter(x=[r['max_sharpe'][1]], y=[r['max_sharpe'][0]], mode='markers', marker=dict(color='red', size=15, symbol='star')))
                st.plotly_chart(f_mpt, use_container_width=True)
            with cb:
                st.write("#### ⚖️ 配置建議"); st.dataframe(r['comparison'].set_index("股票代號").style.format("{:.2f}%"))
            st.divider(); st.write("#### 🔗 相關性矩陣"); st.plotly_chart(px.imshow(r['corr'], text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
else:
    st.info("尚無資料。")
