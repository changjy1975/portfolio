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
# 2. 核心功能函數
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
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean(); exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2; signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def calculate_bb(series, window=20):
    ma = series.rolling(window=window).mean(); std = series.rolling(window=window).std()
    return ma + (std * 2), ma, ma - (std * 2)

# --- MPT 引擎 ---
def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "至少需要 2 支標的才能進行優化模擬。"
    try:
        data = yf.download(symbols, period="3y", interval="1d", auto_adjust=True)['Close']
        returns = data.ffill().pct_change().dropna()
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

    st.markdown("---")
    scost = df["總投入成本"].sum(); sval = df["現值"].sum(); sprof = df["獲利"].sum()
    sroi = (sprof / scost * 100) if scost != 0 else 0
    f_cols = st.columns(COLS_RATIO)
    scol = "red" if sprof > 0 else "green"
    f_cols[0].write("**[ 小計 ]**"); f_cols[4].write(f"**{fmt.format(scost)}**"); f_cols[5].write(f"**{fmt.format(sval)}**"); f_cols[6].markdown(f"**:{scol}[{fmt.format(sprof)}]**"); f_cols[7].markdown(f"**:{scol}[{sroi:.2f}%]**")

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

    if current_user != "All": update_daily_snapshot(current_user, portfolio["現值_TWD"].sum(), portfolio["獲利_TWD"].sum(), usd_rate)

    with tab1:
        t_val = float(portfolio["現值_TWD"].sum()); t_prof = float(portfolio["獲利_TWD"].sum())
        roi = (t_prof / (t_val - t_prof) * 100) if (t_val - t_prof) != 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}"); c3.metric("📊 總報酬率", f"{roi:.2f}%"); c4.metric("💱 匯率", f"{usd_rate:.2f}")

        st.divider(); st.subheader("🎯 投資組合配置分析")
        pc1, pc2 = st.columns(2)
        with pc1: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="市場配置 (TWD)", hole=0.45), use_container_width=True)
        with pc2:
            view_mode = st.selectbox("選擇個股配置範圍：", ["全部", "台股", "美股"], key="pie_filter")
            chart_df = portfolio[portfolio["幣別"] == ( "TWD" if view_mode == "台股" else "USD" )] if view_mode != "全部" else portfolio
            if not chart_df.empty: st.plotly_chart(px.pie(chart_df, values="現值_TWD", names="股票代號", title=f"個股配置 ({view_mode})", hole=0.45), use_container_width=True)
            else: st.info(f"目前沒有 {view_mode} 的資料。")

        st.divider()
        for m, cur in [("🇹🇼 台股庫存", "TWD"), ("🇺🇸 美股庫存", "USD")]:
            m_df = portfolio[portfolio["幣別"] == cur]
            if not m_df.empty: display_market_table(m_df, m, cur, usd_rate, current_user)

        st.divider(); st.subheader("📈 歷史淨值回測 (過去一年模擬)")
        hist_prices = get_backtest_data(portfolio["股票代號"].tolist())
        if not hist_prices.empty:
            equity_curve = pd.Series(0.0, index=hist_prices.index)
            fx_hist = hist_prices["USDTWD=X"].ffill()
            for _, row in portfolio.iterrows():
                p_hist = hist_prices[row["股票代號"]].ffill()
                multiplier = fx_hist if row["幣別"] == "USD" else 1.0
                equity_curve += p_hist * row["股數"] * multiplier
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name="組合淨值", line=dict(color='#00D1FF', width=3)))
            fig_hist.update_layout(height=400, template="plotly_dark", hovermode='x unified', margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        target = st.selectbox("選擇分析標的：", portfolio["股票代號"].tolist())
        period = st.select_slider("時間長度：", options=["3mo", "6mo", "1y", "2y"], value="1y")
        df_tech = yf.Ticker(target).history(period=period)
        
        if not df_tech.empty:
            # 指標計算
            df_tech['MA20'] = df_tech['Close'].rolling(window=20).mean()
            df_tech['MA50'] = df_tech['Close'].rolling(window=50).mean()
            df_tech['RSI'] = calculate_rsi(df_tech['Close'])
            df_tech['BB_U'], df_tech['BB_M'], df_tech['BB_L'] = calculate_bb(df_tech['Close'])
            df_tech['MACD'], df_tech['MACD_S'], df_tech['MACD_H'] = calculate_macd(df_tech['Close'])

            # 建立三層子圖
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, 
                               row_heights=[0.6, 0.15, 0.25],
                               subplot_titles=("K線與關鍵標記", "成交量", "MACD 指標"))

            # 1. K線圖
            fig.add_trace(go.Candlestick(x=df_tech.index, open=df_tech['Open'], high=df_tech['High'],
                                         low=df_tech['Low'], close=df_tech['Close'], name="K線"), row=1, col=1)
            # 均線與布林
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA20'], name="20MA", line=dict(color='yellow', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA50'], name="50MA", line=dict(color='orange', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_U'], name="布林上軌", line=dict(dash='dot', color='rgba(255,255,255,0.2)')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_L'], name="布林下軌", line=dict(dash='dot', color='rgba(255,255,255,0.2)')), row=1, col=1)

            # --- 自動化標記邏輯 ---
            # RSI 超賣/超買
            oversold = df_tech[df_tech['RSI'] < 30]
            overbought = df_tech[df_tech['RSI'] > 70]
            fig.add_trace(go.Scatter(x=oversold.index, y=oversold['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='RSI超跌'), row=1, col=1)
            fig.add_trace(go.Scatter(x=overbought.index, y=overbought['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='RSI超買'), row=1, col=1)

            # 2. 成交量
            vol_colors = ['red' if df_tech.Open.iloc[i] > df_tech.Close.iloc[i] else 'green' for i in range(len(df_tech))]
            fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['Volume'], name="成交量", marker_color=vol_colors), row=2, col=1)

            # 3. MACD
            m_colors = ['#FF5252' if val < 0 else '#4CAF50' for val in df_tech['MACD_H']]
            fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_H'], name="MACD柱狀", marker_color=m_colors), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], name="DIF", line=dict(color='white')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_S'], name="DEA", line=dict(color='yellow')), row=3, col=1)

            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # 健康檢查小卡
            hc1, hc2, hc3 = st.columns(3)
            last_rsi = df_tech['RSI'].iloc[-1]
            last_close = df_tech['Close'].iloc[-1]
            ma20 = df_tech['MA20'].iloc[-1]
            hc1.metric("目前 RSI", f"{last_rsi:.2f}", "超跌區" if last_rsi < 30 else "超買區" if last_rsi > 70 else "中性")
            hc2.metric("與月線(20MA)乖離", f"{((last_close/ma20)-1)*100:.2f}%")
            hc3.info("💡 綠色箭頭代表 RSI 超跌，可能為分批佈局時機。")

    with tab3:
        st.subheader("⚖️ MPT 組合優化模擬")
        if st.button("🚀 啟動模擬計算", type="primary"):
            res, err = perform_mpt_simulation(portfolio)
            if err: st.error(err)
            else: st.session_state.mpt_results = res
        if st.session_state.mpt_results:
            res = st.session_state.mpt_results
            sc1, sc2 = st.columns([2, 1])
            with sc1:
                fig_mpt = px.scatter(res['sim_df'], x='Volatility', y='Return', color='Sharpe', title="效率前緣雲圖")
                fig_mpt.add_trace(go.Scatter(x=[res['max_sharpe'][1]], y=[res['max_sharpe'][0]], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Max Sharpe'))
                st.plotly_chart(fig_mpt, use_container_width=True)
            with sc2:
                st.write("#### 建議配置比例")
                st.dataframe(res['comparison'].set_index("股票代號").style.format("{:.2f}%"))
            st.divider(); st.write("#### 資產相關性矩陣")
            st.plotly_chart(px.imshow(res['corr'], text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
else:
    st.info("尚無持股資料，請從側邊欄新增。")
