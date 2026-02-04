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
# 1. 初始化設定與路徑
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

# 初始化 session_state 用於儲存 MPT 結果與排序
if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None
if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心功能函數 (資料處理與行情)
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
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2; signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def calculate_bb(series, window=20):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return ma + (std * 2), ma, ma - (std * 2)

# ==========================================
# 3. MPT 數學模擬引擎
# ==========================================

def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "至少需要 2 支標的才能模擬。"
    try:
        data = yf.download(symbols, period="3y", interval="1d", auto_adjust=True)
        if data.empty: return None, "無法獲取歷史數據。"
        
        close_prices = data['Close'] if len(symbols) > 1 else data['Close'].to_frame(name=symbols[0])
        returns = close_prices.ffill().pct_change().dropna()
        if returns.empty: return None, "有效數據不足。"
        
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
            results[2,i] = (p_ret - 0.02) / p_std # Rf=2%
            
        max_idx = np.argmax(results[2]); min_idx = np.argmin(results[1])
        curr_val = portfolio_df["現值_TWD"].values
        curr_w = curr_val / np.sum(curr_val)
        
        comparison = pd.DataFrame({
            "股票代號": symbols,
            "目前權重 (%)": curr_w * 100,
            "Max Sharpe 建議 (%)": weights_record[max_idx] * 100,
            "Min Vol 建議 (%)": weights_record[min_idx] * 100
        })
        
        return {
            "sim_df": pd.DataFrame({'Return': results[0], 'Volatility': results[1], 'Sharpe': results[2]}),
            "comparison": comparison,
            "max_sharpe": (results[0, max_idx], results[1, max_idx]),
            "corr": returns.corr()
        }, None
    except Exception as e: return None, str(e)

# ==========================================
# 4. 介面顯示組件 (具備排序功能)
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
            full = load_data(current_user)
            save_data(full[full["股票代號"] != row['股票代號']], current_user); st.rerun()

    s_cost, s_val, s_profit = df["總投入成本"].sum(), df["現值"].sum(), df["獲利"].sum()
    s_roi = (s_profit / s_cost * 100) if s_cost != 0 else 0
    st.markdown("---")
    sc = st.columns(COLS_RATIO)
    sc[0].markdown(f"**{currency} 小計**"); sc[4].markdown(f"**{fmt.format(s_cost)}**"); sc[5].markdown(f"**{fmt.format(s_val)}**"); sc[6].markdown(f":{'red' if s_profit>0 else 'green'}[**{fmt.format(s_profit)}**]"); sc[7].markdown(f":{'red' if s_profit>0 else 'green'}[**{s_roi:.2f}%**]")
    if currency == "USD":
        sc2 = st.columns(COLS_RATIO); sc2[0].caption("*(換算台幣)*"); sc2[4].caption(f"${(s_cost*usd_rate):,.0f}"); sc2[5].caption(f"${(s_val*usd_rate):,.0f}"); sc2[6].caption(f"${(s_profit*usd_rate):,.0f}")

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
        if tc1.button("🔄 刷新報價"): st.cache_data.clear(); st.rerun()
        t_val = float(portfolio["現值_TWD"].sum()); t_prof = float(portfolio["獲利_TWD"].sum())
        roi = (t_prof / (t_val - t_prof) * 100) if (t_val - t_prof) != 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}"); c3.metric("📊 總報酬率", f"{roi:.2f}%"); c4.metric("💱 匯率", f"{usd_rate:.2f}")

        # B. 圓餅圖配置 (置頂)
        st.divider(); st.subheader("🎯 投資組合配置分析")
        pc1, pc2 = st.columns(2)
        with pc1: 
            st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="市場配置 (TWD)", hole=0.45), use_container_width=True)
        with pc2:
            view_mode = st.selectbox("選擇個股配置範圍：", ["全部", "台股", "美股"], key="pie_filter")
            if view_mode == "台股":
                chart_df = portfolio[portfolio["幣別"] == "TWD"]
                chart_title = "個股配置 (台股)"
            elif view_mode == "美股":
                chart_df = portfolio[portfolio["幣別"] == "USD"]
                chart_title = "個股配置 (美股)"
            else:
                chart_df = portfolio
                chart_title = "個股配置 (全部)"

            if not chart_df.empty:
                st.plotly_chart(px.pie(chart_df, values="現值_TWD", names="股票代號", title=f"{chart_title} (TWD)", hole=0.45), use_container_width=True)
            else:
                st.info(f"目前沒有 {view_mode} 的持股資料。")

        # C. 庫存列表 (置底)
        st.divider()
        tw_df = portfolio[portfolio["幣別"] == "TWD"]
        if not tw_df.empty: display_market_table(tw_df, "🇹🇼 台股庫存明細", "TWD", usd_rate, current_user)
        us_df = portfolio[portfolio["幣別"] == "USD"]
        if not us_df.empty: display_market_table(us_df, "🇺🇸 美股庫存明細", "USD", usd_rate, current_user)

    with tab2:
        target = st.selectbox("選擇分析標的：", portfolio["股票代號"].tolist())
        df_tech = yf.Ticker(target).history(period="1y")
        if not df_tech.empty:
            df_tech['RSI'] = calculate_rsi(df_tech['Close'])
            df_tech['BB_U'], df_tech['BB_M'], df_tech['BB_L'] = calculate_bb(df_tech['Close'])
            df_tech['MACD'], df_tech['MACD_S'], df_tech['MACD_H'] = calculate_macd(df_tech['Close'])
            curr = df_tech.iloc[-1]
            
            # 診斷建議
            score = 0; reasons = []
            if curr['RSI'] < 35: score += 1; reasons.append("RSI 超跌")
            elif curr['RSI'] > 65: score -= 1; reasons.append("RSI 超漲")
            if curr['MACD'] > curr['MACD_S']: score += 1; reasons.append("MACD 黃金交叉")
            else: score -= 1; reasons.append("MACD 死亡交叉")
            
            advice = "強力買入 🚀" if score >= 2 else "分批佈局 📈" if score == 1 else "觀望整理 ⚖️" if score == 0 else "分批獲利 💰" if score == -1 else "強勢賣出 📉"
            st.subheader(f"🔍 {target} 綜合診斷報告：**{advice}**")
            st.info("💡 分析依據：" + "、".join(reasons))

            # 繪製技術圖表
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['Close'], name="收盤價"), row=1, col=1)
            # 提高布林通道透明度並使用更亮顯的顏色
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_U'], name="布林上軌", line=dict(dash='dot', color='rgba(255, 82, 82, 0.8)')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_L'], name="布林下軌", line=dict(dash='dot', color='rgba(76, 175, 80, 0.8)')), row=1, col=1)
            
            # MACD 能量柱：亮色處理，並區分紅綠
            macd_colors = ['#FF5252' if val < 0 else '#4CAF50' for val in df_tech['MACD_H']]
            fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_H'], name="MACD 能量柱", marker_color=macd_colors), row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

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
                fig = px.scatter(res['sim_df'], x='Volatility', y='Return', color='Sharpe', title="效率前緣雲圖", labels={'Volatility':'年化波動','Return':'年化回報'})
                fig.add_trace(go.Scatter(x=[res['max_sharpe'][1]], y=[res['max_sharpe'][0]], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Max Sharpe'))
                st.plotly_chart(fig, use_container_width=True)
            with sc2:
                st.write("#### 建議配置比例")
                st.dataframe(res['comparison'].set_index("股票代號").style.format("{:.2f}%"))
            st.divider()
            st.write("#### 資產相關性矩陣")
            st.plotly_chart(px.imshow(res['corr'], text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
else:
    st.info("尚無持股資料，請從側邊欄新增。")
