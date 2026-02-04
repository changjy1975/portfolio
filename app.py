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

if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None
if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心功能函數 (新增風險指標計算)
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
    # 基準加入 S&P 500 用於計算 Beta
    data = yf.download(symbols + ["USDTWD=X", "^GSPC"], period="1y", interval="1d")['Close']
    return data.ffill()

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# --- 風險指標函數 ---
def calculate_mdd(series):
    """計算最大回撤"""
    window = len(series)
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def calculate_beta(portfolio_returns, benchmark_returns):
    """計算 Beta 值"""
    concat_df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(concat_df) < 2: return 0.0
    matrix = np.cov(concat_df.iloc[:, 0], concat_df.iloc[:, 1])
    return matrix[0, 1] / matrix[1, 1]

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
    
    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in df_sorted.iterrows():
        r = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"
        color = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}"); r[4].write(fmt.format(row['總投入成本'])); r[5].write(fmt.format(row['現值'])); r[6].markdown(f":{color}[{fmt.format(row['獲利'])}]"); r[7].markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
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
tab1, tab2, tab3 = st.tabs(["📊 庫存配置與績效", "🧠 技術健診", "⚖️ 風險分析 (Risk)"])

if not df_record.empty:
    usd_rate = get_exchange_rate()
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(
        lambda g: pd.Series({'股數': g['股數'].sum(), '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()}), include_groups=False
    ).reset_index()

    price_map = get_latest_quotes(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)
    portfolio["獲利"] = portfolio["現值"] - (portfolio["股數"] * portfolio["平均持有單價"])
    portfolio["獲利_TWD"] = portfolio.apply(lambda r: r["獲利"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)
    portfolio["總投入成本"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]) * 100

    if current_user != "All": update_daily_snapshot(current_user, portfolio["現值_TWD"].sum(), portfolio["獲利_TWD"].sum(), usd_rate)

    with tab1:
        t_val = float(portfolio["現值_TWD"].sum()); t_prof = float(portfolio["獲利_TWD"].sum())
        roi = (t_prof / (t_val - t_prof) * 100) if (t_val - t_prof) != 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}"); c3.metric("📊 總報酬率", f"{roi:.2f}%"); c4.metric("💱 匯率", f"{usd_rate:.2f}")

        # 回測與績效分析
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
            # 成本線
            start_val = equity_curve.iloc[0]
            days = (equity_curve.index - equity_curve.index[0]).days
            cost_line = start_val * (1 + 0.0265 * (days / 365))
            fig_hist.add_trace(go.Scatter(x=equity_curve.index, y=cost_line, name="借貸成本基準 (2.65%)", line=dict(color='gray', dash='dash')))
            fig_hist.update_layout(height=400, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        # 庫存明細
        st.divider(); st.subheader("🎯 投資組合明細")
        for m, cur in [("🇹🇼 台股庫存", "TWD"), ("🇺🇸 美股庫存", "USD")]:
            m_df = portfolio[portfolio["幣別"] == cur]
            if not m_df.empty: display_market_table(m_df, m, cur, usd_rate, current_user)

    with tab2:
        # (保留原本的明亮版技術分析邏輯...)
        target = st.selectbox("選擇分析標的：", portfolio["股票代號"].tolist())
        df_tech = yf.Ticker(target).history(period="1y")
        if not df_tech.empty:
            st.write(f"正在顯示 {target} 的技術指標分析...")
            # ... 此處省略部分重複代碼以保持精簡 ...

    with tab3:
        st.subheader("⚖️ 投資組合風險深度分析")
        
        # 1. 計算風險指標
        mdd_val = calculate_mdd(equity_curve)
        portfolio_returns = equity_curve.pct_change().dropna()
        benchmark_returns = hist_prices["^GSPC"].pct_change().dropna()
        beta_val = calculate_beta(portfolio_returns, benchmark_returns)
        
        r1, r2, r3 = st.columns(3)
        r1.metric("📉 最大回撤 (MDD)", f"{mdd_val*100:.2f}%")
        r2.metric("📊 組合 Beta 值", f"{beta_val:.2f}", help="相對於 S&P 500。>1 代表波動比大盤大")
        r3.metric("📅 歷史波動率 (年化)", f"{portfolio_returns.std() * np.sqrt(252) * 100:.2f}%")
        
        st.divider()
        
        # 2. 壓力測試
        st.subheader("🚨 壓力測試：市場回檔模擬")
        st.warning("模擬情境：如果所有持股標的同步重挫 **10%**")
        
        impact_amt = t_val * 0.10
        rem_val = t_val - impact_amt
        loan_amt = 3000000 # 假設 300 萬借貸
        
        s1, s2, s3 = st.columns(3)
        s1.write("**預估資產蒸發額**")
        s1.error(f"- ${impact_amt:,.0f} TWD")
        s2.write("**回檔後總市值**")
        s2.info(f"${rem_val:,.0f} TWD")
        s3.write("**對比 300 萬借貸金**")
        cover_color = "green" if rem_val > loan_amt else "red"
        s3.markdown(f"<h3 style='color:{cover_color};'>{'安全 ✅' if rem_val > loan_amt else '危險 ⚠️'}</h3>", unsafe_allow_html=True)
        
        if rem_val < loan_amt:
            st.error(f"注意：若回檔 10%，總市值將低於借貸本金，缺口為 ${loan_amt - rem_val:,.0f} TWD。")
        else:
            st.success(f"目前安全邊際：即使回檔 10%，你仍有 ${rem_val - loan_amt:,.0f} TWD 的緩衝空間。")

        # 3. 相關性矩陣
        st.divider()
        st.subheader("🔗 資產相關性矩陣")
        corr_matrix = hist_prices[portfolio["股票代號"].tolist()].pct_change().corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', title="標的漲跌同步性分析")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("💡 相關性接近 1 代表兩者漲跌同步，接近 0 代表走勢獨立。分散投資應盡量選擇相關性較低的標的。")

else:
    st.info("尚無持股資料，請從側邊欄新增。")
