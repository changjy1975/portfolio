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
# 1. 初始化設定與狀態管理
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None
if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心功能函數 (資料與財務計算)
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

def calculate_remaining_loan(principal, annual_rate, years, months_passed):
    if principal <= 0 or annual_rate <= 0 or years <= 0: return 0.0
    r = annual_rate / 12 / 100
    n = years * 12
    if months_passed >= n: return 0.0
    remaining = principal * ((1 + r)**n - (1 + r)**months_passed) / ((1 + r)**n - 1)
    return float(remaining)

# --- 技術指標 ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2; sig = macd.ewm(span=9, adjust=False).mean()
    return macd, sig, macd - sig

def calculate_bb(series, window=20):
    ma = series.rolling(window=window).mean(); std = series.rolling(window=window).std()
    return ma + (std * 2), ma, ma - (std * 2)

# ==========================================
# 3. MPT 模擬引擎核心
# ==========================================

def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "至少需要 2 支標的。"
    try:
        data = yf.download(symbols, period="3y", interval="1d", auto_adjust=True)
        if data.empty: return None, "無法抓取歷史數據。"
        close = data['Close'] if len(symbols) > 1 else data['Close'].to_frame(name=symbols[0])
        rets = close.ffill().pct_change().dropna()
        m_rets = rets.mean() * 252; c_mat = rets.cov() * 252
        res = np.zeros((3, 2000)); w_rec = []
        for i in range(2000):
            w = np.random.random(len(symbols)); w /= np.sum(w); w_rec.append(w)
            p_r = np.sum(w * m_rets); p_s = np.sqrt(np.dot(w.T, np.dot(c_mat, w)))
            res[0,i] = p_r; res[1,i] = p_s; res[2,i] = (p_r - 0.02) / p_s
        idx = np.argmax(res[2])
        curr_w = portfolio_df["現值_TWD"].values / portfolio_df["現值_TWD"].sum()
        comp = pd.DataFrame({"股票代號": symbols, "目前權重 (%)": curr_w * 100, "建議權重 (%)": w_rec[idx] * 100})
        return {"sim_df": pd.DataFrame({'Return': res[0], 'Volatility': res[1], 'Sharpe': res[2]}), 
                "comparison": comp, "max_sharpe": (res[0, idx], res[1, idx]), "corr": rets.corr()}, None
    except Exception as e: return None, str(e)

# ==========================================
# 4. UI 元件 (表格)
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, title, currency, usd_rate, user):
    st.subheader(title)
    h_map = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本"), ("現值", "現值"), ("獲利", "獲利"), ("報酬率", "獲利率(%)")]
    h_cols = st.columns(COLS_RATIO)
    for i, (label, col_name) in enumerate(h_map):
        arr = " ▲" if st.session_state.sort_col == col_name and st.session_state.sort_asc else " ▼" if st.session_state.sort_col == col_name else ""
        if h_cols[i].button(f"{label}{arr}", key=f"h_{currency}_{col_name}_{user}"):
            if st.session_state.sort_col == col_name: st.session_state.sort_asc = not st.session_state.sort_asc
            else: st.session_state.sort_col, st.session_state.sort_asc = col_name, False
            st.rerun()
    h_cols[8].write("**管理**")
    for _, row in df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc).iterrows():
        r = st.columns(COLS_RATIO); fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"; clr = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}"); r[4].write(fmt.format(row['總投入成本'])); r[5].write(fmt.format(row['現值'])); r[6].markdown(f":{clr}[{fmt.format(row['獲利'])}]"); r[7].markdown(f":{clr}[{row['獲利率(%)']:.2f}%]")
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{user}"): save_data(load_data(user)[lambda x: x["股票代號"] != row['股票代號']], user); st.rerun()

# ==========================================
# 5. 主程式頁面
# ==========================================

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add_form", clear_on_submit=True):
            s_in = st.text_input("代號").upper().strip()
            q_in = st.number_input("股數", min_value=0.0); c_in = st.number_input("成本", min_value=0.0)
            if st.form_submit_button("執行新增"):
                if s_in: save_data(pd.concat([load_data(current_user), pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user); st.rerun()

df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)
usd_rate = get_exchange_rate()

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3, tab4 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)", "💰 資產負債表"])

if not df_record.empty:
    # 資料彙整
    df_record['幣別'] = df_record['股票代號'].apply(lambda s: "TWD" if ".TW" in s or ".TWO" in s else "USD")
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(lambda g: pd.Series({'股數': g['股數'].sum(), '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()}), include_groups=False).reset_index()
    price_map = get_latest_quotes(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    portfolio["總投入成本"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]) * 100
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)
    portfolio["獲利_TWD"] = portfolio.apply(lambda r: r["獲利"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)
    portfolio["成本_TWD"] = portfolio.apply(lambda r: r["總投入成本"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

    with tab1:
        if st.button("🔄 刷新報價"): st.cache_data.clear(); st.rerun()
        t_val, t_cost = portfolio["現值_TWD"].sum(), portfolio["成本_TWD"].sum()
        t_prof = t_val - t_cost; roi = (t_prof/t_cost*100) if t_cost != 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}"); c3.metric("📊 總報酬率", f"{roi:.2f}%"); c4.metric("💱 匯率", f"{usd_rate:.2f}")
        
        # 獲利瀑布圖
        st.divider(); st.subheader("🌊 獲利成長瀑布圖")
        tw_p = portfolio[portfolio["幣別"]=="TWD"]["獲利_TWD"].sum(); us_p = portfolio[portfolio["幣別"]=="USD"]["獲利_TWD"].sum()
        st.plotly_chart(go.Figure(go.Waterfall(orientation="v", measure=["relative","relative","relative","total"], x=["投入成本","台股獲利","美股獲利","最終現值"], y=[t_cost, tw_p, us_p, t_val], connector={"line":{"color":"gray"}}, decreasing={"marker":{"color":"#e74c3c"}}, increasing={"marker":{"color":"#2ecc71"}}, totals={"marker":{"color":"#3498db"}})).update_layout(height=450), use_container_width=True)
        
        st.divider(); pc1, pc2 = st.columns(2)
        with pc1: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="市場佔比", hole=0.4), use_container_width=True)
        with pc2: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="股票代號", title="持股配置", hole=0.4), use_container_width=True)
        
        st.divider(); tw_df = portfolio[portfolio["幣別"]=="TWD"]; us_df = portfolio[portfolio["幣別"]=="USD"]
        if not tw_df.empty: display_market_table(tw_df, "🇹🇼 台股清單", "TWD", usd_rate, current_user)
        if not us_df.empty: display_market_table(us_df, "🇺🇸 美股清單", "USD", usd_rate, current_user)

    with tab2:
        target = st.selectbox("分析標的", portfolio["股票代號"].tolist())
        df_t = yf.Ticker(target).history(period="1y")
        if not df_t.empty:
            df_t['RSI'], (df_t['BU'], df_t['BM'], df_t['BL']), (df_t['M'], df_t['MS'], df_t['MH']) = calculate_rsi(df_t['Close']), calculate_bb(df_t['Close']), calculate_macd(df_t['Close'])
            curr = df_t.iloc[-1]; score = 0; reasons = []
            if curr['RSI'] < 30: score += 1; reasons.append("RSI 超跌")
            elif curr['RSI'] > 70: score -= 1; reasons.append("RSI 超漲")
            if curr['Close'] < curr['BL']: score += 1; reasons.append("觸及布林下軌")
            if curr['M'] > curr['MS']: score += 1; reasons.append("MACD 黃金交叉")
            advice = "強力買入 🚀" if score >= 2 else "分批佈局 📈" if score == 1 else "持股觀望 ⚖️" if score == 0 else "分批獲利 💰"
            st.subheader(f"🔍 {target} 技術診斷：{advice}"); st.info("依據：" + "/".join(reasons))
            f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3]); f.add_trace(go.Scatter(x=df_t.index, y=df_t['Close'], name="價"),1,1); f.add_trace(go.Scatter(x=df_t.index, y=df_t['BU'], name="上軌", line=dict(dash='dot')),1,1); f.add_trace(go.Bar(x=df_t.index, y=df_t['MH'], name="MACD柱"),2,1); st.plotly_chart(f, use_container_width=True)

    with tab3:
        st.subheader("⚖️ MPT 組合優化模擬")
        if st.button("🚀 執行模擬", type="primary"):
            with st.spinner("模擬中..."):
                res, err = perform_mpt_simulation(portfolio)
                if err: st.error(err)
                else: st.session_state.mpt_results = res
        if st.session_state.mpt_results:
            res = st.session_state.mpt_results; sc1, sc2 = st.columns([2, 1])
            with sc1: st.plotly_chart(px.scatter(res['sim_df'], x='Volatility', y='Return', color='Sharpe', title="效率前緣"), use_container_width=True)
            with sc2: st.write("#### 建議配置"); st.dataframe(res['comparison'].set_index("股票代號").style.format("{:.2f}%"))
            st.divider(); st.write("#### 相關性矩陣"); st.plotly_chart(px.imshow(res['corr'], text_auto=".2f"), use_container_width=True)

    with tab4:
        st.subheader("💰 家庭資產負債表")
        c_r = st.number_input("💵 現金預留", value=500000.0)
        st.divider(); lc1, lc2 = st.columns(2)
        with lc1:
            st.write("**貸款 1 (房貸)**"); l1p = st.number_input("本金 1", value=3000000.0); l1r = st.number_input("利率 1", value=2.65); l1y = st.number_input("年限 1", value=30); l1m = st.number_input("已還月 1", value=12)
        with lc2:
            st.write("**貸款 2 (信貸)**"); l2p = st.number_input("本金 2", value=0.0); l2r = st.number_input("利率 2", value=3.5); l2y = st.number_input("年限 2", value=7); l2m = st.number_input("已還月 2", value=0)
        st.divider(); st.write("**股票質押監控**"); pc1, pc2 = st.columns(2)
        with pc1: pl = st.number_input("質押借款金額", value=0.0)
        with pc2: pt = st.multiselect("擔保標的", portfolio["股票代號"].tolist())
        
        rem1, rem2 = calculate_remaining_loan(l1p, l1r, l1y, l1m), calculate_remaining_loan(l2p, l2r, l2y, l2m)
        t_debt = rem1 + rem2 + pl; n_w = (t_val + c_r) - t_debt
        st.divider(); mc1, mc2, mc3 = st.columns(3)
        mc1.metric("💼 家庭總資產", f"${(t_val+c_r):,.0f}"); mc2.metric("📉 剩餘總負債", f"-${t_debt:,.0f}"); mc3.metric("🏆 家庭淨資產", f"${n_w:,.0f}")
        
        if pl > 0 and pt:
            m_r = (portfolio[portfolio["股票代號"].isin(pt)]["現值_TWD"].sum() / pl * 100)
            st.warning(f"🚨 即時質押維持率：**{m_r:.2f}%** (門檻 130%)")
            if len(pt)==1:
                st.error(f"🚩 {pt[0]} 斷頭預警價：**${(1.3 * pl / portfolio[portfolio['股票代號']==pt[0]]['股數'].values[0]):.2f}**")
else: st.info("尚未發現持股。")
