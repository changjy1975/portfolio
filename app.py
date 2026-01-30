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
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

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

def calculate_remaining_principal(principal, annual_rate, years, months_passed):
    """計算房貸剩餘本金公式"""
    if principal <= 0 or annual_rate <= 0 or years <= 0: return 0.0
    r = annual_rate / 12 / 100
    n = years * 12
    if months_passed >= n: return 0.0
    remaining = principal * ((1 + r)**n - (1 + r)**months_passed) / ((1 + r)**n - 1)
    return float(remaining)

# --- 技術指標 ---
def calculate_rsi(series, period=14):
    delta = series.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean(); exp2 = series.ewm(span=26, adjust=False).mean()
    m = exp1 - exp2; s = m.ewm(span=9, adjust=False).mean()
    return m, s, m - s

def calculate_bb(series, window=20):
    ma = series.rolling(window=window).mean(); std = series.rolling(window=window).std()
    return ma + (std * 2), ma, ma - (std * 2)

# ==========================================
# 3. 介面表格組件
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, title, currency, usd_rate, current_user):
    st.subheader(title)
    h_map = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本"), ("現值", "現值"), ("獲利", "獲利"), ("報酬率", "獲利率(%)")]
    h_cols = st.columns(COLS_RATIO)
    for i, (l, c_n) in enumerate(h_map):
        arr = " ▲" if st.session_state.sort_col == c_n and st.session_state.sort_asc else " ▼" if st.session_state.sort_col == c_n else ""
        if h_cols[i].button(f"{l}{arr}", key=f"h_{currency}_{c_n}_{current_user}"):
            if st.session_state.sort_col == c_n: st.session_state.sort_asc = not st.session_state.sort_asc
            else: st.session_state.sort_col, st.session_state.sort_asc = c_n, False
            st.rerun()
    h_cols[8].write("**管理**")
    for _, row in df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc).iterrows():
        r = st.columns(COLS_RATIO); f = "{:,.0f}" if currency == "TWD" else "{:,.2f}"; clr = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}"); r[4].write(f.format(row['總投入成本'])); r[5].write(f.format(row['現值'])); r[6].markdown(f":{clr}[{f.format(row['獲利'])}]"); r[7].markdown(f":{clr}[{row['獲利率(%)']:.2f}%]")
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{current_user}"): save_data(load_data(current_user)[lambda x: x["股票代號"] != row['股票代號']], current_user); st.rerun()

# ==========================================
# 4. 主程式邏輯
# ==========================================

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add_form", clear_on_submit=True):
            s_in = st.text_input("代號").upper().strip()
            q_in = st.number_input("股數", min_value=0.0); c_in = st.number_input("成本", min_value=0.0)
            if st.form_submit_button("新增持股"):
                if s_in: save_data(pd.concat([load_data(current_user), pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user); st.rerun()

df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3, tab4 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)", "💰 資產負債表"])

# --- 資料預處理 ---
usd_rate = get_exchange_rate()
portfolio = pd.DataFrame()
if not df_record.empty:
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(lambda g: pd.Series({'股數': g['股數'].sum(), '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()}), include_groups=False).reset_index()
    price_map = get_latest_quotes(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    portfolio["總投入成本"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]) * 100
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

with tab1:
    if df_record.empty: st.info("尚無持股數據。")
    else:
        if st.button("🔄 刷新最新報價"): st.cache_data.clear(); st.rerun()
        t_val = float(portfolio["現值_TWD"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c4.metric("💱 匯率", f"{usd_rate:.2f}")
        st.divider(); pc1, pc2 = st.columns(2)
        with pc1: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="市場分配", hole=0.4), use_container_width=True)
        with pc2: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="股票代號", title="個股配置", hole=0.4), use_container_width=True)
        st.divider(); tw_df = portfolio[portfolio["幣別"] == "TWD"]; us_df = portfolio[portfolio["幣別"] == "USD"]
        if not tw_df.empty: display_market_table(tw_df, "🇹🇼 台股庫存", "TWD", usd_rate, current_user)
        if not us_df.empty: display_market_table(us_df, "🇺🇸 美股庫存", "USD", usd_rate, current_user)

with tab2:
    if portfolio.empty: st.info("尚無數據。")
    else:
        target = st.selectbox("分析標的", portfolio["股票代號"].tolist())
        df_t = yf.Ticker(target).history(period="1y")
        if not df_t.empty:
            df_t['RSI'], (df_t['BU'], df_t['BM'], df_t['BL']), (df_t['M'], df_t['MS'], df_t['MH']) = calculate_rsi(df_t['Close']), calculate_bb(df_t['Close']), calculate_macd(df_t['Close'])
            f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            f.add_trace(go.Scatter(x=df_t.index, y=df_t['Close'], name="價格"),1,1); f.add_trace(go.Scatter(x=df_t.index, y=df_t['BU'], name="布林上軌", line=dict(dash='dot')),1,1)
            f.add_trace(go.Bar(x=df_t.index, y=df_t['MH'], name="MACD柱"),2,1); f.update_layout(height=500, template="plotly_dark"); st.plotly_chart(f, use_container_width=True)

with tab4:
    st.subheader("💰 家庭資產負債表 (淨資產監控)")
    
    # --- 1. 現金與概覽 ---
    st.markdown("#### 1. 資產端")
    ic1, ic2 = st.columns(2)
    with ic1:
        cash_res = st.number_input("💵 現金預留 (TWD)", min_value=0.0, value=500000.0, step=10000.0)
    with ic2:
        # 修正錯誤：使用正確的 if-else 結構
        if not portfolio.empty:
            st.caption(f"股票現值 (自動導入): ${portfolio['現值_TWD'].sum():,.0f}")
        else:
            st.write("請先在庫存分頁新增持股")

    st.divider()
    
    # --- 2. 負債端：房貸／信貸 ---
    st.markdown("#### 2. 負債端：房貸與一般借貸")
    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        l_p = st.number_input("🏦 貸款原始本金", value=3000000.0) # 預設 300 萬
    with dc2:
        l_r = st.number_input("📈 貸款年利率 (%)", value=2.65) # 預設 2.65%
    with dc3:
        l_y = st.number_input("⏳ 貸款期限 (年)", value=30)
    with dc4:
        m_p = st.number_input("📅 已還款月數", value=12)

    # --- 3. 槓桿端：股票質押監控 ---
    st.divider()
    st.markdown("#### 3. 槓桿監控：股票質押 (Stock Pledging)")
    lc1, lc2, lc3 = st.columns([1.5, 2, 1])
    with lc1:
        pledge_loan = st.number_input("💸 質押借款總額 (TWD)", min_value=0.0, value=0.0, step=10000.0)
    with lc2:
        # 選取擔保品標的
        pledge_target = st.multiselect("🎯 選擇質押擔保標的", portfolio["股票代號"].tolist()) if not portfolio.empty else []
    with lc3:
        st.info("💡 質押維持率門檻：130%")

    # --- 4. 財務結算報告 ---
    st.divider()
    st.markdown("#### 4. 家庭財務診斷報告")
    
    # 計算房貸剩餘
    rem_mortgage = calculate_remaining_principal(l_p, l_r, l_y, m_p)
    # 總負債 = 房貸剩餘 + 質押借款
    total_debt = rem_mortgage + pledge_loan
    # 總資產 = 股票現值 + 現金
    stock_value_twd = float(portfolio["現值_TWD"].sum()) if not portfolio.empty else 0.0
    total_assets = stock_value_twd + cash_res
    # 淨資產
    net_worth = total_assets - total_debt
    
    # 質押維持率計算
    collateral_val = portfolio[portfolio["股票代號"].isin(pledge_target)]["現值_TWD"].sum()
    m_ratio = (collateral_val / pledge_loan * 100) if pledge_loan > 0 else 0
    
    # 呈現看板
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("💼 家庭總資產", f"${total_assets:,.0f}")
    mc2.metric("📉 剩餘總負債", f"-${total_debt:,.0f}", delta=f"含質押:${pledge_loan:,.0f}")
    mc3.metric("🏆 家庭淨資產", f"${net_worth:,.0f}")
    
    # 質押警示指標
    if pledge_loan > 0:
        # 維持率顏色邏輯
        m_color = "normal" if m_ratio > 160 else "off" if m_ratio > 140 else "inverse"
        mc4.metric("🚨 質押維持率", f"{m_ratio:.1f}%", delta="門檻 130%", delta_color=m_color)
        
        # 斷頭價格試算
        if len(pledge_target) == 1:
            target_stock = pledge_target[0]
            target_shares = portfolio[portfolio["股票代號"] == target_stock]["股數"].values[0]
            # 斷頭價公式：維持率 130% 時的股價
            liq_price = (1.3 * pledge_loan) / target_shares
            st.error(f"🚩 **{target_stock} 斷頭警示價預估**： 當股價跌破 **${liq_price:.2f}** 時，維持率將低於 130%。")
        elif len(pledge_target) > 1:
            st.warning("⚠️ 多標的質押暫不支持精確斷頭價試算，請參考整體維持率。")
    else:
        mc4.metric("🚨 質押維持率", "N/A")

    # --- 5. 保險與對沖分析 ---
    st.divider()
    st.write("#### 🛡️ 風險防護：遞減型房貸壽險對沖")
    st.write("您已投保遞減型房貸壽險。")
    
    # 視覺化房貸對沖圖
    st.success(f"目前剩餘房貸：**${rem_mortgage:,.0f}**。")
    st.info(f"💡 您的壽險保額應隨此金額逐月遞減，目前風險覆蓋金額需大於 **${rem_mortgage:,.0f}**。")
    
    # 淨資產組成圓餅圖
    st.write("#### 📊 資產負債結構")
    bal_df = pd.DataFrame({
        "項目": ["股票現值", "現金預留", "剩餘房貸", "質押借款"],
        "金額": [stock_value_twd, cash_res, -rem_mortgage, -pledge_loan],
        "類別": ["資產", "資產", "負債", "負債"]
    })
    st.plotly_chart(px.bar(bal_df, x="項目", y="金額", color="類別", 
                           color_discrete_map={"資產": "#2ecc71", "負債": "#e74c3c"},
                           title="家庭資產負債結構對比圖"), use_container_width=True)
