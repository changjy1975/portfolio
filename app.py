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

def calculate_remaining_loan(principal, annual_rate, years, months_passed):
    if principal <= 0 or annual_rate <= 0 or years <= 0: return 0.0
    r = annual_rate / 12 / 100
    n = years * 12
    if months_passed >= n: return 0.0
    remaining = principal * ((1 + r)**n - (1 + r)**months_passed) / ((1 + r)**n - 1)
    return float(remaining)

# --- 技術指標計算 ---
def calculate_rsi(series, period=14):
    delta = series.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))

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
# 4. 主程式
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

# --- 全域資料預處理 ---
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
    portfolio["獲利_TWD"] = portfolio.apply(lambda r: r["獲利"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)
    portfolio["成本_TWD"] = portfolio.apply(lambda r: r["總投入成本"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

with tab1:
    if df_record.empty: st.info("尚無數據。")
    else:
        if st.button("🔄 刷新報價"): st.cache_data.clear(); st.rerun()
        t_val = float(portfolio["現值_TWD"].sum())
        t_cost = float(portfolio["成本_TWD"].sum())
        t_prof = t_val - t_cost
        roi = (t_prof / t_cost * 100) if t_cost != 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}"); c3.metric("📊 總報酬率", f"{roi:.2f}%"); c4.metric("💱 匯率", f"{usd_rate:.2f}")
        
        # --- 獲利瀑布圖 ---
        st.divider(); st.subheader("🌊 獲利成長瀑布圖")
        tw_prof = portfolio[portfolio["幣別"] == "TWD"]["獲利_TWD"].sum()
        us_prof = portfolio[portfolio["幣別"] == "USD"]["獲利_TWD"].sum()
        
        fig_wf = go.Figure(go.Waterfall(
            orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["總投入成本 (TWD)", "台股總獲利", "美股總獲利", "目前總現值"],
            textposition = "outside",
            text = [f"${t_cost:,.0f}", f"${tw_prof:,.0f}", f"${us_prof:,.0f}", f"${t_val:,.0f}"],
            y = [t_cost, tw_prof, us_prof, t_val],
            connector = {"line":{"color":"gray"}},
            decreasing = {"marker":{"color":"#e74c3c"}},
            increasing = {"marker":{"color":"#2ecc71"}},
            totals = {"marker":{"color":"#3498db"}}
        ))
        fig_wf.update_layout(title="獲利組成拆解 (TWD)", showlegend=False, height=500)
        st.plotly_chart(fig_wf, use_container_width=True)

        # 圓餅圖
        st.divider(); pc1, pc2 = st.columns(2)
        with pc1: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="市場配置", hole=0.4), use_container_width=True)
        with pc2: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="股票代號", title="個股配置", hole=0.4), use_container_width=True)
        
        # 庫存表格
        st.divider(); tw_df = portfolio[portfolio["幣別"] == "TWD"]; us_df = portfolio[portfolio["幣別"] == "USD"]
        if not tw_df.empty: display_market_table(tw_df, "🇹🇼 台股庫存", "TWD", usd_rate, current_user)
        if not us_df.empty: display_market_table(us_df, "🇺🇸 美股庫存", "USD", usd_rate, current_user)

with tab2:
    if portfolio.empty: st.info("尚無數據。")
    else:
        target = st.selectbox("分析標的", portfolio["股票代號"].tolist())
        df_t = yf.Ticker(target).history(period="1y")
        if not df_t.empty:
            df_t['RSI'] = calculate_rsi(df_t['Close'])
            df_t['BU'], df_t['BM'], df_t['BL'] = calculate_bb(df_t['Close'])
            df_t['MACD'], df_t['MS'], df_t['MH'] = calculate_macd(df_t['Close'])
            curr = df_t.iloc[-1]
            
            # --- 技術指標建議邏輯 ---
            score = 0; reasons = []
            if curr['RSI'] < 30: score += 1; reasons.append("RSI 處於超跌區 ( <30 )")
            elif curr['RSI'] > 70: score -= 1; reasons.append("RSI 處於超漲區 ( >70 )")
            
            if curr['Close'] < curr['BL']: score += 1; reasons.append("股價觸及布林下軌 (支撐位)")
            elif curr['Close'] > curr['BU']: score -= 1; reasons.append("股價觸及布林上軌 (壓力位)")
            
            if curr['MACD'] > curr['MS']: score += 1; reasons.append("MACD 呈多頭趨勢 (黃金交叉)")
            else: score -= 1; reasons.append("MACD 呈空頭趨勢 (死亡交叉)")
            
            advice = "強力買入 🚀" if score >= 2 else "分批佈局 📈" if score == 1 else "持股觀望 ⚖️" if score == 0 else "分批獲利 💰" if score == -1 else "建議出場 📉"
            advice_color = "red" if score >= 1 else "green" if score <= -1 else "gray"

            st.subheader(f"🔍 {target} 技術診斷報告")
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("最新 RSI", f"{curr['RSI']:.1f}")
            tc2.metric("MACD 狀態", "多頭" if curr['MACD'] > curr['MS'] else "空頭")
            tc3.metric("布林位置", "下軌支撐" if curr['Close'] < curr['BM'] else "上軌壓力")
            
            st.markdown(f"#### 💡 綜合投資建議：**:{advice_color}[{advice}]**")
            st.info("分析依據：\n* " + "\n* ".join(reasons))

            f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            f.add_trace(go.Scatter(x=df_t.index, y=df_t['Close'], name="價格"),1,1)
            f.add_trace(go.Scatter(x=df_t.index, y=df_t['BU'], name="BB上軌", line=dict(dash='dot', color='rgba(255,0,0,0.3)')),1,1)
            f.add_trace(go.Scatter(x=df_t.index, y=df_t['BL'], name="BB下軌", line=dict(dash='dot', color='rgba(0,255,0,0.3)')),1,1)
            f.add_trace(go.Bar(x=df_t.index, y=df_t['MH'], name="MACD柱"),2,1)
            f.update_layout(height=600, template="plotly_dark", showlegend=False); st.plotly_chart(f, use_container_width=True)

with tab4:
    st.subheader("💰 家庭資產負債表 (淨資產監控)")
    
    # 1. 資產端
    st.markdown("#### 1. 資產端")
    ic1, ic2 = st.columns(2)
    with ic1: cash_res = st.number_input("💵 現金預留 (TWD)", min_value=0.0, value=500000.0, step=10000.0)
    with ic2:
        if not portfolio.empty:
            st.info(f"股票現值 (自動導入): ${portfolio['現值_TWD'].sum():,.0f}")
    
    # 2. 負債端 (雙筆貸款)
    st.divider(); st.markdown("#### 2. 負債端：貸款設定")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.write("**第一筆貸款 (如房貸)**")
        l1_p = st.number_input("🏦 原始本金 (L1)", value=3000000.0)
        l1_r = st.number_input("📈 年利率 (%) (L1)", value=2.65)
        l1_y = st.number_input("⏳ 期限 (年) (L1)", value=30); l1_m = st.number_input("📅 已還月數 (L1)", value=12)
    with lc2:
        st.write("**第二筆貸款 (如信貸)**")
        l2_p = st.number_input("🏦 原始本金 (L2)", value=0.0)
        l2_r = st.number_input("📈 年利率 (%) (L2)", value=3.5)
        l2_y = st.number_input("⏳ 期限 (年) (L2)", value=7); l2_m = st.number_input("📅 已還月數 (L2)", value=0)

    # 3. 股票質押
    st.divider(); st.markdown("#### 3. 槓桿監控：股票質押")
    gc1, gc2 = st.columns(2)
    with gc1: pledge_loan = st.number_input("💸 質押借款總額 (TWD)", min_value=0.0, value=0.0)
    with gc2: pledge_targets = st.multiselect("🎯 選擇擔保標的", portfolio["股票代號"].tolist()) if not portfolio.empty else []

    # 計算剩餘與維持率
    rem_l1 = calculate_remaining_loan(l1_p, l1_r, l1_y, l1_m)
    rem_l2 = calculate_remaining_loan(l2_p, l2_r, l2_y, l2_m)
    total_assets = float(portfolio["現值_TWD"].sum()) + cash_res if not portfolio.empty else cash_res
    total_debts = rem_l1 + rem_l2 + pledge_loan
    net_worth = total_assets - total_debts

    # 顯示維持率
    if pledge_loan > 0 and pledge_targets:
        collateral_val = portfolio[portfolio["股票代號"].isin(pledge_targets)]["現值_TWD"].sum()
        m_ratio = (collateral_val / pledge_loan * 100)
        m_color = "normal" if m_ratio > 150 else "off" if m_ratio > 140 else "inverse"
        st.metric("🚨 即時質押維持率", f"{m_ratio:.2f}%", delta="門檻 130%", delta_color=m_color)
        if len(pledge_targets) == 1:
            liq_p = (1.3 * pledge_loan) / portfolio[portfolio["股票代號"] == pledge_targets[0]]["股數"].values[0]
            st.error(f"🚩 斷頭預警價：當 {pledge_targets[0]} 跌破 **${liq_p:.2f}** 時維持率將低於 130%。")

    # 4. 財務摘要
    st.divider(); mc1, mc2, mc3 = st.columns(3)
    mc1.metric("💼 家庭總資產", f"${total_assets:,.0f}")
    mc2.metric("📉 剩餘總負債", f"-${total_debts:,.0f}", delta=f"L1:${rem_l1:,.0f} | L2:${rem_l2:,.0f}", delta_color="inverse")
    mc3.metric("🏆 家庭淨資產 (Net Worth)", f"${net_worth:,.0f}")

    st.write("#### 📊 資產負債結構分析")
    bal_df = pd.DataFrame({
        "項目": ["股票現值", "現金預留", "貸款 1 餘額", "貸款 2 餘額", "質押借款"],
        "金額": [total_assets - cash_res, cash_res, -rem_l1, -rem_l2, -pledge_loan],
        "類別": ["資產", "資產", "負債", "負債", "負債"]
    })
    st.plotly_chart(px.bar(bal_df, x="項目", y="金額", color="類別", color_discrete_map={"資產":"#2ecc71","負債":"#e74c3c"}), use_container_width=True)
