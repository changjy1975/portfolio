import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import shutil
import json
from datetime import datetime
import pytz
import numpy as np

# ==========================================
# 1. 初始化設定與狀態管理
# ==========================================
st.set_page_config(page_title="投資戰情室", layout="wide")

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

def load_financial_config(user):
    path = f"financial_config_{user}.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except: pass
    return {
        "cash_res": 500000.0,
        "l1_p": 3000000.0, "l1_r": 2.65, "l1_y": 30, "l1_m": 12,
        "l2_p": 0.0, "l2_r": 3.5, "l2_y": 7, "l2_m": 0,
        "pledge_loan": 0.0, "pledge_targets": []
    }

def save_financial_config(user, config):
    path = f"financial_config_{user}.json"
    with open(path, "w") as f: json.dump(config, f)

def calculate_remaining_loan(principal, annual_rate, years, months_passed):
    if principal <= 0 or annual_rate <= 0 or years <= 0: return 0.0
    r = annual_rate / 12 / 100
    n = years * 12
    if months_passed >= n: return 0.0
    remaining = principal * ((1 + r)**n - (1 + r)**months_passed) / ((1 + r)**n - 1)
    return float(remaining)

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

# ==========================================
# 3. 介面表格組件
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, title, currency, usd_rate, user):
    st.subheader(title)
    h_map = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本"), ("現值", "現值"), ("獲利", "獲利"), ("報酬率", "獲利率(%)")]
    h_cols = st.columns(COLS_RATIO)
    for i, (label, col_name) in enumerate(h_map):
        arrow = " ▲" if st.session_state.sort_col == col_name and st.session_state.sort_asc else " ▼" if st.session_state.sort_col == col_name else ""
        if h_cols[i].button(f"{label}{arrow}", key=f"h_{currency}_{col_name}_{user}"):
            if st.session_state.sort_col == col_name: st.session_state.sort_asc = not st.session_state.sort_asc
            else: st.session_state.sort_col, st.session_state.sort_asc = col_name, False
            st.rerun()
    h_cols[8].write("**管理**")
    for _, row in df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc).iterrows():
        r = st.columns(COLS_RATIO); fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"; clr = "red" if row["獲利"] > 0 else "green"
        r[0].write(f"**{row['股票代號']}**"); r[1].write(f"{row['股數']:.2f}"); r[2].write(f"{row['平均持有單價']:.2f}"); r[3].write(f"{row['最新股價']:.2f}"); r[4].write(fmt.format(row['總投入成本'])); r[5].write(fmt.format(row['現值'])); r[6].markdown(f":{clr}[{fmt.format(row['獲利'])}]"); r[7].markdown(f":{clr}[{row['獲利率(%)']:.2f}%]")
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{user}"): save_data(load_data(user)[lambda x: x["股票代號"] != row['股票代號']], user); st.rerun()

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
            if st.form_submit_button("執行新增"):
                if s_in: save_data(pd.concat([load_data(current_user), pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user); st.rerun()

df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)
usd_rate = get_exchange_rate()

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3, tab4 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)", "💰 資產負債表"])

if not df_record.empty:
    df_record['幣別'] = df_record['股票代號'].apply(lambda s: "TWD" if ".TW" in s or ".TWO" in s else "USD")
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(lambda g: pd.Series({'股數': g['股數'].sum(), '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()}), include_groups=False).reset_index()
    price_map = get_latest_quotes(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    portfolio["總投入成本"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)

    with tab1:
        if st.button("🔄 刷新報價"): st.cache_data.clear(); st.rerun()
        t_val = float(portfolio["現值_TWD"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}"); c4.metric("💱 匯率", f"{usd_rate:.2f}")
        st.divider(); pc1, pc2 = st.columns(2)
        with pc1: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="市場配置", hole=0.4), use_container_width=True)
        with pc2: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="股票代號", title="個股配置", hole=0.4), use_container_width=True)
        st.divider(); tw_df = portfolio[portfolio["幣別"]=="TWD"]; us_df = portfolio[portfolio["幣別"]=="USD"]
        if not tw_df.empty: display_market_table(tw_df, "🇹🇼 台股庫存", "TWD", usd_rate, current_user)
        if not us_df.empty: display_market_table(us_df, "🇺🇸 美股庫存", "USD", usd_rate, current_user)

    # ... Tab 2 & 3 邏輯 (略) ...

    with tab4:
        f_cfg = load_financial_config(current_user if current_user != "All" else "Alan")
        st.subheader("💰 家庭資產負債表管理")
        
        with st.form("financial_form"):
            st.markdown("#### 1. 資金與貸款設定")
            c_r = st.number_input("💵 現金預留 (TWD)", value=f_cfg["cash_res"])
            st.divider(); lc1, lc2 = st.columns(2)
            with lc1:
                st.write("**貸款 1**"); l1p = st.number_input("本金 (L1)", value=f_cfg["l1_p"]); l1r = st.number_input("利率 (L1)", value=f_cfg["l1_r"]); l1y = st.number_input("年限 (L1)", value=f_cfg["l1_y"]); l1m = st.number_input("已還月 (L1)", value=f_cfg["l1_m"])
            with lc2:
                st.write("**貸款 2**"); l2p = st.number_input("本金 (L2)", value=f_cfg["l2_p"]); l2r = st.number_input("利率 (L2)", value=f_cfg["l2_r"]); l2y = st.number_input("年限 (L2)", value=f_cfg["l2_y"]); l2m = st.number_input("已還月 (L2)", value=f_cfg["l2_m"])
            
            st.divider(); st.write("#### 2. 股票質押監控")
            gc1, gc2 = st.columns(2)
            with gc1: pl = st.number_input("質押借款金額 (TWD)", value=f_cfg["pledge_loan"])
            
            # --- 核心修正處：過濾不存在的代號 ---
            all_symbols = portfolio["股票代號"].tolist()
            valid_defaults = [t for t in f_cfg["pledge_targets"] if t in all_symbols]
            with gc2: pt = st.multiselect("擔保標的選擇", all_symbols, default=valid_defaults)
            
            if st.form_submit_button("💾 儲存財務資料"):
                if current_user != "All":
                    save_financial_config(current_user, {"cash_res": c_r, "l1_p": l1p, "l1_r": l1r, "l1_y": l1y, "l1_m": l1m, "l2_p": l2p, "l2_r": l2r, "l2_y": l2y, "l2_m": l2m, "pledge_loan": pl, "pledge_targets": pt})
                    st.success("已儲存！"); st.rerun()

        # 3. 質押即時數據 (移出 Form 外以免報錯)
        if pl > 0 and pt:
            st.divider()
            collateral_val = portfolio[portfolio["股票代號"].isin(pt)]["現值_TWD"].sum()
            m_ratio = (collateral_val / pl * 100)
            st.metric("🚨 即時質押維持率", f"{m_ratio:.2f}%", delta="門檻 130%")
            if len(pt) == 1:
                shares = portfolio[portfolio["股票代號"] == pt[0]]["股數"].values[0]
                st.error(f"🚩 斷頭預警價：{pt[0]} 跌破 **${(1.3 * pl / shares):.2f}**")

        rem1, rem2 = calculate_remaining_loan(l1p, l1r, l1y, l1m), calculate_remaining_loan(l2p, l2r, l2y, l2m)
        t_debt = rem1 + rem2 + pl; n_w = (t_val + c_r) - t_debt
        st.divider(); mc1, mc2, mc3 = st.columns(3)
        mc1.metric("💼 家庭總資產", f"${(t_val+c_r):,.0f}"); mc2.metric("📉 剩餘總負債", f"-${t_debt:,.0f}"); mc3.metric("🏆 家庭淨資產", f"${n_w:,.0f}")
else: st.info("請先新增持股。")
