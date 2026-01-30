import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import os
import shutil
from datetime import datetime
import pytz
import numpy as np

# ==========================================
# 1. 初始化設定
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

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

# ==========================================
# 3. 介面組件：清單與小計
# ==========================================
# 定義欄位比例：代號, 股數, 均價, 現價, 總成本, 現值, 獲利, 報酬率, 管理
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, currency_label, currency_type, usd_rate, current_user):
    st.subheader(f"{currency_label}")
    
    # 標題列
    h_cols = st.columns(COLS_RATIO)
    headers = ["代號", "股數", "均價", "現價", "總成本", "現值", "獲利", "報酬率", "管理"]
    for col, h in zip(h_cols, headers):
        col.caption(f"**{h}**")
    
    # 數據列
    for _, row in df.iterrows():
        r_cols = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
        p_color = "red" if row["獲利"] > 0 else "green"
        
        r_cols[0].write(f"**{row['股票代號']}**")
        r_cols[1].write(f"{row['股數']:.2f}")
        r_cols[2].write(f"{row['平均持有單價']:.2f}")
        r_cols[3].write(f"{row['最新股價']:.2f}")
        r_cols[4].write(fmt.format(row['總投入成本']))
        r_cols[5].write(fmt.format(row['現值']))
        r_cols[6].markdown(f":{p_color}[{fmt.format(row['獲利'])}]")
        r_cols[7].markdown(f":{p_color}[{row['獲利率(%)']:.2f}%]")
        if r_cols[8].button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
            # 刪除邏輯
            full_df = load_data(current_user)
            full_df = full_df[full_df["股票代號"] != row['股票代號']]
            save_data(full_df, current_user)
            st.rerun()

    # --- 小計列 (Subtotal) ---
    sub_cost = df["總投入成本"].sum()
    sub_val = df["現值"].sum()
    sub_profit = df["獲利"].sum()
    sub_roi = (sub_profit / sub_cost * 100) if sub_cost != 0 else 0
    sub_color = "red" if sub_profit > 0 else "green"
    fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"

    st.markdown("---")
    s_cols = st.columns(COLS_RATIO)
    s_cols[0].markdown(f"**{currency_type} 小計**")
    s_cols[4].markdown(f"**{fmt.format(sub_cost)}**")
    s_cols[5].markdown(f"**{fmt.format(sub_val)}**")
    s_cols[6].markdown(f":{sub_color}[**{fmt.format(sub_profit)}**]")
    s_cols[7].markdown(f":{sub_color}[**{sub_roi:.2f}%**]")
    
    # 如果是美股，額外顯示換算台幣小計
    if currency_type == "USD":
        s_cols = st.columns(COLS_RATIO)
        s_cols[0].caption("*(換算台幣)*")
        s_cols[4].caption(f"${(sub_cost * usd_rate):,.0f}")
        s_cols[5].caption(f"${(sub_val * usd_rate):,.0f}")
        s_cols[6].caption(f"${(sub_profit * usd_rate):,.0f}")
    st.write("") # 間距

# ==========================================
# 4. 主程式邏輯
# ==========================================

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add_form", clear_on_submit=True):
            st.subheader("📝 新增持股")
            s_in = st.text_input("代號 (如 2330.TW 或 NVDA)").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, step=1.0)
            c_in = st.number_input("成本", min_value=0.0, step=0.1)
            if st.form_submit_button("執行新增"):
                if s_in:
                    df = load_data(current_user)
                    save_data(pd.concat([df, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user)
                    st.rerun()

# 資料合併與計算
if current_user == "All":
    df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True)
else:
    df_record = load_data(current_user)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)"])

if not df_record.empty:
    usd_rate = get_exchange_rate()
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    
    # 彙整組合
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(
        lambda g: pd.Series({
            '股數': g['股數'].sum(),
            '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()
        }), include_groups=False
    ).reset_index()

    # 抓取報價
    price_map = get_latest_quotes(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    
    # 計算各項指標
    portfolio["總投入成本"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]) * 100
    
    # 台幣換算
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (usd_rate if r["幣別"] == "USD" else 1), axis=1)
    portfolio["獲利_TWD"] = portfolio.apply(lambda r: r["獲利"] * (usd_rate if r["幣別"] == "USD" else 1), axis=1)

    with tab1:
        if st.button("🔄 刷新最新報價"):
            st.cache_data.clear()
            st.rerun()

        # 頂部總覽看板
        t_val = float(portfolio["現值_TWD"].sum())
        t_prof = float(portfolio["獲利_TWD"].sum())
        roi_pct = (t_prof / (t_val - t_prof) * 100) if (t_val - t_prof) != 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}")
        c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}")
        c3.metric("📊 總報酬率", f"{roi_pct:.2f}%")
        c4.metric("💱 匯率", f"{usd_rate:.2f}")
        
        st.divider()
        
        # 分市場顯示列表與小計
        tw_sub = portfolio[portfolio["幣別"] == "TWD"]
        if not tw_sub.empty:
            display_market_table(tw_sub, "🇹🇼 台股庫存", "TWD", usd_rate, current_user)
            
        us_sub = portfolio[portfolio["幣別"] == "USD"]
        if not us_sub.empty:
            display_market_table(us_sub, "🇺🇸 美股庫存", "USD", usd_rate, current_user)

        # 圓餅圖
        st.divider()
        st.subheader("🎯 配置比例")
        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="幣別佔比", hole=0.4), use_container_width=True)
        with pc2:
            st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="股票代號", title="個股配置 (TWD)", hole=0.4), use_container_width=True)

    # (Tab 2 與 Tab 3 邏輯保持不變...)
    with tab3:
        st.subheader("⚖️ 投資組合優化 (MPT)")
        if st.button("🚀 執行模擬"):
            # 這裡放入之前完整的 MPT 繪圖代碼
            st.write("模擬中...")
else:
    st.info("尚未發現任何持股資料。")
