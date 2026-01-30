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
# 2. 核心功能函數 (資料處理與報價)
# ==========================================

def load_data(user):
    path = f"portfolio_{user}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    """存檔並自動備份"""
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
    """
    使用 yf.Tickers 抓取即時報價 (最穩定的跨市場方案)
    """
    if not symbols: return {}
    quotes = {}
    try:
        tickers = yf.Tickers(" ".join(symbols))
        for s in symbols:
            try:
                # 優先從 fast_info 抓取最後價格
                price = tickers.tickers[s].fast_info.last_price
                if price is None or np.isnan(price):
                    # 備案：抓取歷史最後一筆
                    price = tickers.tickers[s].history(period="1d")['Close'].iloc[-1]
                quotes[s] = float(price)
            except:
                quotes[s] = 0.0 # 抓不到則設為 0，避免整列消失
        return quotes
    except: return {s: 0.0 for s in symbols}

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 3. 介面顯示組件 (Columns 比例)
# ==========================================
COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def display_headers(key_suffix, current_user):
    cols = st.columns(COLS_RATIO)
    headers = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本"), ("現值", "現值"), ("獲利", "獲利"), ("報酬率%", "獲利率(%)")]
    for col, (label, col_name) in zip(cols[:-1], headers):
        arrow = "▲" if st.session_state.sort_asc and st.session_state.sort_col == col_name else "▼" if st.session_state.sort_col == col_name else ""
        if col.button(f"{label}{arrow}", key=f"h_{col_name}_{key_suffix}_{current_user}"):
            st.session_state.sort_asc = not st.session_state.sort_asc if st.session_state.sort_col == col_name else False
            st.session_state.sort_col = col_name
            st.rerun()
    cols[-1].write("管理")

# ==========================================
# 4. 主程式
# ==========================================

if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    
    if current_user != "All":
        with st.form("add_form", clear_on_submit=True):
            st.subheader(f"📝 新增持股")
            s_in = st.text_input("代號 (如 2330.TW 或 NVDA)").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, step=1.0)
            c_in = st.number_input("持有成本單價", min_value=0.0, step=0.1)
            if st.form_submit_button("執行新增"):
                if s_in:
                    old_df = load_data(current_user)
                    new_row = pd.DataFrame([{"股票代號": s_in, "股數": q_in, "持有成本單價": c_in}])
                    save_data(pd.concat([old_df, new_row], ignore_index=True), current_user)
                    st.success(f"已新增 {s_in}")
                    st.rerun()

# --- 資料讀取與合併 ---
if current_user == "All":
    df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True)
else:
    df_record = load_data(current_user)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)"])

if not df_record.empty:
    # 彙整計算
    usd_rate = get_exchange_rate()
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(
        lambda g: pd.Series({
            '股數': g['股數'].sum(),
            '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()
        }), include_groups=False
    ).reset_index()

    # 抓取報價 (關鍵修復點)
    with st.spinner("更新即時行情中..."):
        price_map = get_latest_quotes(portfolio["股票代號"].tolist())
        portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    
    # 計算財務指標
    portfolio["總投入成本"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]) * 100
    
    # 換算台幣用於總計
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (usd_rate if r["幣別"] == "USD" else 1), axis=1)
    portfolio["獲利_TWD"] = portfolio.apply(lambda r: r["獲利"] * (usd_rate if r["幣別"] == "USD" else 1), axis=1)

    t_val = float(portfolio["現值_TWD"].sum())
    t_prof = float(portfolio["獲利_TWD"].sum())
    roi_pct = (t_prof / (t_val - t_prof) * 100) if (t_val - t_prof) != 0 else 0

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}")
        c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}")
        c3.metric("📊 總報酬率", f"{roi_pct:.2f}%")
        c4.metric("💱 美金匯率", f"{usd_rate:.2f}")

        st.divider()
        for label, cur in [("🇹🇼 台股列表", "TWD"), ("🇺🇸 美股列表", "USD")]:
            sub = portfolio[portfolio["幣別"] == cur]
            if not sub.empty:
                st.subheader(label)
                display_headers(cur, current_user)
                # 排序顯示
                sub_sorted = sub.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
                for _, row in sub_sorted.iterrows():
                    cols = st.columns(COLS_RATIO)
                    fmt = "{:,.0f}" if cur == "TWD" else "{:,.2f}"
                    color = "red" if row["獲利"] > 0 else "green"
                    cols[0].write(f"**{row['股票代號']}**")
                    cols[1].write(f"{row['股數']:.2f}")
                    cols[2].write(f"{row['平均持有單價']:.2f}")
                    cols[3].write(f"{row['最新股價']:.2f}")
                    cols[4].write(fmt.format(row['總投入成本']))
                    cols[5].write(fmt.format(row['現值']))
                    cols[6].markdown(f":{color}[{fmt.format(row['獲利'])}]")
                    cols[7].markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
                    if cols[8].button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
                        df_new = df_record[df_record["股票代號"] != row['股票代號']]
                        save_data(df_new, current_user)
                        st.rerun()
                st.markdown("---")

    with tab2:
        target = st.selectbox("分析標的：", portfolio["股票代號"].tolist())
        stock = yf.Ticker(target)
        hist = stock.history(period="1y")
        if not hist.empty:
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            st.metric(f"{target} RSI (14D)", f"{rsi:.2f}")
            st.line_chart(hist['Close'])

    with tab3:
        st.info("MPT 模擬引擎已就緒，請點擊按鈕執行（建議在股市收盤時分析）。")
        if st.button("🚀 執行組合模擬"):
            # ... (保持原本的 MPT 邏輯) ...
            st.success("模擬完成！")
else:
    st.info("尚未發現任何持股資料，請從左側選單新增。")
