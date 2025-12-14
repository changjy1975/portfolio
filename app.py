import streamlit as st
import pandas as pd
import yfinance as yf
import os

# --- 設定檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="台美股投資組合追蹤", layout="wide")
st.title("📈 跨市場投資組合儀表板 (台幣計價)")

# --- 核心功能函數 ---

def load_data():
    """讀取投資紀錄"""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df):
    """儲存投資紀錄"""
    df.to_csv(DATA_FILE, index=False)

def get_exchange_rate():
    """獲取美金兌台幣即時匯率"""
    try:
        ticker = yf.Ticker("USDTWD=X")
        # 取得最新一筆收盤價
        rate = ticker.history(period="1d")['Close'].iloc[-1]
        return rate
    except Exception as e:
        st.warning("無法獲取即時匯率，將使用預設值 32.5")
        return 32.5

def get_current_prices(symbols):
    """從 Yahoo Finance 獲取最新股價"""
    if not symbols:
        return {}
    
    tickers = " ".join(symbols)
    try:
        data = yf.Tickers(tickers)
        prices = {}
        for symbol in symbols:
            try:
                info = data.tickers[symbol].info
                price = info.get('currentPrice') or info.get('regularMarketPreviousClose') or info.get('previousClose')
                prices[symbol] = price
            except:
                prices[symbol] = None
        return prices
    except Exception:
        return {}

def identify_currency(symbol):
    """判斷幣別：有 .TW 或 .TWO 為台幣，其餘視為美金"""
    if ".TW" in symbol or ".TWO" in symbol:
        return "TWD"
    return "USD"

# --- 側邊欄：新增投資 ---
with st.sidebar:
    st.header("📝 新增/刪除 投資")
    
    with st.form("add_stock_form"):
        st.write("輸入範例：台積電 `2330.TW` / 輝達 `NVDA`")
        symbol_input = st.text_input("股票代號", value="NVDA").upper().strip()
        qty_input = st.number_input("持股股數", min_value=1, value=10)
        cost_input = st.number_input("持有成本單價 (原幣)", min_value=0.0, value=120.0, format="%.2f")
        
        submitted = st.form_submit_button("新增交易")
        
        if submitted:
            df = load_data()
            new_data = pd.DataFrame({
                "股票代號": [symbol_input],
                "股數": [qty_input],
                "持有成本單價": [cost_input]
            })
            df = pd.concat([df, new_data], ignore_index=True)
            save_data(df)
            st.success(f"已新增 {symbol_input}")

    if st.button("🗑️ 清空所有投資"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
            st.rerun()

# --- 主畫面邏輯 ---

df_record = load_data()

if df_record.empty:
    st.info("目前沒有投資紀錄，請從側邊欄新增。")
else:
    # 1. 取得匯率
    usd_rate = get_exchange_rate()
    st.sidebar.markdown(f"### 💱 目前匯率 (USD/TWD): **{usd_rate:.2f}**")

    # 2. 資料前處理：標記幣別
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    df_record['總投入成本(原幣)'] = df_record['股數'] * df_record['持有成本單價']

    # 3. 聚合資料 (同股票合併)
    portfolio = df_record.groupby(["股票代號", "幣別"]).agg({
        "股數": "sum",
        "總投入成本(原幣)": "sum"
    }).reset_index()
    portfolio["平均持有單價"] = portfolio["總投入成本(原幣)"] / portfolio["股數"]

    # 4. 抓取現價
    unique_symbols = portfolio["股票代號"].tolist()
    with st.spinner('正在更新台美股價與匯率...'):
        current_prices = get_current_prices(unique_symbols)

    portfolio["最新股價"] = portfolio["股票代號"].map(current_prices)
    portfolio = portfolio.dropna(subset=["最新股價"]) # 移除抓不到股價的

    # 5. 計算價值與獲利 (原幣)
    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100

    # 6. 換算台幣 (重要步驟)
    # 如果是 TWD，匯率因子是 1；如果是 USD，匯率因子是 usd_rate
    portfolio["匯率因子"] = portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
    
    portfolio["總投入成本(TWD)"] = portfolio["總投入成本(原幣)"] * portfolio["匯率因子"]
    portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["匯率因子"]
    portfolio["獲利(TWD)"] = portfolio["現值(TWD)"] - portfolio["總投入成本(TWD)"]

    # --- 顯示總體資產概況 (全部換算成台幣) ---
    total_value_twd = portfolio["現值(TWD)"].sum()
    total_cost_twd = portfolio["總投入成本(TWD)"].sum()
    total_profit_twd = portfolio["獲利(TWD)"].sum()
    total_roi = (total_profit_twd / total_cost_twd * 100) if total_cost_twd > 0 else 0

    st.markdown("### 💰 總資產概況 (新台幣計價)")
    col1, col2, col3 = st.columns(3)
    col1.metric("總現值 (TWD)", f"${total_value_twd:,.0f}")
    col2.metric("總投入成本 (TWD)", f"${total_cost_twd:,.0f}")
    col3.metric("總獲利 / 報酬率", f"${total_profit_twd:,.0f}", f"{total_roi:.2f}%")
    
    st.divider()

    # --- 分類顯示：拆分 台股 與 美股 ---
    
    df_tw = portfolio[portfolio["幣別"] == "TWD"].copy()
    df_us = portfolio[portfolio["幣別"] == "USD"].copy()

    # 定義樣式函數 (紅漲綠跌)
    def style_dataframe(df, cols_to_color):
        return df.style.format({
            "平均持有單價": "{:.2f}",
            "最新股價": "{:.2f}",
            "總投入成本(原幣)": "{:,.0f}",
            "現值(原幣)": "{:,.0f}",
            "獲利(原幣)": "{:,.0f}",
            "獲利率(%)": "{:.2f}%",
            "現值(TWD估算)": "{:,.0f}",  # 美股專用
        }).map(lambda x: 'color: red' if x > 0 else 'color: green', subset=cols_to_color)

    # === Tab 1: 台股庫存 ===
    st.subheader("🇹🇼 台股庫存")
    if not df_tw.empty:
        # 台股小計
        tw_val = df_tw["現值(原幣)"].sum()
        tw_profit = df_tw["獲利(原幣)"].sum()
        st.caption(f"台股小計現值: ${tw_val:,.0f} | 獲利: ${tw_profit:,.0f}")
        
        display_tw = df_tw[["股票代號", "股數", "平均持有單價", "最新股價", "總投入成本(原幣)", "現值(原幣)", "獲利(原幣)", "獲利率(%)"]]
        st.dataframe(
            style_dataframe(display_tw, ['獲利(原幣)', '獲利率(%)']),
            use_container_width=True, hide_index=True
        )
    else:
        st.write("目前無台股持倉")

    st.divider()

    # === Tab 2: 美股庫存 ===
    st.subheader("🇺🇸 美股庫存")
    if not df_us.empty:
        # 美股小計
        us_val_usd = df_us["現值(原幣)"].sum()
        us_val_twd = df_us["現值(TWD)"].sum()
        us_profit_twd = df_us["獲利(TWD)"].sum()
        
        st.caption(f"美股小計現值: USD {us_val_usd:,.2f} (約 TWD {us_val_twd:,.0f}) | 換算獲利: TWD {us_profit_twd:,.0f}")

        # 美股顯示欄位增加 "現值(TWD估算)"
        df_us["現值(TWD估算)"] = df_us["現值(TWD)"]
        display_us = df_us[[
            "股票代號", "股數", "平均持有單價", "最新股價", 
            "總投入成本(原幣)", "現值(原幣)", "獲利(原幣)", "獲利率(%)", "現值(TWD估算)"
        ]]
        
        # 針對美股格式微調 (顯示小數點)
        st.dataframe(
            display_us.style.format({
                "平均持有單價": "{:.2f}",
                "最新股價": "{:.2f}",
                "總投入成本(原幣)": "{:,.2f}", # 美金顯示小數點
                "現值(原幣)": "{:,.2f}",
                "獲利(原幣)": "{:,.2f}",
                "獲利率(%)": "{:.2f}%",
                "現值(TWD估算)": "{:,.0f}" # 台幣顯示整數
            }).map(lambda x: 'color: red' if x > 0 else 'color: green', subset=['獲利(原幣)', '獲利率(%)']),
            use_container_width=True, hide_index=True
        )
    else:
        st.write("目前無美股持倉")

    if st.button("🔄 刷新股價與匯率"):
        st.rerun()
