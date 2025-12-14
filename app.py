import streamlit as st
import pandas as pd
import yfinance as yf
import os

# --- 設定檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="台美股投資組合追蹤", layout="wide")
st.title("📈 跨市場投資組合儀表板 (含分類小計)")

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

def remove_stock(symbol):
    """刪除指定股票代號的所有紀錄"""
    df = load_data()
    df = df[df["股票代號"] != symbol]
    save_data(df)

def get_exchange_rate():
    """獲取美金兌台幣即時匯率"""
    try:
        ticker = yf.Ticker("USDTWD=X")
        rate = ticker.history(period="1d")['Close'].iloc[-1]
        return rate
    except Exception:
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
    """判斷幣別"""
    if ".TW" in symbol or ".TWO" in symbol:
        return "TWD"
    return "USD"

# --- 介面顯示函數 ---

# 定義欄位比例 (讓標題、列表、小計都能對齊)
COLS_RATIO = [1.3, 0.8, 1, 1, 1.3, 1.3, 1.3, 1, 0.5]

def display_headers():
    """顯示表格標題"""
    headers = ["代號", "股數", "均價", "現價", "總成本", "現值", "獲利", "報酬率%", "管理"]
    cols = st.columns(COLS_RATIO)
    for col, header in zip(cols, headers):
        col.markdown(f"**{header}**")
    st.markdown("<hr style='margin: 5px 0; border-top: 1px solid #ddd;'>", unsafe_allow_html=True)

def display_stock_rows(df, currency_type):
    """顯示每一行股票資料"""
    for index, row in df.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        
        symbol = row["股票代號"]
        price = row["最新股價"]
        cost_total = row["總投入成本(原幣)"]
        value_total = row["現值(原幣)"]
        profit = row["獲利(原幣)"]
        roi = row["獲利率(%)"]

        color = "red" if profit > 0 else "green"
        fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
        
        c1.write(f"**{symbol}**")
        c2.write(f"{row['股數']:.0f}")
        c3.write(f"{row['平均持有單價']:.2f}")
        c4.write(f"{price:.2f}")
        c5.write(fmt.format(cost_total))
        c6.write(fmt.format(value_total))
        c7.markdown(f":{color}[{fmt.format(profit)}]")
        c8.markdown(f":{color}[{roi:.2f}%]")
        
        if c9.button("🗑️", key=f"del_{symbol}"):
            remove_stock(symbol)
            st.rerun()

def display_subtotal_row(df, currency_type):
    """顯示分類小計行"""
    # 計算小計
    total_cost = df["總投入成本(原幣)"].sum()
    total_value = df["現值(原幣)"].sum()
    total_profit = df["獲利(原幣)"].sum()
    total_roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    
    # 畫分隔線
    st.markdown("<hr style='margin: 5px 0; border-top: 2px solid #888;'>", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
    
    fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
    color = "red" if total_profit > 0 else "green"

    c1.markdown("**🔹 類別小計**")
    # c2, c3, c4 留白
    c5.markdown(f"**{fmt.format(total_cost)}**")
    c6.markdown(f"**{fmt.format(total_value)}**")
    c7.markdown(f":{color}[**{fmt.format(total_profit)}**]")
    c8.markdown(f":{color}[**{total_roi:.2f}%**]")
    
    return total_value, total_profit # 回傳值供後續換算使用

# --- 側邊欄：新增投資 ---
with st.sidebar:
    st.header("📝 新增投資")
    with st.form("add_stock_form"):
        st.write("輸入範例：`2330.TW` 或 `NVDA`")
        symbol_input = st.text_input("股票代號", value="2330.TW").upper().strip()
        qty_input = st.number_input("股數", min_value=1, value=1000)
        cost_input = st.number_input("單價 (原幣)", min_value=0.0, value=500.0)
        
        if st.form_submit_button("新增"):
            df = load_data()
            new_data = pd.DataFrame({"股票代號": [symbol_input], "股數": [qty_input], "持有成本單價": [cost_input]})
            df = pd.concat([df, new_data], ignore_index=True)
            save_data(df)
            st.success(f"已新增 {symbol_input}")
            st.rerun()

    if st.button("🚨 清空所有投資"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
            st.rerun()

# --- 主畫面邏輯 ---

df_record = load_data()

if df_record.empty:
    st.info("目前沒有投資紀錄，請從側邊欄新增。")
else:
    usd_rate = get_exchange_rate()
    st.sidebar.markdown(f"--- \n 💱 匯率 (USD/TWD): **{usd_rate:.2f}**")

    # 資料計算
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    df_record['總投入成本(原幣)'] = df_record['股數'] * df_record['持有成本單價']

    # 聚合資料
    portfolio = df_record.groupby(["股票代號", "幣別"]).agg({
        "股數": "sum",
        "總投入成本(原幣)": "sum"
    }).reset_index()
    portfolio["平均持有單價"] = portfolio["總投入成本(原幣)"] / portfolio["股數"]

    # 抓取股價
    unique_symbols = portfolio["股票代號"].tolist()
    with st.spinner('更新最新股價中...'):
        current_prices = get_current_prices(unique_symbols)

    portfolio["最新股價"] = portfolio["股票代號"].map(current_prices)
    portfolio = portfolio.dropna(subset=["最新股價"])

    # 計算基本欄位
    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100

    # 換算台幣總表
    portfolio["匯率因子"] = portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
    portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["匯率因子"]
    portfolio["總投入成本(TWD)"] = portfolio["總投入成本(原幣)"] * portfolio["匯率因子"]
    portfolio["獲利(TWD)"] = portfolio["現值(TWD)"] - portfolio["總投入成本(TWD)"]

    # --- 1. 總資產看板 ---
    total_val = portfolio["現值(TWD)"].sum()
    total_cost = portfolio["總投入成本(TWD)"].sum()
    total_profit = portfolio["獲利(TWD)"].sum()
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 總資產 (TWD)", f"${total_val:,.0f}")
    col2.metric("💳 總投入成本 (TWD)", f"${total_cost:,.0f}")
    col3.metric("📈 總獲利", f"${total_profit:,.0f}", f"{roi:.2f}%")

    st.markdown("---")

    # 分類資料
    df_tw = portfolio[portfolio["幣別"] == "TWD"].copy()
    df_us = portfolio[portfolio["幣別"] == "USD"].copy()

    # --- 2. 台股庫存區塊 ---
    st.subheader("🇹🇼 台股庫存")
    if not df_tw.empty:
        display_headers()
        display_stock_rows(df_tw, "TWD")
        # 顯示台股小計
        display_subtotal_row(df_tw, "TWD")
    else:
        st.write("目前無台股持倉")

    st.write("") # 間距
    st.write("") 

    # --- 3. 美股庫存區塊 ---
    st.subheader("🇺🇸 美股庫存")
    if not df_us.empty:
        display_headers()
        display_stock_rows(df_us, "USD")
        
        # 顯示美股小計 (美金)
        us_val, us_profit = display_subtotal_row(df_us, "USD")
        
        # 顯示美股折合台幣 (補充資訊)
        st.markdown(
            f"""
            <div style="text-align: right; color: gray; font-size: 0.9em; margin-top: 5px;">
            換算台幣約： 現值 NT$ {us_val * usd_rate:,.0f} | 
            獲利 NT$ {us_profit * usd_rate:,.0f}
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.write("目前無美股持倉")

    st.markdown("---")
    if st.button("🔄 刷新最新股價"):
        st.rerun()
