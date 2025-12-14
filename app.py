import streamlit as st
import pandas as pd
import yfinance as yf
import os

# --- 設定檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="台美股投資組合追蹤", layout="wide")
st.title("📈 跨市場投資組合儀表板 (含管理功能)")

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
    # 保留不等於該代號的資料 (即刪除該代號)
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

# --- 顯示用的輔助函數 (客製化列表與刪除按鈕) ---
def display_stock_rows(df, currency_type, usd_rate=1):
    """
    客製化顯示每一行股票，包含刪除按鈕
    currency_type: 'TWD' 或 'USD'
    """
    # 定義欄位比例 (最後一欄留給按鈕)
    # 代號, 股數, 均價, 現價, 總成本, 現值, 獲利, %, 按鈕
    cols_ratio = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.5]
    
    # 顯示標題列
    headers = ["代號", "股數", "均價", "現價", "總成本", "現值", "獲利", "報酬率%", "管理"]
    cols = st.columns(cols_ratio)
    for col, header in zip(cols, headers):
        col.markdown(f"**{header}**")
    
    st.divider()

    # 顯示資料列
    for index, row in df.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(cols_ratio)
        
        symbol = row["股票代號"]
        price = row["最新股價"]
        cost_total = row["總投入成本(原幣)"]
        value_total = row["現值(原幣)"]
        profit = row["獲利(原幣)"]
        roi = row["獲利率(%)"]

        # 設定顏色
        color = "red" if profit > 0 else "green"
        
        # 欄位內容填充
        c1.write(f"**{symbol}**")
        c2.write(f"{row['股數']:.0f}")
        c3.write(f"{row['平均持有單價']:.2f}")
        c4.write(f"{price:.2f}")
        
        # 金額顯示 (美金顯示小數位)
        fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
        c5.write(fmt.format(cost_total))
        c6.write(fmt.format(value_total))
        
        # 獲利與報酬率帶顏色
        c7.markdown(f":{color}[{fmt.format(profit)}]")
        c8.markdown(f":{color}[{roi:.2f}%]")
        
        # --- 刪除按鈕 ---
        # 使用 key 確保每個按鈕唯一
        if c9.button("🗑️", key=f"del_{symbol}", help=f"刪除 {symbol}"):
            remove_stock(symbol)
            st.rerun() # 刪除後立即刷新頁面
            
        # 美股額外顯示台幣估算
        if currency_type == "USD":
            # 這裡用 caption 小字顯示在該行下方或旁邊，避免太擠
            pass 

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

    portfolio = df_record.groupby(["股票代號", "幣別"]).agg({
        "股數": "sum",
        "總投入成本(原幣)": "sum"
    }).reset_index()
    portfolio["平均持有單價"] = portfolio["總投入成本(原幣)"] / portfolio["股數"]

    unique_symbols = portfolio["股票代號"].tolist()
    with st.spinner('更新股價中...'):
        current_prices = get_current_prices(unique_symbols)

    portfolio["最新股價"] = portfolio["股票代號"].map(current_prices)
    portfolio = portfolio.dropna(subset=["最新股價"])

    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100

    portfolio["匯率因子"] = portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
    portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["匯率因子"]
    portfolio["總投入成本(TWD)"] = portfolio["總投入成本(原幣)"] * portfolio["匯率因子"]
    portfolio["獲利(TWD)"] = portfolio["現值(TWD)"] - portfolio["總投入成本(TWD)"]

    # 總資產顯示
    total_val = portfolio["現值(TWD)"].sum()
    total_cost = portfolio["總投入成本(TWD)"].sum()
    total_profit = portfolio["獲利(TWD)"].sum()
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("總資產 (TWD)", f"${total_val:,.0f}")
    col2.metric("總成本 (TWD)", f"${total_cost:,.0f}")
    col3.metric("總獲利", f"${total_profit:,.0f}", f"{roi:.2f}%")

    st.markdown("---")

    # 分類顯示
    df_tw = portfolio[portfolio["幣別"] == "TWD"].copy()
    df_us = portfolio[portfolio["幣別"] == "USD"].copy()

    # --- 台股區塊 ---
    st.subheader("🇹🇼 台股庫存")
    if not df_tw.empty:
        display_stock_rows(df_tw, "TWD")
    else:
        st.write("無台股持倉")

    st.markdown("---")

    # --- 美股區塊 ---
    st.subheader("🇺🇸 美股庫存")
    if not df_us.empty:
        us_total_twd = df_us["現值(TWD)"].sum()
        st.caption(f"美股總現值折合台幣約: ${us_total_twd:,.0f}")
        display_stock_rows(df_us, "USD", usd_rate)
    else:
        st.write("無美股持倉")

    if st.button("🔄 刷新"):
        st.rerun()
