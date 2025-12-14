import streamlit as st
import pandas as pd
import yfinance as yf
import os

# --- 設定檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="我的投資組合追蹤", layout="wide")
st.title("📈 即時投資組合儀表板")

# --- 核心功能函數 ---

def load_data():
    """讀取投資紀錄，如果檔案不存在則建立一個空的"""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df):
    """儲存投資紀錄"""
    df.to_csv(DATA_FILE, index=False)

def get_current_prices(symbols):
    """從 Yahoo Finance 獲取最新股價"""
    if not symbols:
        return {}
    
    # 下載數據 (一次下載多檔股票比較快)
    tickers = " ".join(symbols)
    try:
        data = yf.Tickers(tickers)
        prices = {}
        for symbol in symbols:
            # 嘗試獲取最新價格，有些股票可能只有 regularMarketPrice
            try:
                info = data.tickers[symbol].info
                # 優先順序: 當前價格 -> 前收盤價
                price = info.get('currentPrice') or info.get('regularMarketPreviousClose') or info.get('previousClose')
                prices[symbol] = price
            except:
                prices[symbol] = None
        return prices
    except Exception as e:
        st.error(f"獲取股價時發生錯誤: {e}")
        return {}

# --- 側邊欄：新增投資 ---
with st.sidebar:
    st.header("📝 新增/刪除 投資")
    
    with st.form("add_stock_form"):
        symbol_input = st.text_input("股票代號 (台股請加 .TW, 如 2330.TW)", value="2330.TW").upper()
        qty_input = st.number_input("持股股數", min_value=1, value=1000)
        cost_input = st.number_input("持有成本單價", min_value=0.0, value=500.0, format="%.2f")
        
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

# --- 主畫面：顯示投資組合 ---

df_record = load_data()

if df_record.empty:
    st.info("目前沒有投資紀錄，請從側邊欄新增。")
else:
    # 1. 資料聚合 (處理同一檔股票分批買入的情況)
    # 我們需要計算：總股數、加權平均成本
    df_record['總投入成本'] = df_record['股數'] * df_record['持有成本單價']
    
    portfolio = df_record.groupby("股票代號").agg({
        "股數": "sum",
        "總投入成本": "sum"
    }).reset_index()
    
    portfolio["平均持有單價"] = portfolio["總投入成本"] / portfolio["股數"]

    # 2. 獲取即時股價
    unique_symbols = portfolio["股票代號"].tolist()
    with st.spinner('正在更新最新股價...'):
        current_prices = get_current_prices(unique_symbols)

    # 3. 計算各項指標
    portfolio["最新股價"] = portfolio["股票代號"].map(current_prices)
    
    # 移除無法抓到股價的股票 (避免計算錯誤)
    portfolio = portfolio.dropna(subset=["最新股價"])

    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]) * 100

    # 4. 整理顯示欄位順序與格式
    display_cols = [
        "股票代號", "股數", "平均持有單價", "最新股價", 
        "總投入成本", "現值", "獲利", "獲利率(%)"
    ]
    final_df = portfolio[display_cols]

    # --- 顯示總資產摘要 ---
    total_value = final_df["現值"].sum()
    total_cost = final_df["總投入成本"].sum()
    total_profit = final_df["獲利"].sum()
    total_roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("總資產現值", f"${total_value:,.0f}")
    col2.metric("總投入成本", f"${total_cost:,.0f}")
    col3.metric("總獲利 / 報酬率", f"${total_profit:,.0f}", f"{total_roi:.2f}%")

    st.divider()

    # --- 顯示詳細表格 ---
    st.subheader("詳細庫存")

    # 使用 Streamlit 的 dataframe 進行美化顯示
    st.dataframe(
        final_df.style.format({
            "平均持有單價": "{:.2f}",
            "最新股價": "{:.2f}",
            "總投入成本": "{:,.0f}",
            "現值": "{:,.0f}",
            "獲利": "{:,.0f}",
            "獲利率(%)": "{:.2f}%"
        }).map(lambda x: 'color: red' if x > 0 else 'color: green', subset=['獲利', '獲利率(%)']), # 台股紅漲綠跌
        use_container_width=True,
        hide_index=True
    )

    st.caption("* 註：股價資料來源為 Yahoo Finance，可能有約 15 分鐘延遲。台股代號請加上 .TW (上市) 或 .TWO (上櫃)。")
    
    if st.button("🔄 手動刷新股價"):
        st.rerun()