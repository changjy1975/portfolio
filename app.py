import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import os

# --- 設定檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="台美股投資戰情室", layout="wide")
st.title("📈 智能投資組合戰情室")

# ==========================================
# 核心功能函數 (資料存取)
# ==========================================

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
    """刪除指定股票代號"""
    df = load_data()
    df = df[df["股票代號"] != symbol]
    save_data(df)

def get_exchange_rate():
    """獲取 USD/TWD 匯率"""
    try:
        ticker = yf.Ticker("USDTWD=X")
        rate = ticker.history(period="1d")['Close'].iloc[-1]
        return rate
    except:
        return 32.5

def get_current_prices(symbols):
    """獲取最新股價"""
    if not symbols: return {}
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
    except:
        return {}

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# ==========================================
# 技術分析邏輯 (新功能)
# ==========================================

def calculate_rsi(series, period=14):
    """計算 RSI 指標"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stock_technical(symbol):
    """針對單一個股進行半年週線分析"""
    try:
        # 1. 抓取半年週線資料
        stock = yf.Ticker(symbol)
        # 抓取稍微多一點資料以計算移動平均
        df = stock.history(period="1y", interval="1wk")
        
        if df.empty:
            return None, "無法獲取歷史資料"

        # 取最近半年的資料用於顯示，但保留舊資料算指標
        df_recent = df.tail(26) # 半年約 26 週

        # 2. 計算指標
        current_price = df['Close'].iloc[-1]
        
        # 支撐與壓力 (過去半年高低點)
        high_6m = df_recent['High'].max()
        low_6m = df_recent['Low'].min()
        
        # 移動平均 (20週均線，約等於季線/半年線趨勢)
        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        
        # RSI (14週)
        rsi_series = calculate_rsi(df['Close'], 14)
        rsi_curr = rsi_series.iloc[-1]

        # 3. 策略判定 (簡單邏輯)
        trend = "多頭排列 🐂" if current_price > ma_20 else "空頭/整理 🐻"
        
        # 建議進場價：支撐位附近 或 突破均線回測
        entry_price = low_6m * 1.02 # 支撐上方 2%
        entry_price_2 = ma_20 # 均線支撐
        
        # 建議出場價：壓力位附近
        exit_price = high_6m * 0.98 # 壓力下方 2%

        # 綜合建議
        if rsi_curr > 70:
            advice = "過熱，建議分批獲利了結"
            color = "red"
        elif rsi_curr < 30:
            advice = "超賣，可考慮分批佈局"
            color = "green"
        elif current_price > ma_20:
            advice = "趨勢向上，持股續抱"
            color = "orange"
        else:
            advice = "趨勢偏弱，觀望或區間操作"
            color = "gray"

        return {
            "current_price": current_price,
            "high_6m": high_6m,
            "low_6m": low_6m,
            "ma_20": ma_20,
            "rsi": rsi_curr,
            "trend": trend,
            "entry_target": max(entry_price, entry_price_2), # 取較高的支撐
            "exit_target": exit_price,
            "advice": advice,
            "advice_color": color,
            "history_df": df_recent
        }, None

    except Exception as e:
        return None, str(e)

# ==========================================
# 介面顯示組件
# ==========================================

# 定義庫存列表的欄位比例
COLS_RATIO = [1.3, 0.8, 1, 1, 1.3, 1.3, 1.3, 1, 0.5]

def display_headers():
    headers = ["代號", "股數", "均價", "現價", "總成本", "現值", "獲利", "報酬率%", "管理"]
    cols = st.columns(COLS_RATIO)
    for col, header in zip(cols, headers):
        col.markdown(f"**{header}**")
    st.markdown("<hr style='margin: 5px 0; border-top: 1px solid #ddd;'>", unsafe_allow_html=True)

def display_stock_rows(df, currency_type):
    for index, row in df.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        symbol = row["股票代號"]
        price = row["最新股價"]
        cost = row["總投入成本(原幣)"]
        val = row["現值(原幣)"]
        prof = row["獲利(原幣)"]
        roi = row["獲利率(%)"]
        
        color = "red" if prof > 0 else "green"
        fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"

        c1.write(f"**{symbol}**")
        c2.write(f"{row['股數']:.0f}")
        c3.write(f"{row['平均持有單價']:.2f}")
        c4.write(f"{price:.2f}")
        c5.write(fmt.format(cost))
        c6.write(fmt.format(val))
        c7.markdown(f":{color}[{fmt.format(prof)}]")
        c8.markdown(f":{color}[{roi:.2f}%]")
        
        if c9.button("🗑️", key=f"del_{symbol}"):
            remove_stock(symbol)
            st.rerun()

def display_subtotal_row(df, currency_type):
    total_cost = df["總投入成本(原幣)"].sum()
    total_val = df["現值(原幣)"].sum()
    total_profit = df["獲利(原幣)"].sum()
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    
    st.markdown("<hr style='margin: 5px 0; border-top: 2px solid #888;'>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
    fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
    color = "red" if total_profit > 0 else "green"
    
    c1.markdown("**🔹 類別小計**")
    c5.markdown(f"**{fmt.format(total_cost)}**")
    c6.markdown(f"**{fmt.format(total_val)}**")
    c7.markdown(f":{color}[**{fmt.format(total_profit)}**]")
    c8.markdown(f":{color}[**{roi:.2f}%**]")
    return total_val, total_profit

# ==========================================
# 主程式邏輯
# ==========================================

# 建立分頁
tab1, tab2 = st.tabs(["📊 庫存監控", "🧠 AI 技術分析與建議"])

df_record = load_data()
if not df_record.empty:
    usd_rate = get_exchange_rate()
    
    # 預先計算基礎資料供兩個分頁使用
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    df_record['總投入成本(原幣)'] = df_record['股數'] * df_record['持有成本單價']
    
    portfolio = df_record.groupby(["股票代號", "幣別"]).agg({
        "股數": "sum",
        "總投入成本(原幣)": "sum"
    }).reset_index()
    portfolio["平均持有單價"] = portfolio["總投入成本(原幣)"] / portfolio["股數"]

# --- 分頁 1: 庫存監控 (原有功能) ---
with tab1:
    with st.sidebar:
        st.header("📝 新增投資")
        with st.form("add_stock_form"):
            symbol_input = st.text_input("股票代號 (如 2330.TW)", value="2330.TW").upper().strip()
            qty_input = st.number_input("股數", min_value=1, value=1000)
            cost_input = st.number_input("單價 (原幣)", min_value=0.0, value=500.0)
            if st.form_submit_button("新增"):
                df = load_data()
                new_data = pd.DataFrame({"股票代號": [symbol_input], "股數": [qty_input], "持有成本單價": [cost_input]})
                df = pd.concat([df, new_data], ignore_index=True)
                save_data(df)
                st.success(f"已新增 {symbol_input}")
                st.rerun()
        
        if st.button("🚨 清空所有"):
            if os.path.exists(DATA_FILE): os.remove(DATA_FILE); st.rerun()

    if df_record.empty:
        st.info("請先從側邊欄新增投資紀錄。")
    else:
        st.sidebar.markdown(f"--- \n 💱 匯率: **{usd_rate:.2f}**")
        
        # 抓即時股價
        unique_symbols = portfolio["股票代號"].tolist()
        with st.spinner('正在同步市場數據...'):
            current_prices = get_current_prices(unique_symbols)
        
        portfolio["最新股價"] = portfolio["股票代號"].map(current_prices)
        portfolio = portfolio.dropna(subset=["最新股價"])

        # 計算
        portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
        portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
        portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
        
        portfolio["匯率因子"] = portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
        portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["匯率因子"]
        portfolio["總投入成本(TWD)"] = portfolio["總投入成本(原幣)"] * portfolio["匯率因子"]
        portfolio["獲利(TWD)"] = portfolio["現值(TWD)"] - portfolio["總投入成本(TWD)"]

        # 總資產
        total_val = portfolio["現值(TWD)"].sum()
        total_cost = portfolio["總投入成本(TWD)"].sum()
        total_profit = portfolio["獲利(TWD)"].sum()
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 總資產 (TWD)", f"${total_val:,.0f}")
        col2.metric("💳 總投入成本 (TWD)", f"${total_cost:,.0f}")
        col3.metric("📈 總獲利", f"${total_profit:,.0f}", f"{roi:.2f}%")
        st.markdown("---")

        # 分類顯示
        df_tw = portfolio[portfolio["幣別"] == "TWD"].copy()
        df_us = portfolio[portfolio["幣別"] == "USD"].copy()

        st.subheader("🇹🇼 台股庫存")
        if not df_tw.empty:
            display_headers()
            display_stock_rows(df_tw, "TWD")
            display_subtotal_row(df_tw, "TWD")
        else: st.write("無台股")

        st.write(""); st.write("")

        st.subheader("🇺🇸 美股庫存")
        if not df_us.empty:
            display_headers()
            display_stock_rows(df_us, "USD")
            us_val, us_prof = display_subtotal_row(df_us, "USD")
            st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.9em;'>約 NT$ {us_val*usd_rate:,.0f} | 獲利 NT$ {us_prof*usd_rate:,.0f}</div>", unsafe_allow_html=True)
        else: st.write("無美股")
        
        st.markdown("---")
        if st.button("🔄 刷新股價"): st.rerun()

# --- 分頁 2: 技術分析與建議 ---
with tab2:
    if df_record.empty:
        st.info("請先新增庫存股票，系統才能進行分析。")
    else:
        st.subheader("🧠 持股健診與進出建議 (週線級別)")
        st.markdown("針對您的持股進行 **過去半年週線 (Weekly)** 分析，提供未來三個月操作參考。")
        
        # 選擇要分析的股票
        stock_list = portfolio["股票代號"].tolist()
        selected_stock = st.selectbox("請選擇要分析的股票：", stock_list)

        if st.button(f"🔍 分析 {selected_stock}") or selected_stock:
            with st.spinner(f"正在分析 {selected_stock} 的技術型態..."):
                result, error = analyze_stock_technical(selected_stock)
                
                if error:
                    st.error(f"分析失敗: {error}")
                else:
                    # 1. 顯示關鍵數據
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("目前價格", f"{result['current_price']:.2f}")
                    c2.metric("半年最高 (壓力參考)", f"{result['high_6m']:.2f}")
                    c3.metric("半年最低 (支撐參考)", f"{result['low_6m']:.2f}")
                    c4.metric("RSI 強弱指標", f"{result['rsi']:.1f}")

                    st.markdown("### 📊 走勢圖 (近半年週線)")
                    # 繪製簡單圖表 (Close Price & MA20)
                    chart_data = result['history_df'][['Close']].copy()
                    chart_data['20週均線'] = chart_data['Close'].rolling(window=20).mean()
                    st.line_chart(chart_data)

                    # 2. AI 建議區塊
                    st.divider()
                    st.subheader("💡 系統操作建議 (未來3個月)")
                    
                    # 使用不同顏色的 Callout
                    st.markdown(f"#### 趨勢判斷： **{result['trend']}**")
                    
                    col_buy, col_sell = st.columns(2)
                    
                    with col_buy:
                        st.info(f"""
                        **🟢 建議進場/加碼點位**
                        
                        **${result['entry_target']:.2f} 附近**
                        
                        *邏輯：接近半年線支撐或波段低點，風險報酬比較佳。*
                        """)
                    
                    with col_sell:
                        st.warning(f"""
                        **🔴 建議停利/減碼點位**
                        
                        **${result['exit_target']:.2f} 附近**
                        
                        *邏輯：接近前波高點壓力區，建議分批獲利。*
                        """)

                    st.success(f"**綜合點評：** :{result['advice_color']}[**{result['advice']}**]")
                    
                    st.caption("* 免責聲明：本分析基於歷史數據計算之支撐壓力與技術指標，不代表未來股價保證，投資請自負風險。")
