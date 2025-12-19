import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
from datetime import datetime
import pytz
import numpy as np # 新增 numpy 用於數學計算

# --- 設定檔案儲存路徑 ---
DATA_FILE = "portfolio.csv"

# --- 頁面設定 ---
st.set_page_config(page_title="台美股投資戰情室", layout="wide")
st.title("📈 智能投資組合戰情室")

# ==========================================
# 狀態初始化
# ==========================================
if "sort_col" not in st.session_state:
    st.session_state.sort_col = "獲利(原幣)"
if "sort_asc" not in st.session_state:
    st.session_state.sort_asc = False
if "last_updated" not in st.session_state:
    st.session_state.last_updated = "尚未更新"

# ==========================================
# 頂部控制區
# ==========================================
col_refresh, col_time = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 刷新全部數據"):
        st.session_state.last_updated = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()
with col_time:
    st.markdown(f"<div style='padding-top: 10px; color: gray;'>最後更新時間: {st.session_state.last_updated} (台股來源: Yahoo Fast Info)</div>", unsafe_allow_html=True)

st.divider()

# ==========================================
# 核心功能函數
# ==========================================

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def remove_stock(symbol):
    df = load_data()
    df = df[df["股票代號"] != symbol]
    save_data(df)

def get_exchange_rate():
    try:
        ticker = yf.Ticker("USDTWD=X")
        rate = ticker.fast_info.last_price
        if rate is None or pd.isna(rate):
             rate = ticker.history(period="1d")['Close'].iloc[-1]
        return rate
    except:
        return 32.5

def get_current_prices(symbols):
    if not symbols: return {}
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            price = None
            try:
                price = ticker.fast_info.last_price
            except:
                price = None

            if price is None or pd.isna(price):
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
            
            if price is None or pd.isna(price):
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPreviousClose') or info.get('previousClose')
            
            prices[symbol] = price
        except:
            prices[symbol] = None
    return prices

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# ==========================================
# 技術分析邏輯 (Tab 2)
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stock_technical(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", interval="1wk")
        if df.empty: return None, "無法獲取歷史資料"
        df_recent = df.tail(26) 
        current_price = df['Close'].iloc[-1]
        high_6m = df_recent['High'].max()
        low_6m = df_recent['Low'].min()
        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        rsi_series = calculate_rsi(df['Close'], 14)
        rsi_curr = rsi_series.iloc[-1]
        trend = "多頭排列 🐂" if current_price > ma_20 else "空頭/整理 🐻"
        entry_price = max(low_6m * 1.02, ma_20)
        exit_price = high_6m * 0.98
        
        if rsi_curr > 70: advice, color = "過熱，建議分批獲利", "red"
        elif rsi_curr < 30: advice, color = "超賣，可考慮分批佈局", "green"
        elif current_price > ma_20: advice, color = "趨勢向上，持股續抱", "orange"
        else: advice, color = "趨勢偏弱，觀望或區間操作", "gray"

        return {
            "current_price": current_price, "high_6m": high_6m, "low_6m": low_6m,
            "ma_20": ma_20, "rsi": rsi_curr, "trend": trend,
            "entry_target": entry_price, "exit_target": exit_price,
            "advice": advice, "advice_color": color, "history_df": df_recent
        }, None
    except Exception as e:
        return None, str(e)

# ==========================================
# 投資組合分析邏輯 (Tab 3 新增)
# ==========================================
def perform_portfolio_analysis(portfolio_df):
    """
    執行投資組合深度分析：相關性、波動度、再平衡建議
    """
    symbols = portfolio_df["股票代號"].unique().tolist()
    if not symbols: return None, "無持股資料"

    try:
        # 1. 抓取過去 36 個月的收盤價
        tickers_str = " ".join(symbols)
        # 使用 auto_adjust=True 獲取還原權息股價，計算報酬率較準確
        hist_data = yf.download(tickers_str, period="3y", interval="1d", auto_adjust=True)['Close']
        
        # 如果只有一支股票，yf.download 回傳的是 Series，需轉為 DataFrame
        if isinstance(hist_data, pd.Series):
            hist_data = hist_data.to_frame(name=symbols[0])
            
        # 清理數據 (去除空值)
        hist_data = hist_data.dropna(how='all')
        
        # 2. 計算日報酬率
        returns = hist_data.pct_change().dropna()
        
        # 3. 計算相關係數矩陣
        corr_matrix = returns.corr()
        
        # 4. 準備再平衡建議
        suggestions = []
        
        # 4.1 集中度檢查
        total_val = portfolio_df["現值(TWD)"].sum()
        for idx, row in portfolio_df.iterrows():
            weight = row["現值(TWD)"] / total_val
            if weight > 0.3: # 單一個股超過 30%
                suggestions.append(f"⚠️ **集中度風險**：{row['股票代號']} 佔比達 {weight*100:.1f}%，建議適度減碼以分散風險。")
        
        # 4.2 相關性檢查 (找出高度連動的股票對)
        # 只看相關係數 > 0.8 的 (且不是自己對自己)
        high_corr_pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                c = corr_matrix.iloc[i, j]
                if c > 0.8:
                    high_corr_pairs.append(f"{cols[i]} & {cols[j]} (相關係數 {c:.2f})")
        
        if high_corr_pairs:
            suggestions.append(f"🔗 **高相關性警示**：以下股票走勢高度連動，分散效果有限：\n" + ", ".join(high_corr_pairs))
        
        if not suggestions:
            suggestions.append("✅ **太棒了！** 目前投資組合配置健康，無明顯集中或連動風險。")

        return {
            "corr_matrix": corr_matrix,
            "suggestions": suggestions,
            "returns_std": returns.std() * np.sqrt(252) * 100 # 年化波動率
        }, None

    except Exception as e:
        return None, str(e)

# ==========================================
# 介面顯示組件
# ==========================================

COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def update_sort(column_name):
    if st.session_state.sort_col == column_name:
        st.session_state.sort_asc = not st.session_state.sort_asc
    else:
        st.session_state.sort_col = column_name
        st.session_state.sort_asc = False

def get_header_label(label, col_name):
    if st.session_state.sort_col == col_name:
        arrow = "▲" if st.session_state.sort_asc else "▼"
        return f"{label} {arrow}"
    return label

def display_headers(key_suffix):
    st.markdown("<div style='padding-right: 15px;'>", unsafe_allow_html=True) 
    cols = st.columns(COLS_RATIO)
    headers_map = [
        ("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), 
        ("現價", "最新股價"), ("總成本", "總投入成本(原幣)"), 
        ("現值", "現值(原幣)"), ("獲利", "獲利(原幣)"), ("報酬率%", "獲利率(%)")
    ]
    for col, (label, col_name) in zip(cols[:-1], headers_map):
        if col.button(get_header_label(label, col_name), key=f"btn_head_{col_name}_{key_suffix}"):
            update_sort(col_name)
            st.rerun()
            
    cols[-1].markdown("**管理**")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0px 0 10px 0; border-top: 2px solid #666;'>", unsafe_allow_html=True)

def display_stock_rows(df, currency_type):
    try:
        df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    except:
        df_sorted = df

    for index, row in df_sorted.iterrows():
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
        c2.write(f"{row['股數']:.3f}") 
        c3.write(f"{row['平均持有單價']:.2f}")
        c4.write(f"{price:.2f}")
        c5.write(fmt.format(cost))
        c6.write(fmt.format(val))
        c7.markdown(f":{color}[{fmt.format(prof)}]")
        c8.markdown(f":{color}[{roi:.2f}%]")
        if c9.button("🗑️", key=f"del_{symbol}"): remove_stock(symbol); st.rerun()
        
        st.markdown("<hr style='margin: 5px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)

def display_subtotal_row(df, currency_type):
    total_cost = df["總投入成本(原幣)"].sum()
    total_val = df["現值(原幣)"].sum()
    total_profit = df["獲利(原幣)"].sum()
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    
    st.markdown("<hr style='margin: 10px 0; border-top: 2px solid #666;'>", unsafe_allow_html=True)
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

# 修改 Tabs 定義，加入第三個 Tab
tab1, tab2, tab3 = st.tabs(["📊 庫存與資產配置", "🧠 AI 技術分析與建議", "⚖️ 投資組合分析與再平衡"])

df_record = load_data()

# 共同資料計算 (如果有的話)
if not df_record.empty:
    usd_rate = get_exchange_rate()
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    df_record['總投入成本(原幣)'] = df_record['股數'] * df_record['持有成本單價']
    
    portfolio = df_record.groupby(["股票代號", "幣別"]).agg({
        "股數": "sum",
        "總投入成本(原幣)": "sum"
    }).reset_index()
    portfolio["平均持有單價"] = portfolio["總投入成本(原幣)"] / portfolio["股數"]

# --- Tab 1: 庫存與資產配置 ---
with tab1:
    with st.sidebar:
        st.header("📝 新增投資")
        with st.form("add_stock_form"):
            symbol_input = st.text_input("股票代號", value="2330.TW").upper().strip()
            qty_input = st.number_input("股數", min_value=0.0, value=1000.0, step=0.001, format="%.3f")
            cost_input = st.number_input("單價 (原幣)", min_value=0.0, value=500.0)
            if st.form_submit_button("新增"):
                df = load_data()
                new_data = pd.DataFrame({"股票代號": [symbol_input], "股數": [qty_input], "持有成本單價": [cost_input]})
                df = pd.concat([df, new_data], ignore_index=True)
                save_data(df)
                st.success(f"已新增 {symbol_input}"); st.rerun()
        if st.button("🚨 清空所有"):
            if os.path.exists(DATA_FILE): os.remove(DATA_FILE); st.rerun()

    if df_record.empty:
        st.info("請先從側邊欄新增投資紀錄。")
    else:
        st.sidebar.markdown(f"--- \n 💱 匯率: **{usd_rate:.2f}**")
        
        unique_symbols = portfolio["股票代號"].tolist()
        with st.spinner('正在同步最新市場即時價格 (Fast Info)...'):
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

        # 總資產看板
        total_val = portfolio["現值(TWD)"].sum()
        total_cost = portfolio["總投入成本(TWD)"].sum()
        total_profit = portfolio["獲利(TWD)"].sum()
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 總資產 (TWD)", f"${total_val:,.0f}")
        col2.metric("💳 總投入成本 (TWD)", f"${total_cost:,.0f}")
        col3.metric("📈 總獲利", f"${total_profit:,.0f}", f"{roi:.2f}%")
        
        st.markdown("---")

        # 圖表區
        st.subheader("📊 資產分佈分析")
        col_pie1, col_pie2 = st.columns(2)
        
        with col_pie1:
            st.markdown("#### 🔹 資產類別佔比")
            st.write("") 
            st.write("") 

            df_pie_cat = portfolio.groupby("幣別")["現值(TWD)"].sum().reset_index()
            df_pie_cat["類別名稱"] = df_pie_cat["幣別"].map({"TWD": "台股 (TWD)", "USD": "美股 (USD)"})
            
            fig1 = px.pie(df_pie_cat, values="現值(TWD)", names="類別名稱", title=None, hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig1, use_container_width=True)

        with col_pie2:
            st.markdown("#### 🔹 個股權重分佈")
            filter_option = st.selectbox("選擇顯示範圍", ["全部 (ALL)", "台股 (TW)", "美股 (US)"], label_visibility="collapsed")
            
            if filter_option == "台股 (TW)": df_pie_filtered = portfolio[portfolio["幣別"] == "TWD"]
            elif filter_option == "美股 (US)": df_pie_filtered = portfolio[portfolio["幣別"] == "USD"]
            else: df_pie_filtered = portfolio

            if not df_pie_filtered.empty:
                fig2 = px.pie(df_pie_filtered, values="現值(TWD)", names="股票代號", title=None, hole=0.4)
                fig2.update_traces(textinfo='percent+label')
                st.plotly_chart(fig2, use_container_width=True)
            else: st.info(f"無 {filter_option} 資料")

        st.markdown("---")

        # 詳細庫存列表
        st.subheader("📦 詳細庫存列表")
        df_tw = portfolio[portfolio["幣別"] == "TWD"].copy()
        df_us = portfolio[portfolio["幣別"] == "USD"].copy()

        st.caption("🇹🇼 台股")
        if not df_tw.empty:
            display_headers("tw") 
            with st.container(height=300, border=False):
                display_stock_rows(df_tw, "TWD")
            display_subtotal_row(df_tw, "TWD")
        else: st.write("無持倉")

        st.write("") 

        st.caption("🇺🇸 美股")
        if not df_us.empty:
            display_headers("us") 
            with st.container(height=300, border=False):
                display_stock_rows(df_us, "USD")
            us_val, us_prof = display_subtotal_row(df_us, "USD")
            st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.9em;'>約 NT$ {us_val*usd_rate:,.0f} | 獲利 NT$ {us_prof*usd_rate:,.0f}</div>", unsafe_allow_html=True)
        else: st.write("無持倉")

# --- Tab 2: 技術分析 ---
with tab2:
    if df_record.empty:
        st.info("請先新增庫存股票。")
    else:
        st.subheader("🧠 持股健診與進出建議")
        stock_list = portfolio["股票代號"].tolist()
        selected_stock = st.selectbox("請選擇要分析的股票：", stock_list)

        if selected_stock:
            with st.spinner(f"分析中 {selected_stock}..."):
                result, error = analyze_stock_technical(selected_stock)
                if error: st.error(error)
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("目前價格", f"{result['current_price']:.2f}")
                    c2.metric("半年高 (壓力)", f"{result['high_6m']:.2f}")
                    c3.metric("半年低 (支撐)", f"{result['low_6m']:.2f}")
                    c4.metric("RSI 指標", f"{result['rsi']:.1f}")

                    st.divider()

                    st.subheader("💡 系統操作建議 (未來3個月)")
                    st.markdown(f"#### 趨勢： **{result['trend']}**")
                    
                    col_b, col_s = st.columns(2)
                    with col_b: st.info(f"**🟢 建議進場**: ${result['entry_target']:.2f} 附近\n\n(支撐位/均線回測)")
                    with col_s: st.warning(f"**🔴 建議停利**: ${result['exit_target']:.2f} 附近\n\n(前波壓力區)")
                    
                    st.success(f"**綜合點評**：:{result['advice_color']}[{result['advice']}]")
                    st.markdown("---")
                    
                    st.markdown("### 📊 週線走勢圖 (近半年)")
                    chart_data = result['history_df'][['Close']].copy()
                    chart_data['20週均線'] = chart_data['Close'].rolling(window=20).mean()
                    st.line_chart(chart_data)

# --- Tab 3: 投資組合分析與再平衡 (新增功能) ---
with tab3:
    if df_record.empty:
        st.info("請先新增投資紀錄，系統才能進行組合分析。")
    else:
        st.subheader("⚖️ 投資組合健檢與再平衡")
        st.markdown("透過數據分析您的投資組合風險，包含**資產配置**、**相關性分析**與**再平衡建議**。")
        
        # 分析按鈕 (避免每次切換頁面都重新抓取大量歷史資料)
        if st.button("🚀 啟動/更新 深度分析 (需抓取36個月資料)", type="primary"):
            with st.spinner("正在下載歷史股價並進行關聯性分析... (可能需要幾秒鐘)"):
                # 這裡需要傳入最新的 portfolio (已計算現值TWD)
                analysis_result, err = perform_portfolio_analysis(portfolio)
                
                if err:
                    st.error(f"分析失敗: {err}")
                else:
                    # 將結果存入 session_state 以便重繪時保留
                    st.session_state['analysis_result'] = analysis_result

        # 如果有分析結果就顯示
        if 'analysis_result' in st.session_state:
            res = st.session_state['analysis_result']
            
            st.divider()

            # 1. 投資組合分佈 (圓餅圖 - 直接複用)
            st.markdown("### 1️⃣ 投資組合權重分佈 (TWD計價)")
            fig_pie_all = px.pie(portfolio, values="現值(TWD)", names="股票代號", title="全投組資金佔比", hole=0.4)
            fig_pie_all.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie_all, use_container_width=True)

            st.divider()

            # 2. 相關係數分析 (Heatmap)
            st.markdown("### 2️⃣ 個股相關係數矩陣 (近36個月)")
            st.caption("相關係數越接近 **1 (深紅)** 代表兩者走勢越同步 (風險無法分散)；越接近 **-1 (深藍)** 代表走勢互補。")
            
            corr_matrix = res['corr_matrix']
            fig_heatmap = px.imshow(
                corr_matrix, 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale='RdBu_r', # 紅藍配色
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # 顯示波動率排行
            st.markdown("**📊 年化波動率 (風險值)**")
            vol_df = res['returns_std'].sort_values(ascending=False).to_frame(name="年化波動率(%)")
            st.dataframe(vol_df.style.format("{:.2f}%"), use_container_width=True)

            st.divider()

            # 3. 再平衡建議
            st.markdown("### 3️⃣ 優劣分析與再平衡建議")
            for suggestion in res['suggestions']:
                st.info(suggestion)
                
        else:
            st.write("👆 請點擊上方按鈕開始分析。")
