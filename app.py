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
# 1. 初始化設定與路徑
# ==========================================
st.set_page_config(page_title="Alan & Jenny 投資戰情室", layout="wide")

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心功能函數 (資料、備份與行情)
# ==========================================

def manage_backups(user, max_backups=10):
    """保持備份資料夾整潔"""
    backups = sorted([
        os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR) 
        if f.startswith(f"backup_{user}_")
    ], key=os.path.getmtime)
    while len(backups) > max_backups:
        os.remove(backups.pop(0))

def create_backup(user):
    """存檔前自動備份"""
    source_path = f"portfolio_{user}.csv"
    if os.path.exists(source_path):
        now = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"backup_{user}_{now}.csv")
        shutil.copy2(source_path, backup_path)
        manage_backups(user)

def load_data(user):
    path = f"portfolio_{user}.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    create_backup(user)
    df.to_csv(f"portfolio_{user}.csv", index=False)

def remove_stock(symbol, user):
    df = load_data(user)
    df = df[df["股票代號"] != symbol]
    save_data(df, user)

@st.cache_data(ttl=3600)
def get_exchange_rate():
    """快取匯率數據，每小時更新一次"""
    try:
        ticker = yf.Ticker("USDTWD=X")
        rate = ticker.fast_info.last_price
        if rate is None or pd.isna(rate):
             rate = ticker.history(period="1d")['Close'].iloc[-1]
        return rate
    except: return 32.5

@st.cache_data(ttl=600)
def get_batch_prices(symbols):
    """批次抓取最新報價，快取 10 分鐘"""
    if not symbols: return {}
    try:
        # 使用 yf.download 批次抓取所有代號的最新 1 分鐘資料
        data = yf.download(symbols, period="1d", interval="1m", progress=False)['Close']
        if len(symbols) == 1:
            return {symbols[0]: data.iloc[-1]}
        return data.iloc[-1].to_dict()
    except:
        return {}

def identify_currency(symbol):
    return "TWD" if (".TW" in symbol or ".TWO" in symbol) else "USD"

# --- 技術分析邏輯 (RSI 優化為 EMA) ---
def calculate_rsi(series, period=14):
    """精確化 RSI：改用 Wilder's Smoothing (EMA)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 使用指數移動平均 (EMA)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def analyze_stock_technical(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", interval="1wk")
        if df.empty: return None, "無法獲取歷史資料"
        current_price = df['Close'].iloc[-1]
        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        rsi_curr = calculate_rsi(df['Close'], 14).iloc[-1]
        trend = "多頭排列 🐂" if current_price > ma_20 else "空頭/整理 🐻"
        advice = "過熱，分批獲利" if rsi_curr > 70 else "超賣，分批佈局" if rsi_curr < 30 else "趨勢持穩"
        color = "red" if rsi_curr > 70 else "green" if rsi_curr < 30 else "gray"
        return {"current_price": current_price, "rsi": rsi_curr, "trend": trend, "advice": advice, "advice_color": color, "history_df": df.tail(26)}, None
    except Exception as e: return None, str(e)

# ==========================================
# 3. MPT 數學模擬器邏輯
# ==========================================

def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "至少需要2支股票才能模擬。"
    try:
        data = yf.download(symbols, period="3y", interval="1d", auto_adjust=True)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=symbols[0])
        data = data.dropna(how='all').ffill().pct_change().dropna()
        
        mean_returns = data.mean() * 252
        cov_matrix = data.cov() * 252
        
        num_portfolios = 2000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(len(symbols))
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            results[0,i] = portfolio_return
            results[1,i] = portfolio_std
            results[2,i] = (portfolio_return - 0.02) / portfolio_std # Rf=2%
            
        max_sharpe_idx = np.argmax(results[2])
        min_vol_idx = np.argmin(results[1])
        
        current_weights_val = portfolio_df["現值(TWD)"].values
        current_weights = current_weights_val / np.sum(current_weights_val)
        
        comparison = pd.DataFrame({
            "股票代號": symbols,
            "目前權重 (%)": current_weights * 100,
            "回報最高 (Max Sharpe) (%)": weights_record[max_sharpe_idx] * 100,
            "波動最低 (Min Vol) (%)": weights_record[min_vol_idx] * 100
        })

        return {
            "sim_df": pd.DataFrame({'Return': results[0], 'Volatility': results[1], 'Sharpe': results[2]}),
            "comparison": comparison,
            "max_sharpe": (results[0, max_sharpe_idx], results[1, max_sharpe_idx]),
            "min_vol": (results[0, min_vol_idx], results[1, min_vol_idx]),
            "corr": data.corr()
        }, None
    except Exception as e: return None, str(e)

# ==========================================
# 4. 介面顯示組件
# ==========================================
COLS_RATIO = [1.3, 0.9, 1, 1, 1.3, 1.3, 1.3, 1, 0.6]

def display_headers(key_suffix, current_user):
    cols = st.columns(COLS_RATIO)
    headers = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本(原幣)"), ("現值", "現值(原幣)"), ("獲利", "獲利(原幣)"), ("報酬率%", "獲利率(%)")]
    for col, (label, col_name) in zip(cols[:-1], headers):
        arrow = "▲" if st.session_state.sort_asc and st.session_state.sort_col == col_name else "▼" if st.session_state.sort_col == col_name else ""
        if col.button(f"{label} {arrow}", key=f"h_{col_name}_{key_suffix}_{current_user}"):
            st.session_state.sort_asc = not st.session_state.sort_asc if st.session_state.sort_col == col_name else False
            st.session_state.sort_col = col_name
            st.rerun()
    cols[-1].write("管理")

def display_stock_rows(df, currency_type, current_user):
    df_sorted = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in df_sorted.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
        color = "red" if row["獲利(原幣)"] > 0 else "green"
        c1.write(f"**{row['股票代號']}**"); c2.write(f"{row['股數']:.2f}"); c3.write(f"{row['平均持有單價']:.2f}"); c4.write(f"{row['最新股價']:.2f}"); c5.write(fmt.format(row['總投入成本(原幣)'])); c6.write(fmt.format(row['現值(原幣)'])); c7.markdown(f":{color}[{fmt.format(row['獲利(原幣)'])}]"); c8.markdown(f":{color}[{row['獲利率(%)']:.2f}%]")
        if current_user == "All": c9.write("🔒")
        else:
            if c9.button("🗑️", key=f"del_{row['股票代號']}_{current_user}"):
                remove_stock(row['股票代號'], current_user); st.rerun()

def display_subtotal_row(df, currency_type, usd_rate):
    tc, tv, tp = df["總投入成本(原幣)"].sum(), df["現值(原幣)"].sum(), df["獲利(原幣)"].sum()
    roi = (tp / tc * 100) if tc > 0 else 0
    fmt = "{:,.0f}" if currency_type == "TWD" else "{:,.2f}"
    color = "red" if tp > 0 else "green"
    st.markdown("<hr style='margin: 5px 0; border-top: 2px solid #666;'>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
    c1.markdown(f"**🔹 {currency_type} 小計**"); c5.markdown(f"**{fmt.format(tc)}**"); c6.markdown(f"**{fmt.format(tv)}**"); c7.markdown(f":{color}[**{fmt.format(tp)}**]"); c8.markdown(f":{color}[**{roi:.2f}%**]")
    if currency_type == "USD":
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(COLS_RATIO)
        c1.markdown("<span style='color: gray; font-size: 0.9em;'>└ 換算台幣 (TWD)</span>", unsafe_allow_html=True)
        c5.markdown(f"<span style='color: gray; font-size: 0.85em;'>${(tc * usd_rate):,.0f}</span>", unsafe_allow_html=True)
        c6.markdown(f"<span style='color: gray; font-size: 0.85em;'>${(tv * usd_rate):,.0f}</span>", unsafe_allow_html=True)
        c7.markdown(f"<span style='color: gray; font-size: 0.85em;'>${(tp * usd_rate):,.0f}</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. 主程式邏輯與分頁
# ==========================================

if 'sort_col' not in st.session_state: st.session_state.sort_col = "獲利(原幣)"
if 'sort_asc' not in st.session_state: st.session_state.sort_asc = False

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    if current_user != "All":
        with st.form("add_form"):
            st.subheader(f"📝 新增 {current_user} 持股")
            s_in = st.text_input("代號 (如 2330.TW / NVDA)", "2330.TW").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, value=100.0); c_in = st.number_input("成本", min_value=0.0, value=600.0)
            if st.form_submit_button("執行新增"):
                df = load_data(current_user)
                save_data(pd.concat([df, pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])], ignore_index=True), current_user)
                st.rerun()

# --- 全局資料準備 ---
df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)

if not df_record.empty:
    usd_rate = get_exchange_rate()
    df_record['幣別'] = df_record['股票代號'].apply(identify_currency)
    
    # 彙整投資組合
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(
        lambda g: pd.Series({
            '股數': g['股數'].sum(), 
            '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()
        }), include_groups=False
    ).reset_index()
    
    # 批次更新價格
    price_map = get_batch_prices(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    portfolio = portfolio.dropna(subset=["最新股價"])
    
    # 獲利計算
    portfolio["總投入成本(原幣)"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值(原幣)"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利(原幣)"] = portfolio["現值(原幣)"] - portfolio["總投入成本(原幣)"]
    portfolio["獲利率(%)"] = (portfolio["獲利(原幣)"] / portfolio["總投入成本(原幣)"]) * 100
    portfolio["現值(TWD)"] = portfolio["現值(原幣)"] * portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)
    portfolio["獲利(TWD)"] = portfolio["獲利(原幣)"] * portfolio["幣別"].apply(lambda x: 1 if x == "TWD" else usd_rate)

st.title(f"📈 {current_user} 投資組合戰情室")
tab1, tab2, tab3 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)"])

with tab1:
    if df_record.empty: st.info("尚無數據。")
    else:
        # 頁面頂部功能列
        top_c1, top_c2 = st.columns([1, 4])
        with top_c1:
            if st.button("🔄 刷新最新報價", type="secondary"):
                st.cache_data.clear()
                st.rerun()
        with top_c2:
            st.caption(f"最後更新時間 (台北): {datetime.now(pytz.timezone('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S')}")

        t_val, t_prof = portfolio["現值(TWD)"].sum(), portfolio["獲利(TWD)"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}")
        c2.metric("📈 總獲利 (TWD)", f"${t_prof:,.0f}")
        c3.metric("📊 總報酬率", f"{(t_prof/(t_val-t_prof)*100):.2f}%" if t_val!=t_prof else "0%")
        c4.metric("💱 匯率", f"{usd_rate:.2f}")
        
        st.divider(); st.subheader("🎯 組合配置圖解")
        cc1, cc2 = st.columns(2)
        with cc1: st.plotly_chart(px.pie(portfolio.groupby("幣別")["現值(TWD)"].sum().reset_index(), values="現值(TWD)", names="幣別", title="市場佔比", hole=0.5), use_container_width=True)
        with cc2:
            v_opt = st.selectbox("配置視圖：", ["全部", "台股", "美股"], key="pv")
            pdf = portfolio[portfolio["幣別"] == "TWD"] if v_opt == "台股" else portfolio[portfolio["幣別"] == "USD"] if v_opt == "美股" else portfolio
            if not pdf.empty: st.plotly_chart(px.pie(pdf, values="現值(TWD)", names="股票代號", title=f"{v_opt}分佈", hole=0.5), use_container_width=True)
        
        st.divider()
        for l, cur in [("🇹🇼 台股列表", "TWD"), ("🇺🇸 美股列表", "USD")]:
            sub = portfolio[portfolio["幣別"] == cur]
            if not sub.empty:
                st.subheader(l); display_headers(cur.lower(), current_user); display_stock_rows(sub, cur, current_user); display_subtotal_row(sub, cur, usd_rate)

with tab2:
    if df_record.empty: st.info("無數據。")
    else:
        target = st.selectbox("分析標的：", portfolio["股票代號"].tolist())
        res, err = analyze_stock_technical(target)
        if err: st.error(err)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("現價", f"{res['current_price']:.2f}")
            c2.metric("RSI (EMA)", f"{res['rsi']:.1f}")
            c3.write(f"趨勢: {res['trend']}")
            st.success(f"建議：{res['advice']}")
            st.line_chart(res['history_df']['Close'])

with tab3:
    if df_record.empty: st.info("無數據。")
    else:
        st.subheader("⚖️ 現代投資組合理論 (MPT) 模擬引擎")
        st.markdown(r"目標：在給定風險下極大化回報，或在給定回報下極小化風險：$\min \sigma_p^2 = w^T \Sigma w$")
        
        if st.button("🚀 啟動數學模擬器", type="primary"):
            with st.spinner("模擬 2000 種權重組合中..."):
                data, err = perform_mpt_simulation(portfolio)
                if err: st.error(err)
                else:
                    st.write("#### 1️⃣ 效率前緣雲圖 (Efficient Frontier)")
                    fig = px.scatter(data['sim_df'], x='Volatility', y='Return', color='Sharpe', color_continuous_scale='Viridis', labels={'Volatility':'年化波動度','Return':'預期年化回報'})
                    fig.add_trace(go.Scatter(x=[data['max_sharpe'][1]], y=[data['max_sharpe'][0]], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Max Sharpe'))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("#### 2️⃣ 建議調整比例")
                    st.table(data['comparison'].set_index("股票代號").style.format("{:.2f}%"))
                    st.info("💡 **Max Sharpe**：每單位風險帶來的超額回報最高；**Min Vol**：整體組合震盪幅度最小。")
                    
                    st.write("#### 3️⃣ 相關性矩陣 (Correlation Matrix)")
                    st.plotly_chart(px.imshow(data['corr'], text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1), use_container_width=True)
