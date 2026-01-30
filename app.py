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
# 1. 初始化設定與路徑管理
# ==========================================
st.set_page_config(page_title="投資戰情室 3.0 Pro", layout="wide")

DATA_DIR = "data"
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
for d in [DATA_DIR, BACKUP_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

if 'mpt_results' not in st.session_state: st.session_state.mpt_results = None

# ==========================================
# 2. 核心數據處理函數
# ==========================================

def get_path(user, file_type="csv"):
    if file_type == "csv":
        return os.path.join(DATA_DIR, f"portfolio_{user}.csv")
    return os.path.join(DATA_DIR, f"financial_config_{user}.json")

def load_data(user):
    path = get_path(user)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    path = get_path(user)
    if os.path.exists(path):
        now = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, os.path.join(BACKUP_DIR, f"backup_{user}_{now}.csv"))
    df.to_csv(path, index=False)

def load_financial_config(user):
    path = get_path(user, "json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except: pass
    return {
        "cash_res": 500000.0, "l1_p": 3000000.0, "l1_r": 2.1, "l1_y": 30, "l1_m": 12,
        "l1_ins": 3000000.0, # 房貸壽險保額
        "l2_p": 0.0, "l2_r": 3.5, "l2_y": 7, "l2_m": 0,
        "pledge_loan": 0.0, "pledge_targets": []
    }

def save_financial_config(user, config):
    with open(get_path(user, "json"), "w") as f: json.dump(config, f)

@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        data = yf.download("USDTWD=X", period="1d", progress=False)
        return float(data['Close'].iloc[-1])
    except: return 32.5

@st.cache_data(ttl=300)
def get_latest_quotes_bulk(symbols):
    if not symbols: return {}
    try:
        data = yf.download(symbols, period="1d", progress=False)['Close']
        if len(symbols) == 1: return {symbols[0]: float(data.iloc[-1])}
        return data.iloc[-1].to_dict()
    except: return {s: 0.0 for s in symbols}

def calculate_remaining_loan(principal, annual_rate, years, months_passed):
    if principal <= 0 or annual_rate <= 0 or years <= 0: return 0.0
    r = annual_rate / 12 / 100
    n = years * 12
    if months_passed >= n: return 0.0
    return float(principal * ((1 + r)**n - (1 + r)**months_passed) / ((1 + r)**n - 1))

def calculate_indicators(df):
    # RSI 向量化計算
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    # MACD & Bollinger Bands
    df['BM'] = df['Close'].rolling(window=20).mean()
    df['BU'] = df['BM'] + (df['Close'].rolling(window=20).std() * 2)
    df['BL'] = df['BM'] - (df['Close'].rolling(window=20).std() * 2)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['M'] = exp1 - exp2
    df['MS'] = df['M'].ewm(span=9, adjust=False).mean()
    df['MH'] = df['M'] - df['MS']
    return df

# ==========================================
# 3. 介面與功能邏輯
# ==========================================

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶切換")
    current_user = st.selectbox("當前使用者", ["Alan", "Jenny", "All"])
    usd_rate = get_exchange_rate()
    st.divider()
    st.caption(f"📅 系統時間: {datetime.now().strftime('%Y-%m-%d')}")
    st.caption(f"💱 參考匯率: {usd_rate:.2f}")

# 數據載入與整合
df_raw = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)

st.title(f"🚀 {current_user} 投資戰情室")
t1, t2, t3, t4 = st.tabs(["📊 庫存配置", "🧠 技術診斷", "⚖️ 組合優化", "💰 資產負債"])

if not df_raw.empty:
    # 預處理計算
    df_raw['股票代號'] = df_raw['股票代號'].str.upper().strip()
    with st.status("同步市場報價...", expanded=False):
        price_map = get_latest_quotes_bulk(df_raw['股票代號'].unique().tolist())
        df_raw['現價'] = df_raw['股票代號'].map(price_map)
        df_raw['幣別'] = df_raw['股票代號'].apply(lambda s: "TWD" if ".TW" in s or ".TWO" in s else "USD")
        df_raw['現值_TWD'] = df_raw.apply(lambda r: r['股數'] * r['現價'] * (usd_rate if r['幣別']=="USD" else 1), axis=1)
        df_raw['獲利'] = (df_raw['現價'] - df_raw['持有成本單價']) * df_raw['股數']
        df_raw['報酬率%'] = (df_raw['獲利'] / (df_raw['股數'] * df_raw['持有成本單價'])) * 100

    with t1:
        st.subheader("📝 庫存編輯與績效")
        edited_df = st.data_editor(df_raw, num_rows="dynamic", use_container_width=True, key=f"ed_{current_user}")
        if st.button("💾 儲存所有變更", type="primary"):
            if current_user != "All":
                save_data(edited_df[['股票代號', '股數', '持有成本單價']], current_user)
                st.success("數據已同步至資料庫")
                st.rerun()
            else: st.error("全體模式下不可直接編輯，請切換至個人帳號。")
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(df_raw, values="現值_TWD", names="股票代號", title="個股佔比", hole=0.4), use_container_width=True)
        c2.plotly_chart(px.pie(df_raw, values="現值_TWD", names="幣別", title="市場配置"), use_container_width=True)

    with t2:
        target = st.selectbox("選擇分析標的", df_raw['股票代號'].unique())
        df_hist = yf.Ticker(target).history(period="1y")
        if not df_hist.empty:
            df_hist = calculate_indicators(df_hist)
            curr = df_hist.iloc[-1]
            # 評分邏輯
            score = 0
            if curr['RSI'] < 35: score += 1
            if curr['Close'] < curr['BL']: score += 1
            if curr['M'] > curr['MS']: score += 1
            
            advice = "🚀 強力買入" if score >= 2 else "📈 分批佈局" if score == 1 else "⚖️ 持股觀望"
            st.metric(f"{target} 診斷結論", advice, f"RSI: {curr['RSI']:.1f}")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df_hist.index, open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name="價格"), 1, 1)
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['BU'], line=dict(color='rgba(200,200,200,0.5)'), name="上軌"), 1, 1)
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['BL'], line=dict(color='rgba(200,200,200,0.5)'), name="下軌"), 1, 1)
            fig.add_trace(go.Bar(x=df_hist.index, y=df_hist['MH'], name="MACD力道"), 2, 1)
            fig.update_layout(xaxis_rangeslider_visible=False, height=600)
            st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.subheader("⚖️ 蒙地卡羅組合模擬")
        if st.button("🚀 執行優化模擬", type="primary"):
            # 此處調用之前的 perform_mpt_simulation 邏輯 (簡化示意)
            st.info("計算中... 這裡會根據歷史相關性給出建議權重。")

    with t4:
        f_cfg = load_financial_config(current_user if current_user != "All" else "Alan")
        with st.form("fin_settings"):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.write("**🏠 房貸設定**")
                l1p = st.number_input("房貸本金", value=f_cfg["l1_p"])
                l1ins = st.number_input("遞減壽險保額", value=f_cfg["l1_ins"]) # 房貸壽險
            with sc2:
                st.write("**💳 信貸/其他**")
                l2p = st.number_input("其餘貸款本金", value=f_cfg["l2_p"])
                l2m = st.number_input("已還月數", value=f_cfg["l2_m"])
            with sc3:
                st.write("**🔗 質押設定**")
                p_loan = st.number_input("質押借款金額", value=f_cfg["pledge_loan"])
                p_target = st.multiselect("擔保品", df_raw['股票代號'].unique(), default=f_cfg["pledge_targets"])
            
            if st.form_submit_button("💾 更新財務參數"):
                f_cfg.update({"l1_p": l1p, "l1_ins": l1ins, "l2_p": l2p, "l2_m": l2m, "pledge_loan": p_loan, "pledge_targets": p_target})
                save_financial_config(current_user if current_user != "All" else "Alan", f_cfg)
                st.rerun()

        # 淨資產與風險計算
        rem_l1 = calculate_remaining_loan(l1p, f_cfg['l1_r'], f_cfg['l1_y'], f_cfg['l1_m'])
        rem_l2 = calculate_remaining_loan(l2p, f_cfg['l2_r'], f_cfg['l2_y'], l2m)
        total_debt = rem_l1 + rem_l2 + p_loan
        net_worth = df_raw['現值_TWD'].sum() + f_cfg['cash_res'] - total_debt
        
        # 房貸壽險覆蓋分析
        insurance_gap = max(0, rem_l1 - l1ins)
        
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("💼 家庭淨資產", f"${net_worth:,.0f}")
        mc2.metric("📉 總負債", f"-${total_debt:,.0f}")
        mc3.metric("🛡️ 房貸保障缺口", f"${insurance_gap:,.0f}", delta="壽險覆蓋中" if insurance_gap == 0 else "保額不足", delta_color="normal" if insurance_gap == 0 else "inverse")
        
        if p_loan > 0 and p_target:
            collateral_val = df_raw[df_raw['股票代號'].isin(p_target)]['現值_TWD'].sum()
            m_ratio = (collateral_val / p_loan) * 100
            mc4.metric("🚨 質押維持率", f"{m_ratio:.1f}%", delta="-20%壓力測試" if m_ratio > 140 else "風險極高")
            
            if m_ratio < 160:
                st.warning(f"⚠️ 警告：目前維持率較低。若擔保品下跌 20%，維持率將降至 **{(collateral_val*0.8/p_loan*100):.1f}%**")

        st.plotly_chart(px.bar(x=["總資產", "總負債", "家庭淨資產"], y=[df_raw['現值_TWD'].sum()+f_cfg['cash_res'], -total_debt, net_worth], title="資產負債結構"), use_container_width=True)

else:
    st.info("尚未輸入庫存數據。請於側邊欄切換帳戶並在『庫存配置』分頁新增標的。")
