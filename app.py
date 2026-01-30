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
# 1. 初始化與檔案系統管理
# ==========================================
st.set_page_config(page_title="投資戰情室 3.0 Pro", layout="wide")

DATA_DIR = "data"
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
for d in [DATA_DIR, BACKUP_DIR]:
    if not os.path.exists(d): 
        os.makedirs(d)

if 'mpt_results' not in st.session_state: 
    st.session_state.mpt_results = None

# --- 檔案存取工具 ---
def get_path(user, file_type="csv"):
    if file_type == "csv": return os.path.join(DATA_DIR, f"portfolio_{user}.csv")
    return os.path.join(DATA_DIR, f"financial_config_{user}.json")

def load_data(user):
    path = get_path(user)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

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
        "l1_ins": 3000000.0, # 房貸壽險保額 (根據您的個人動態)
        "l2_p": 0.0, "l2_r": 3.5, "l2_y": 7, "l2_m": 0,
        "pledge_loan": 0.0, "pledge_targets": []
    }

def save_financial_config(user, config):
    with open(get_path(user, "json"), "w") as f: 
        json.dump(config, f)

# ==========================================
# 2. 核心計算模組 (優化性能)
# ==========================================
@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        data = yf.download("USDTWD=X", period="1d", progress=False)
        return float(data['Close'].iloc[-1])
    except: return 32.5

@st.cache_data(ttl=300)
def get_latest_quotes_bulk(symbols):
    if not symbols: return {}
    valid_symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
    if not valid_symbols: return {}
    try:
        data = yf.download(valid_symbols, period="1d", progress=False)['Close']
        if len(valid_symbols) == 1: 
            return {valid_symbols[0]: float(data.iloc[-1])}
        return data.iloc[-1].to_dict()
    except: return {s: 0.0 for s in valid_symbols}

def calculate_remaining_loan(principal, annual_rate, years, months_passed):
    if principal <= 0 or annual_rate <= 0 or years <= 0: return 0.0
    r = annual_rate / 12 / 100
    n = years * 12
    if months_passed >= n: return 0.0
    return float(principal * ((1 + r)**n - (1 + r)**months_passed) / ((1 + r)**n - 1))

def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['BM'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BU'], df['BL'] = df['BM'] + (std * 2), df['BM'] - (std * 2)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['M'] = exp1 - exp2
    df['MS'] = df['M'].ewm(span=9, adjust=False).mean()
    df['MH'] = df['M'] - df['MS']
    return df

def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "標的不足（至少需 2 檔）。"
    try:
        data = yf.download(symbols, period="2y", progress=False)['Close']
        rets = data.pct_change().dropna()
        m_rets = rets.mean() * 252; c_mat = rets.cov() * 252
        num_ports = 2000
        weights = np.random.random((num_ports, len(symbols)))
        weights = (weights.T / np.sum(weights, axis=1)).T
        p_rets = np.dot(weights, m_rets)
        p_vols = np.sqrt(np.diagonal(np.dot(np.dot(weights, c_mat), weights.T)))
        sharpe = (p_rets - 0.02) / p_vols
        idx = np.argmax(sharpe)
        curr_val = portfolio_df["現值_TWD"].values
        curr_w = curr_val / curr_val.sum()
        comp = pd.DataFrame({"股票代號": symbols, "目前權重 (%)": curr_w * 100, "建議權重 (%)": weights[idx] * 100})
        return {"sim_df": pd.DataFrame({'Return': p_rets, 'Volatility': p_vols, 'Sharpe': sharpe}), "comparison": comp, "corr": rets.corr()}, None
    except Exception as e: return None, str(e)

# ==========================================
# 3. 主介面邏輯
# ==========================================
with st.sidebar:
    st.title("👨‍👩‍👧 帳戶切換")
    current_user = st.selectbox("當前使用者", ["Alan", "Jenny", "All"])
    usd_rate = get_exchange_rate()
    st.divider()
    st.caption(f"📅 系統日期: {datetime.now().strftime('%Y-%m-%d')}")
    st.caption(f"💱 當前匯率: {usd_rate:.2f}")
    st.info("💡 提醒：輸入台股代號請加 .TW (如 2330.TW)。")

# 數據讀取
df_raw = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)

st.title(f"🚀 {current_user} 投資戰情室")
t1, t2, t3, t4 = st.tabs(["📊 庫存配置", "🧠 技術診斷", "⚖️ 組合優化", "💰 資產負債"])

# --- TAB 1: 庫存配置 (核心修復：隨時可輸入) ---
with t1:
    st.subheader("📝 庫存編輯器")
    edited_df = st.data_editor(
        df_raw, 
        num_rows="dynamic", 
        use_container_width=True, 
        key=f"ed_{current_user}",
        column_config={
            "股票代號": st.column_config.TextColumn("代號 (e.g. 2330.TW, AAPL)"),
            "股數": st.column_config.NumberColumn("持股數量", min_value=0, format="%.2f"),
            "持有成本單價": st.column_config.NumberColumn("成本", min_value=0, format="%.2f")
        }
    )
    
    if st.button("💾 儲存並同步行情", type="primary"):
        if current_user != "All":
            save_data(edited_df.dropna(subset=['股票代號']), current_user)
            st.success("儲存成功！")
            st.rerun()
        else: st.error("『All』模式僅供檢視，請切換帳戶編輯。")

    if not df_raw.empty:
        st.divider()
        with st.status("獲取即時報價...", expanded=False):
            symbols = df_raw['股票代號'].unique().tolist()
            price_map = get_latest_quotes_bulk(symbols)
            df_display = df_raw.copy()
            df_display['現價'] = df_display['股票代號'].map(price_map)
            df_display['幣別'] = df_display['股票代號'].apply(lambda s: "TWD" if ".TW" in str(s) or ".TWO" in str(s) else "USD")
            df_display['現值_TWD'] = df_display.apply(lambda r: r['股數'] * r['現價'] * (usd_rate if r['幣別']=="USD" else 1) if pd.notnull(r['現價']) else 0, axis=1)
            df_display['獲利'] = (df_display['現價'] - df_display['持有成本單價']) * df_display['股數']
            df_display['報酬率(%)'] = (df_display['獲利'] / (df_display['持有成本單價'] * df_display['股數'])) * 100
        
        m1, m2, m3 = st.columns(3)
        total_val = df_display['現值_TWD'].sum()
        m1.metric("💰 總資產現值", f"${total_val:,.0f} TWD")
        m2.metric("📈 總盈虧", f"${df_display['獲利'].sum() * (usd_rate if 'USD' in df_display['幣別'].values else 1):,.0f}", 
                  f"{(df_display['獲利'].sum() / (df_display['持有成本單價']*df_display['股數']).sum()*100):.2f}%")
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(df_display, values="現值_TWD", names="股票代號", title="投資組合分佈", hole=0.4), use_container_width=True)
        st.dataframe(df_display.style.format({"現價":"{:.2f}","現值_TWD":"{:,.0f}","報酬率(%)":"{:.2f}%"}), use_container_width=True)
    else:
        st.info("👆 請在上方表格輸入股票代號（如 2330.TW）、股數與成本，然後按下儲存。")

# --- TAB 2: 技術診斷 ---
with t2:
    if not df_raw.empty:
        target = st.selectbox("選擇分析標的", df_raw['股票代號'].unique())
        with st.spinner("載入歷史數據..."):
            df_h = yf.Ticker(target).history(period="1y")
            if not df_h.empty:
                df_h = calculate_indicators(df_h)
                curr = df_h.iloc[-1]
                st.subheader(f"🔍 {target} 技術健診")
                col_i1, col_i2, col_i3 = st.columns(3)
                col_i1.metric("RSI (14)", f"{curr['RSI']:.2f}", "超賣" if curr['RSI']<30 else "超買" if curr['RSI']>70 else "正常")
                col_i2.metric("MACD 強度", f"{curr['MH']:.2f}", "多頭" if curr['MH']>0 else "空頭")
                col_i3.metric("價格位置", f"{curr['Close']:.2f}", f"距下軌 {((curr['Close']-curr['BL'])/curr['Close']*100):.1f}%")

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df_h.index, open=df_h['Open'], high=df_h['High'], low=df_h['Low'], close=df_h['Close'], name="K線"), 1, 1)
                fig.add_trace(go.Scatter(x=df_h.index, y=df_h['BU'], line=dict(color='rgba(150,150,150,0.5)', dash='dot'), name="上軌"), 1, 1)
                fig.add_trace(go.Scatter(x=df_h.index, y=df_h['BL'], line=dict(color='rgba(150,150,150,0.5)', dash='dot'), name="下軌"), 1, 1)
                fig.add_trace(go.Bar(x=df_h.index, y=df_h['MH'], name="MACD力道"), 2, 1)
                fig.update_layout(xaxis_rangeslider_visible=False, height=600)
                st.plotly_chart(fig, use_container_width=True)
    else: st.warning("請先在第一頁輸入資料。")

# --- TAB 3: MPT 優化 ---
with t3:
    if not df_raw.empty and len(df_raw['股票代號'].unique()) >= 2:
        if st.button("🚀 啟動蒙地卡羅優化模擬", type="primary"):
            res, err = perform_mpt_simulation(df_display)
            if err: st.error(err)
            else: st.session_state.mpt_results = res
        
        if st.session_state.mpt_results:
            res = st.session_state.mpt_results
            sc1, sc2 = st.columns([2, 1])
            sc1.plotly_chart(px.scatter(res['sim_df'], x='Volatility', y='Return', color='Sharpe', title="效率前緣分佈"), use_container_width=True)
            sc2.write("#### 權重建議")
            sc2.dataframe(res['comparison'].set_index("股票代號").style.format("{:.2f}%"))
            st.plotly_chart(px.imshow(res['corr'], text_auto=".2f", title="標的相關性矩陣"), use_container_width=True)
    else: st.info("至少需要兩檔股票才能進行組合分析。")

# --- TAB 4: 資產負債表 (包含房貸遞減壽險與質押監控) ---
with t4:
    f_cfg = load_financial_config(current_user if current_user != "All" else "Alan")
    st.subheader("🏦 家庭資產負債管理")
    with st.form("fin_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**房貸 (L1)**")
            l1p = st.number_input("房貸原始本金", value=f_cfg["l1_p"])
            l1ins = st.number_input("遞減壽險保額", value=f_cfg["l1_ins"], help="您在 2025-12-26 加保的額度")
            l1m = st.number_input("房貸已還月數", value=f_cfg["l1_m"])
        with c2:
            st.write("**信貸/其餘 (L2)**")
            l2p = st.number_input("其餘貸款本金", value=f_cfg["l2_p"])
            l2m = st.number_input("其餘已還月數", value=f_cfg["l2_m"])
        with c3:
            st.write("**股票質押**")
            pl = st.number_input("質押借款金額", value=f_cfg["pledge_loan"])
            pt = st.multiselect("擔保品選擇", df_raw['股票代號'].unique(), default=f_cfg["pledge_targets"])
        
        if st.form_submit_button("💾 儲存財務參數"):
            f_cfg.update({"l1_p":l1p, "l1_ins":l1ins, "l1_m":l1m, "l2_p":l2p, "l2_m":l2m, "pledge_loan":pl, "pledge_targets":pt})
            save_financial_config(current_user if current_user != "All" else "Alan", f_cfg)
            st.rerun()

    # 計算結果
    rem1 = calculate_remaining_loan(l1p, f_cfg['l1_r'], f_cfg['l1_y'], l1m)
    rem2 = calculate_remaining_loan(l2p, f_cfg['l2_r'], f_cfg['l2_y'], l2m)
    total_debt = rem1 + rem2 + pl
    net_worth = (total_val if not df_raw.empty else 0) + f_cfg['cash_res'] - total_debt
    
    # 風險看板
    k1, k2, k3 = st.columns(3)
    k1.metric("🏆 家庭總淨資產", f"${net_worth:,.0f}")
    
    # 房貸壽險缺口分析 (個人化功能)
    gap = max(0, rem1 - l1ins)
    k2.metric("🛡️ 房貸保障缺口", f"${gap:,.0f}", delta="保障充足" if gap==0 else "保障不足", delta_color="normal" if gap==0 else "inverse")
    
    # 質押維持率
    if pl > 0 and pt:
        collat_val = df_display[df_display['股票代號'].isin(pt)]['現值_TWD'].sum()
        ratio = (collat_val / pl) * 100
        k3.metric("🚨 質押維持率", f"{ratio:.1f}%", delta="-20% 壓力預警" if ratio > 160 else "⚠️ 補人頭風險")
        if ratio < 160:
            st.error(f"🚩 壓力測試：若擔保品下跌 20%，維持率將降至 **{(collat_val*0.8/pl*100):.1f}%** (門檻 130%)")

    st.plotly_chart(px.bar(x=["股票資產", "預留現金", "剩餘負債"], y=[total_val if not df_raw.empty else 0, f_cfg['cash_res'], -total_debt], color=["資產","資產","負債"], title="資產負債結構"), use_container_width=True)
