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
# 1. 初始化設定與狀態管理
# ==========================================
st.set_page_config(page_title="投資戰情室 V2", layout="wide")

# 初始化 Session State
for key, default in {
    'mpt_results': None,
    'sort_col': "獲利",
    'sort_asc': False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

BACKUP_DIR = "backups"
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

# ==========================================
# 2. 核心功能函數 (資料處理)
# ==========================================

def load_data(user):
    path = f"portfolio_{user}.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["股票代號", "股數", "持有成本單價"])

def save_data(df, user):
    source_path = f"portfolio_{user}.csv"
    if os.path.exists(source_path):
        now = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y%m%d_%H%M%S")
        shutil.copy2(source_path, os.path.join(BACKUP_DIR, f"backup_{user}_{now}.csv"))
    df.to_csv(source_path, index=False)

def load_financial_config(user):
    path = f"financial_config_{user}.json"
    default_config = {
        "cash_res": 500000.0,
        "l1_p": 3000000.0, "l1_r": 2.65, "l1_y": 30, "l1_m": 12,
        "l2_p": 0.0, "l2_r": 3.5, "l2_y": 7, "l2_m": 0,
        "pledge_loan": 0.0, "pledge_targets": []
    }
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return {**default_config, **json.load(f)}
        except: pass
    return default_config

def save_financial_config(user, config):
    path = f"financial_config_{user}.json"
    with open(path, "w") as f:
        json.dump(config, f)

@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        rate = yf.Ticker("USDTWD=X").fast_info.last_price
        return float(rate) if rate else 32.5
    except: return 32.5

@st.cache_data(ttl=300)
def get_latest_quotes(symbols):
    if not symbols: return {}
    quotes = {}
    try:
        # 使用批次抓取提升效率
        data = yf.download(symbols, period="1d", interval="1m", progress=False, group_by='ticker')
        for s in symbols:
            try:
                if len(symbols) == 1:
                    price = data['Close'].iloc[-1]
                else:
                    price = data[s]['Close'].iloc[-1]
                quotes[s] = float(price)
            except: quotes[s] = 0.0
    except: 
        return {s: 0.0 for s in symbols}
    return quotes

def process_portfolio_data(df_record, usd_rate):
    """封裝所有資產計算邏輯"""
    if df_record.empty: return pd.DataFrame()
    
    df_record['幣別'] = df_record['股票代號'].apply(lambda s: "TWD" if ".TW" in s or ".TWO" in s else "USD")
    portfolio = df_record.groupby(["股票代號", "幣別"]).apply(
        lambda g: pd.Series({
            '股數': g['股數'].sum(), 
            '平均持有單價': (g['股數'] * g['持有成本單價']).sum() / g['股數'].sum()
        }), include_groups=False
    ).reset_index()
    
    price_map = get_latest_quotes(portfolio["股票代號"].tolist())
    portfolio["最新股價"] = portfolio["股票代號"].map(price_map)
    portfolio["總投入成本"] = portfolio["股數"] * portfolio["平均持有單價"]
    portfolio["現值"] = portfolio["股數"] * portfolio["最新股價"]
    portfolio["獲利"] = portfolio["現值"] - portfolio["總投入成本"]
    portfolio["獲利率(%)"] = (portfolio["獲利"] / portfolio["總投入成本"]).replace([np.inf, -np.inf], 0) * 100
    portfolio["現值_TWD"] = portfolio.apply(lambda r: r["現值"] * (usd_rate if r["幣別"]=="USD" else 1), axis=1)
    
    return portfolio

# ==========================================
# 3. MPT 模擬引擎 (優化快取)
# ==========================================

@st.cache_data(ttl=3600)
def get_historical_data(symbols):
    return yf.download(symbols, period="3y", interval="1d", auto_adjust=True)['Close']

def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "標的不足（至少需要 2 檔）。"
    try:
        close = get_historical_data(tuple(symbols))
        if close.empty: return None, "無法取得歷史資料。"
        
        rets = close.ffill().pct_change().dropna()
        m_rets = rets.mean() * 252
        c_mat = rets.cov() * 252
        
        num_sim = 2000
        res = np.zeros((3, num_sim))
        w_rec = []
        
        for i in range(num_sim):
            w = np.random.random(len(symbols))
            w /= np.sum(w)
            w_rec.append(w)
            p_r = np.sum(w * m_rets)
            p_s = np.sqrt(np.dot(w.T, np.dot(c_mat, w)))
            res[0,i] = p_r
            res[1,i] = p_s
            res[2,i] = (p_r - 0.02) / p_s  # 無風險利率設為 2%
            
        idx = np.argmax(res[2])
        curr_val = portfolio_df["現值_TWD"].values
        curr_w = curr_val / curr_val.sum()
        
        comp = pd.DataFrame({
            "股票代號": symbols, 
            "目前權重 (%)": curr_w * 100, 
            "建議權重 (%)": w_rec[idx] * 100
        })
        return {
            "sim_df": pd.DataFrame({'Return': res[0], 'Volatility': res[1], 'Sharpe': res[2]}), 
            "comparison": comp, 
            "max_sharpe": (res[0, idx], res[1, idx]), 
            "corr": rets.corr()
        }, None
    except Exception as e: 
        return None, f"模擬出錯: {str(e)}"

# ==========================================
# 4. 介面表格組件
# ==========================================
COLS_RATIO = [1.2, 0.8, 1, 1, 1.2, 1.2, 1.2, 1, 0.6]

def display_market_table(df, title, currency, usd_rate, user):
    st.subheader(title)
    h_map = [("代號", "股票代號"), ("股數", "股數"), ("均價", "平均持有單價"), ("現價", "最新股價"), ("總成本", "總投入成本"), ("現值", "現值"), ("獲利", "獲利"), ("報酬率", "獲利率(%)")]
    h_cols = st.columns(COLS_RATIO)
    
    for i, (label, col_name) in enumerate(h_map):
        arrow = " ▲" if st.session_state.sort_col == col_name and st.session_state.sort_asc else " ▼" if st.session_state.sort_col == col_name else ""
        if h_cols[i].button(f"{label}{arrow}", key=f"h_{currency}_{col_name}_{user}"):
            if st.session_state.sort_col == col_name: 
                st.session_state.sort_asc = not st.session_state.sort_asc
            else: 
                st.session_state.sort_col, st.session_state.sort_asc = col_name, False
            st.rerun()
            
    h_cols[8].write("**管理**")
    
    sorted_df = df.sort_values(by=st.session_state.sort_col, ascending=st.session_state.sort_asc)
    for _, row in sorted_df.iterrows():
        r = st.columns(COLS_RATIO)
        fmt = "{:,.0f}" if currency == "TWD" else "{:,.2f}"
        clr = "red" if row["獲利"] > 0 else "green"
        
        r[0].write(f"**{row['股票代號']}**")
        r[1].write(f"{row['股數']:.2f}")
        r[2].write(f"{row['平均持有單價']:.2f}")
        r[3].write(f"{row['最新股價']:.2f}")
        r[4].write(fmt.format(row['總投入成本']))
        r[5].write(fmt.format(row['現值']))
        r[6].markdown(f":{clr}[{fmt.format(row['獲利'])}]")
        r[7].markdown(f":{clr}[{row['獲利率(%)']:.2f}%]")
        
        if r[8].button("🗑️", key=f"del_{row['股票代號']}_{user}"):
            full_data = load_data(user)
            save_data(full_data[full_data["股票代號"] != row['股票代號']], user)
            st.rerun()

# ==========================================
# 5. 主程式
# ==========================================

with st.sidebar:
    st.title("👨‍👩‍👧 帳戶管理")
    current_user = st.selectbox("切換使用者：", ["Alan", "Jenny", "All"])
    
    if current_user != "All":
        with st.form("add_form", clear_on_submit=True):
            s_in = st.text_input("股票代號 (如: 2330.TW, AAPL)").upper().strip()
            q_in = st.number_input("股數", min_value=0.0, step=1.0)
            c_in = st.number_input("持有成本單價", min_value=0.0, step=0.1)
            if st.form_submit_button("新增持股"):
                if s_in:
                    new_row = pd.DataFrame([{"股票代號":s_in,"股數":q_in,"持有成本單價":c_in}])
                    save_data(pd.concat([load_data(current_user), new_row], ignore_index=True), current_user)
                    st.rerun()

# 載入與處理資料
df_record = pd.concat([load_data("Alan"), load_data("Jenny")], ignore_index=True) if current_user == "All" else load_data(current_user)
usd_rate = get_exchange_rate()
portfolio = process_portfolio_data(df_record, usd_rate)

st.title(f"📈 {current_user} 投資戰情室")
tab1, tab2, tab3, tab4 = st.tabs(["📊 庫存配置", "🧠 技術健診", "⚖️ 組合分析 (MPT)", "💰 資產負債表"])

if not portfolio.empty:
    with tab1:
        if st.button("🔄 刷新報價"): 
            st.cache_data.clear()
            st.rerun()
            
        t_val = float(portfolio["現值_TWD"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 總資產 (TWD)", f"${t_val:,.0f}")
        c4.metric("💱 匯率", f"{usd_rate:.2f}")
        
        st.divider()
        pc1, pc2 = st.columns(2)
        with pc1: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="幣別", title="市場配置", hole=0.4), use_container_width=True)
        with pc2: st.plotly_chart(px.pie(portfolio, values="現值_TWD", names="股票代號", title="個股配置", hole=0.4), use_container_width=True)
        
        st.divider()
        tw_df = portfolio[portfolio["幣別"]=="TWD"]
        us_df = portfolio[portfolio["幣別"]=="USD"]
        if not tw_df.empty: display_market_table(tw_df, "🇹🇼 台股庫存", "TWD", usd_rate, current_user)
        if not us_df.empty: display_market_table(us_df, "🇺🇸 美股庫存", "USD", usd_rate, current_user)

    with tab2:
        target = st.selectbox("選擇診斷標的", portfolio["股票代號"].tolist())
        df_t = yf.Ticker(target).history(period="1y")
        if not df_t.empty:
            # 使用更穩健的指標計算（此處維持原邏輯，但建議未來封裝）
            from pandas import Series
            def get_indicators(s: Series):
                # RSI
                delta = s.diff()
                g, l = delta.clip(lower=0), -delta.clip(upper=0)
                ma_g = g.ewm(com=13, adjust=False).mean()
                ma_l = l.ewm(com=13, adjust=False).mean()
                rsi = 100 - (100 / (1 + ma_g / (ma_l + 1e-9)))
                # MACD
                e1, e2 = s.ewm(span=12).mean(), s.ewm(span=26).mean()
                m = e1 - e2
                sig = m.ewm(span=9).mean()
                # BB
                ma = s.rolling(20).mean()
                std = s.rolling(20).std()
                return rsi, (ma+2*std, ma, ma-2*std), (m, sig, m-sig)

            df_t['RSI'], (df_t['BU'], df_t['BM'], df_t['BL']), (df_t['M'], df_t['MS'], df_t['MH']) = get_indicators(df_t['Close'])
            
            curr = df_t.iloc[-1]
            score = 0
            reasons = []
            if curr['RSI'] < 35: score += 1; reasons.append("RSI 進入超賣區")
            if curr['Close'] < curr['BL']: score += 1; reasons.append("股價觸及布林下軌")
            if curr['M'] > curr['MS']: score += 1; reasons.append("MACD 黃金交叉")
            
            advice = "強力建議 🚀" if score >= 2 else "分批佈局 📈" if score == 1 else "持股觀望 ⚖️"
            st.subheader(f"🔍 {target} 技術診斷：{advice}")
            st.info("診斷參考：" + (" / ".join(reasons) if reasons else "目前無顯著訊號"))
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['Close'], name="股價"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['BU'], name="布林上軌", line=dict(dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t['BL'], name="布林下軌", line=dict(dash='dash')), row=1, col=1)
            fig.add_trace(go.Bar(x=df_t.index, y=df_t['MH'], name="MACD 柱狀體"), row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("⚖️ MPT 組合優化模擬 (基於近 3 年數據)")
        if st.button("🚀 執行模擬計算", type="primary"):
            with st.spinner("模擬計算中..."):
                res, err = perform_mpt_simulation(portfolio)
                if err: st.error(err)
                else: st.session_state.mpt_results = res
        
        if st.session_state.mpt_results:
            res = st.session_state.mpt_results
            sc1, sc2 = st.columns([2, 1])
            with sc1: st.plotly_chart(px.scatter(res['sim_df'], x='Volatility', y='Return', color='Sharpe', title="效率前緣雲圖"), use_container_width=True)
            with sc2: 
                st.write("#### 建議配置 (最高夏普比率)")
                st.dataframe(res['comparison'].set_index("股票代號").style.format("{:.2f}%"))
            st.divider()
            st.write("#### 標的相關性矩陣")
            st.plotly_chart(px.imshow(res['corr'], text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)

    with tab4:
        f_cfg = load_financial_config(current_user if current_user != "All" else "Alan")
        st.subheader("💰 家庭資產負債管理")
        
        with st.form("financial_form"):
            c_r = st.number_input("💵 現金預留 (TWD)", value=float(f_cfg["cash_res"]))
            st.divider()
            lc1, lc2 = st.columns(2)
            with lc1:
                st.write("**貸款 1 (房貸)**")
                l1p = st.number_input("本金 (L1)", value=float(f_cfg["l1_p"]))
                l1r = st.number_input("利率 (L1) %", value=float(f_cfg["l1_r"]))
                l1y = st.number_input("年限 (L1)", value=int(f_cfg["l1_y"]))
                l1m = st.number_input("已還月 (L1)", value=int(f_cfg["l1_m"]))
            with lc2:
                st.write("**貸款 2 (其他)**")
                l2p = st.number_input("本金 (L2)", value=float(f_cfg["l2_p"]))
                l2r = st.number_input("利率 (L2) %", value=float(f_cfg["l2_r"]))
                l2y = st.number_input("年限 (L2)", value=int(f_cfg["l2_y"]))
                l2m = st.number_input("已還月 (L2)", value=int(f_cfg["l2_m"]))
            
            st.divider()
            pl = st.number_input("質押借款金額 (TWD)", value=float(f_cfg["pledge_loan"]))
            pt = st.multiselect("擔保標的選擇", portfolio["股票代號"].tolist(), default=f_cfg["pledge_targets"])
            
            if st.form_submit_button("💾 儲存並更新"):
                if current_user != "All":
                    save_financial_config(current_user, {
                        "cash_res": c_r, "l1_p": l1p, "l1_r": l1r, "l1_y": l1y, "l1_m": l1m,
                        "l2_p": l2p, "l2_r": l2r, "l2_y": l2y, "l2_m": l2m,
                        "pledge_loan": pl, "pledge_targets": pt
                    })
                    st.success("財務資料已儲存！")
                    st.rerun()

        # 質押風險監控
        if pl > 0 and pt:
            st.divider()
            collateral_val = portfolio[portfolio["股票代號"].isin(pt)]["現值_TWD"].sum()
            m_ratio = (collateral_val / pl * 100) if pl > 0 else 0
            
            st.markdown("#### 📉 股票質押即時風險監控")
            m_clr = "normal" if m_ratio > 160 else "off" if m_ratio > 140 else "inverse"
            st.metric("🚨 即時維持率", f"{m_ratio:.2f}%", delta="門檻 130%", delta_color=m_clr)
            
            if len(pt) == 1:
                t_stock = pt[0]
                t_shares = portfolio[portfolio["股票代號"] == t_stock]["股數"].values[0]
                liq_price = (1.3 * pl) / t_shares
                st.error(f"🚩 **{t_stock} 斷頭警示價**：當股價跌破 **${liq_price:.2f}** 時將低於 130%。")

        # 淨資產摘要
        from math import pow
        def calc_rem(p, r, y, m):
            if p <= 0 or r <= 0 or y * 12 <= m: return 0
            rate = r / 12 / 100
            n = y * 12
            return p * ((pow(1 + rate, n) - pow(1 + rate, m)) / (pow(1 + rate, n) - 1))

        rem1, rem2 = calc_rem(l1p, l1r, l1y, l1m), calc_rem(l2p, l2r, l2y, l2m)
        t_debt = rem1 + rem2 + pl
        n_w = (portfolio["現值_TWD"].sum() + c_r) - t_debt
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("💼 家庭總資產", f"${(portfolio['現值_TWD'].sum()+c_r):,.0f}")
        mc2.metric("📉 剩餘總負債", f"-${t_debt:,.0f}", delta=f"L1+L2+質押", delta_color="inverse")
        mc3.metric("🏆 家庭淨資產", f"${n_w:,.0f}")

else:
    st.info("👋 歡迎！目前尚未發現持股。請先從左側邊欄新增標的（例如：台股輸入 `2330.TW`，美股輸入 `AAPL`）。")
