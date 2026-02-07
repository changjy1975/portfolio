import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import shutil
from datetime import datetime
import pytz
import numpy as np

# ==========================================
# 1. 核心指標計算函數
# ==========================================

def calculate_indicators(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    e1, e2 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = e1 - e2
    df['MACD_S'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_H'] = df['MACD'] - df['MACD_S']
    l9, h9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
    rsv = (df['Close'] - l9) / (h9 - l9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

def get_filtered_signals(df):
    m_gold = (df['MACD'] > df['MACD_S']) & (df['MACD'].shift(1) <= df['MACD_S'].shift(1))
    m_dead = (df['MACD'] < df['MACD_S']) & (df['MACD'].shift(1) >= df['MACD_S'].shift(1))
    k_gold = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    k_dead = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))
    buy = ( (m_gold & (df['Close'] > df['MA20'])) | (k_gold & (df['K'] < 30) & (df['RSI'] < 45)) )
    sell = ( (m_dead & (df['Close'] < df['MA5'])) | (k_dead & (df['K'] > 75) & (df['RSI'] > 70)) | ((df['Close'].shift(1) > df['MA20']) & (df['Close'] < df['MA20'])) )
    return buy, sell

# ==========================================
# 2. MPT 引擎 - 修復強化版
# ==========================================

def perform_mpt_simulation(portfolio_df):
    symbols = portfolio_df["股票代號"].tolist()
    if len(symbols) < 2: return None, "至少需要 2 支標的才能進行優化。"
    try:
        # 下載 3 年資料以計算協方差
        raw_data = yf.download(symbols, period="3y", interval="1d", progress=False)
        
        # 處理 yfinance 可能回傳的多重索引 (MultiIndex)
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close']
        else:
            data = raw_data[['Close']] if 'Close' in raw_data.columns else raw_data
        
        # 關鍵修正：先計算報酬率，再處理不同市場的交易日缺口 (填補前值)
        data = data.ffill()
        returns = data.pct_change().dropna(how='all').fillna(0)
        
        if returns.empty: return None, "資料對齊後無有效數據。"

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        num_portfolios = 2000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            w = np.random.random(len(symbols))
            w /= np.sum(w)
            weights_record.append(w)
            p_ret = np.sum(w * mean_returns)
            p_std = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            results[0,i] = p_ret # 報酬
            results[1,i] = p_std # 風險
            results[2,i] = (p_ret - 0.02) / p_std # Sharpe (無風險利率 2%)
            
        max_idx = np.argmax(results[2])
        min_idx = np.argmin(results[1])
        
        comparison = pd.DataFrame({
            "股票代號": symbols,
            "目前權重 (%)": (portfolio_df["現值_TWD"] / portfolio_df["現值_TWD"].sum() * 100).values,
            "建議最優權重 (%)": weights_record[max_idx] * 100,
            "最低風險權重 (%)": weights_record[min_idx] * 100
        })
        
        return {
            "sim_df": pd.DataFrame({'Return': results[0], 'Volatility': results[1], 'Sharpe': results[2]}),
            "comparison": comparison,
            "max_sharpe": (results[0, max_idx], results[1, max_idx]),
            "corr": returns.corr()
        }, None
    except Exception as e:
        return None, f"MPT 計算失敗: {str(e)}"

# ==========================================
# 3. 介面呈現 (整合台美股小計)
# ==========================================
# ... (load_data, save_data, get_latest_quotes 保持與之前一致)

# ==========================================
# 4. 主程式頁面 (重點：MPT 分頁完整化)
# ==========================================
# ... (Tab 1, Tab 2 邏輯保持 V11.0 穩定版內容)

# 在 Tab 3 組合分析中：
with tab3:
    if not df_raw.empty:
        if st.button("🚀 啟動 MPT 優化模擬 (2000 次模擬計算)", type="primary"):
            with st.spinner("正在對齊跨國市場資料並計算效率前緣..."):
                res, err = perform_mpt_simulation(portfolio)
                if err:
                    st.error(err)
                else:
                    st.session_state.mpt_results = res
        
        if st.session_state.mpt_results:
            r = st.session_state.mpt_results
            col_left, col_right = st.columns([2, 1])
            with col_left:
                fig_mpt = px.scatter(r['sim_df'], x='Volatility', y='Return', color='Sharpe', 
                                     title="效率前緣雲圖 (風險 vs 報酬)",
                                     labels={'Volatility': '預期波動率 (風險)', 'Return': '預期年化報酬'})
                fig_mpt.add_trace(go.Scatter(x=[r['max_sharpe'][1]], y=[r['max_sharpe'][0]], 
                                             mode='markers', marker=dict(color='red', size=15, symbol='star'), name='最優夏普組合'))
                st.plotly_chart(fig_mpt, use_container_width=True)
            with col_right:
                st.write("#### ⚖️ 資產配置建議")
                st.dataframe(r['comparison'].set_index("股票代號").style.format("{:.2f}%"))
            
            st.divider()
            st.write("#### 🔗 持股相關性矩陣")
            st.plotly_chart(px.imshow(r['corr'], text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
    else:
        st.info("尚無持股資料可供模擬。")
