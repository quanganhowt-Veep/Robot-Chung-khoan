import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# CẤU HÌNH TRANG
st.set_page_config(page_title="ROBOT V9.0 - FIBONACCI & PATTERNS", layout="wide")

# CSS
st.markdown("""
    <style>
    .stButton>button {background-color: #4B0082; color: white; font-weight: bold; border-radius: 8px; height: 50px; width: 100%;}
    </style>
""", unsafe_allow_html=True)

# DANH MỤC
DANH_MUC = {
    "VN30": ["ACB.VN", "BCM.VN", "BID.VN", "BVH.VN", "CTG.VN", "FPT.VN", "GAS.VN", "GVR.VN", "HDB.VN", "HPG.VN", "MBB.VN", "MSN.VN", "MWG.VN", "PLX.VN", "POW.VN", "SAB.VN", "SHB.VN", "SSB.VN", "SSI.VN", "STB.VN", "TCB.VN", "TPB.VN", "VCB.VN", "VHM.VN", "VIB.VN", "VIC.VN", "VJC.VN", "VNM.VN", "VPB.VN", "VRE.VN"],
    "Bất Động Sản": ["DIG.VN", "CEO.VN", "DXG.VN", "NVL.VN", "PDR.VN", "KBC.VN", "NLG.VN", "VHM.VN", "VIC.VN", "VRE.VN", "HQC.VN", "ITA.VN", "HDG.VN"],
    "Chứng Khoán": ["SSI.VN", "VND.VN", "VCI.VN", "HCM.VN", "SHS.VN", "MBS.VN", "FTS.VN", "BSI.VN", "CTS.VN", "AGR.VN", "VIX.VN"],
    "Ngân Hàng": ["VCB.VN", "BID.VN", "CTG.VN", "TCB.VN", "VPB.VN", "MBB.VN", "ACB.VN", "STB.VN", "HDB.VN", "VIB.VN", "SHB.VN", "LPB.VN", "OCB.VN", "MSB.VN"],
    "Thép": ["HPG.VN", "HSG.VN", "NKG.VN", "TLH.VN", "VGS.VN"],
    "Dầu Khí": ["PVD.VN", "PVS.VN", "BSR.VN", "OIL.VN", "PLX.VN", "GAS.VN", "PVT.VN"],
    "Công Nghệ": ["FPT.VN", "CMG.VN", "ELC.VN", "LCG.VN", "ITD.VN", "VGI.VN", "CTR.VN"],
    "HNX": ["SHS.VN", "CEO.VN", "PVS.VN", "IDC.VN", "MBS.VN", "TNG.VN", "VCS.VN", "HUT.VN", "NVB.VN"],
    "UPCOM": ["BSR.VN", "VGI.VN", "VEA.VN", "QNS.VN", "OIL.VN", "MSR.VN", "ACV.VN", "SIP.VN", "LTG.VN"]
}

# HÀM LOGIC VSA
def phan_tich_vsa(vol_now, vol_avg, change):
    ratio = vol_now / vol_avg if vol_avg > 0 else 0
    tin_hieu = "-"
    if ratio >= 2.0:
        if change > 0.5: tin_hieu = "🦈 CÁ MẬP GOM"
        elif change < -0.5: tin_hieu = "🏃 XẢ HÀNG MẠNH"
    elif ratio < 0.6: tin_hieu = "💤 Cạn Cung"
    return ratio, tin_hieu

# HÀM NHẬN DIỆN MÔ HÌNH NẾN (Nến Nhật)
def soi_nen_nhat(df):
    # Tính thân nến và bóng nến
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    
    # 1. HAMMER (Búa) - Đảo chiều Tăng
    # Bóng dưới dài gấp 2 lần thân, bóng trên nhỏ
    df['Hammer'] = np.where((df['Lower_Shadow'] > 2 * df['Body']) & 
                            (df['Upper_Shadow'] < 0.5 * df['Body']), 1, 0)
    
    # 2. SHOOTING STAR (Sao đổi ngôi) - Đảo chiều Giảm
    # Bóng trên dài gấp 2 lần thân, bóng dưới nhỏ
    df['Shooting_Star'] = np.where((df['Upper_Shadow'] > 2 * df['Body']) & 
                                   (df['Lower_Shadow'] < 0.5 * df['Body']), 1, 0)
    return df

# HÀM VẼ FIBONACCI
def ve_fibonacci(fig, df):
    # Lấy đỉnh và đáy trong kỳ
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price
    
    # Các mức Fibo quan trọng
    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    colors = ['gray', 'red', 'orange', 'green', 'blue', 'red', 'gray']
    
    for i, level in enumerate(levels):
        price_level = max_price - (diff * level)
        fig.add_hline(y=price_level, line_dash="dash", line_width=1, line_color=colors[i], 
                      annotation_text=f"Fibo {level:.3f} ({price_level:,.0f})", 
                      annotation_position="top right", row=1, col=1)

# HÀM VẼ BIỂU ĐỒ CHÍNH
def ve_bieu_do(df, symbol, show_bb, show_ma, show_ichi, show_fibo, show_pattern):
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Signal_MACD'] = np.where(df['MACD_12_26_9'] > df['MACDs_12_26_9'], 1, 0)
    df['Crossover'] = df['Signal_MACD'].diff()
    buy_signals = df[df['Crossover'] == 1]
    sell_signals = df[df['Crossover'] == -1]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2], subplot_titles=(f'{symbol}', 'Volume (VSA)', 'MACD'))

    # 1. GIÁ
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Giá'), row=1, col=1)

    # FIBONACCI
    if show_fibo:
        ve_fibonacci(fig, df)

    # MÔ HÌNH NẾN
    if show_pattern:
        df = soi_nen_nhat(df)
        hammer = df[df['Hammer'] == 1]
        shooting = df[df['Shooting_Star'] == 1]
        # Vẽ nốt Hammer (Màu Vàng)
        fig.add_trace(go.Scatter(x=hammer.index, y=hammer['Low']*0.99, mode='markers+text', marker_symbol='circle-open', marker_color='gold', marker_size=10, text="🔨", textposition="bottom center", name='Hammer'), row=1, col=1)
        # Vẽ nốt Shooting Star (Màu Tím)
        fig.add_trace(go.Scatter(x=shooting.index, y=shooting['High']*1.01, mode='markers+text', marker_symbol='circle-open', marker_color='purple', marker_size=10, text="🌠", textposition="top center", name='Shooting Star'), row=1, col=1)

    # ICHIMOKU
    if show_ichi:
        try:
            ichi = df.ta.ichimoku(tenkan=9, kijun=26, senkou=52)[0]
            col_span_a = [c for c in ichi.columns if c.startswith('ISA')][0]
            col_span_b = [c for c in ichi.columns if c.startswith('ISB')][0]
            col_tenkan = [c for c in ichi.columns if c.startswith('ITS')][0]
            col_kijun = [c for c in ichi.columns if c.startswith('IKS')][0]

            fig.add_trace(go.Scatter(x=df.index, y=ichi[col_tenkan], line=dict(color='#FF5252', width=1), name='Tenkan'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ichi[col_kijun], line=dict(color='#2196F3', width=1), name='Kijun'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ichi[col_span_a], line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ichi[col_span_b], line=dict(color='rgba(0,0,0,0)'), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name='Mây Kumo'), row=1, col=1)
        except: pass

    # BOLLINGER
    if show_bb:
        try:
            bb = df.ta.bbands(length=20, std=2)
            if bb is not None and not bb.empty:
                col_upper = [c for c in bb.columns if c.startswith('BBU')][0]
                col_lower = [c for c in bb.columns if c.startswith('BBL')][0]
                fig.add_trace(go.Scatter(x=df.index, y=bb[col_upper], line=dict(color='gray', width=1, dash='dot'), name='BB Trên'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=bb[col_lower], line=dict(color='gray', width=1, dash='dot'), name='BB Dưới', fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
        except: pass

    # MA
    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), line=dict(color='purple', width=1), name='MA50'), row=1, col=1)

    # MŨI TÊN MACD
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low']*0.98, mode='markers', marker_symbol='triangle-up', marker_color='#00CC00', marker_size=12, name='MACD MUA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High']*1.02, mode='markers', marker_symbol='triangle-down', marker_color='#FF0000', marker_size=12, name='MACD BÁN'), row=1, col=1)

    # 2. VOLUME
    colors_vol = []
    for i in range(len(df)):
        vol = df['Volume'].iloc[i]
        ma_vol = df['Vol_MA20'].iloc[i]
        close = df['Close'].iloc[i]
        open_p = df['Open'].iloc[i]
        if vol > 2.0 * ma_vol:
            if close >= open_p: colors_vol.append('#9400D3')
            else: colors_vol.append('#FF8C00')
        else:
            if close >= open_p: colors_vol.append('#26a69a')
            else: colors_vol.append('#ef5350')
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vol, name='Volume'), row=2, col=1)

    # 3. MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], line=dict(color='blue'), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], line=dict(color='orange'), name='Signal'), row=3, col=1)
    colors_hist = ['green' if v >= 0 else 'red' for v in df['MACDh_12_26_9']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], marker_color=colors_hist, name='Hist'), row=3, col=1)
    
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# UI
st.sidebar.title("🚀 ROBOT V9.0 (FIBO + NẾN)")
mode = st.sidebar.radio("CHẾ ĐỘ:", ["🔍 SOI CHI TIẾT", "🌊 QUÉT SÓNG"])
st.sidebar.markdown("---")
show_fibo = st.sidebar.checkbox("Fibonacci Retracement", value=True)
show_pattern = st.sidebar.checkbox("Soi Nến Nhật (Búa/Sao)", value=True)
show_ichi = st.sidebar.checkbox("Ichimoku Cloud", value=False)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
show_ma = st.sidebar.checkbox("MA 20/50", value=False)

if mode == "🔍 SOI CHI TIẾT":
    c1, c2, c3 = st.columns([1,1,3])
    with c1: ma = st.text_input("Mã:", "VIC.VN").upper()
    with c2: time = st.selectbox("TG:", ["6mo", "1y", "2y"], index=1)
    with c3: 
        st.write("")
        st.write("")
        btn = st.button("PHÂN TÍCH NGAY")
    
    if btn:
        try:
            df = yf.download(ma, period=time, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if not df.empty:
                df.ta.macd(append=True)
                df.ta.rsi(append=True)
                
                # Metrics
                gia = df['Close'].iloc[-1]
                chg = df['Close'].pct_change().iloc[-1]*100
                vol_now = df['Volume'].iloc[-1]
                vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
                ratio, vsa_txt = phan_tich_vsa(vol_now, vol_avg, chg)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("💰 Giá", f"{gia:,.0f}", f"{chg:.2f}%")
                m2.metric("🌊 VSA", f"x{ratio:.1f}", vsa_txt)
                m3.metric("📊 RSI", f"{df['RSI_14'].iloc[-1]:.1f}")
                
                # Vẽ biểu đồ với Fibo và Pattern
                st.plotly_chart(ve_bieu_do(df, ma, show_bb, show_ma, show_ichi, show_fibo, show_pattern), use_container_width=True)
                
                st.subheader("📋 Bảng Giá Lịch Sử")
                st.dataframe(df.sort_index(ascending=False).head(10), use_container_width=True)
            else: st.error("Lỗi mã hoặc không có dữ liệu")
        except Exception as e: st.error(f"Lỗi hệ thống: {e}")

else:
    nganh = st.selectbox("Chọn Ngành:", list(DANH_MUC.keys()))
    if st.button(f"QUÉT {nganh}"):
        res = []
        bar = st.progress(0)
        lst = DANH_MUC[nganh]
        for i, m in enumerate(lst):
            try:
                d = yf.download(m, period="6mo", auto_adjust=True, progress=False)
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                if not d.empty and len(d)>30:
                    vn = d['Volume'].iloc[-1]
                    va = d['Volume'].rolling(20).mean().iloc[-1]
                    ch = d['Close'].pct_change().iloc[-1]*100
                    ra, vt = phan_tich_vsa(vn, va, ch)
                    d.ta.rsi(append=True)
                    dar = "✅ BREAK" if d['Close'].iloc[-1] > d['High'].rolling(20).max().iloc[-2] else "-"
                    res.append({"Mã": m, "Giá": d['Close'].iloc[-1], "%": round(ch,2), "RSI": round(d['RSI_14'].iloc[-1],1), "VSA": vt, "Darvas": dar})
            except: pass
            bar.progress((i+1)/len(lst))
        
        def stl(v):
            if "GOM" in str(v): return 'background-color: #00FF00; color: black; font-weight: bold'
            if "XẢ" in str(v): return 'background-color: #FF4500; color: white; font-weight: bold'
            if "BREAK" in str(v): return 'color: blue; font-weight: bold'
            return ''
        st.dataframe(pd.DataFrame(res).style.applymap(stl), use_container_width=True, height=600)
