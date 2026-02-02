import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yfinance as yf
import numpy as np
from plotly.subplots import make_subplots
import back
import importlib
try:
    importlib.reload(back)
except:
    pass
from back import StructuralRiskDetector2026

# 페이지 설정
st.set_page_config(
    page_title="Structural Risk Monitor",
    page_icon="📊",
    layout="wide"
)

# 사이드바
st.sidebar.title("⚙️ 설정")
st.sidebar.markdown("---")

# 모델 초기화 (캐시)
# 모델 초기화 (캐시)
# 모델 초기화 (캐시)
@st.cache_resource
def load_detector_v48():
    """모델 로드"""
    try:
        # 1. Streamlit Secrets (Cloud 배포용)
        if 'FRED_API_KEY' in st.secrets:
            api_key = st.secrets['FRED_API_KEY']
        # 2. 로컬 파일 (개발용)
        else:
            with open('api.txt', 'r') as f:
                api_key = f.read().strip()
    except Exception as e:
        st.error(f"API 키 로드 실패: {e}. Streamlit Cloud Secrets에 'FRED_API_KEY'를 설정하거나 로컬에 'api.txt'가 있는지 확인하세요.")
        return None
        
    return StructuralRiskDetector2026(fred_api_key=api_key)

@st.cache_data(ttl=3600)
def load_data_v48(_detector):
    """데이터 로드 (모델 학습 제외)"""
    with st.spinner('데이터 로딩 중...'):
        # [AUTO] 매일 날짜 자동 갱신
        today = datetime.now().strftime('%Y-%m-%d')
        df = _detector.prepare_training_data(start_date='2002-01-01', end_date=today)
    return df

@st.cache_resource
def run_training_v48(_detector, df):
    """모델 학습 (별도 캐시)"""
    if df is None or df.empty:
        return _detector

    with st.spinner('모델 학습 및 백테스트 중...'):
        # [AUTO] 검증 구간 자동 설정 (최근 1년)
        # split_date = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        # [USER REQUEST] AI 상승장(2023)을 학습에 포함
        # 2023년 SVB 사태와 AI 랠리 초입을 배워야 2024-2026을 예측 가능
        _detector.train_model(df, split_date='2024-01-01')
    return _detector

# 모델 및 데이터 로드
detector = load_detector_v48()
if detector is None:
    st.stop()
df = load_data_v48(detector)

if df is None:
    st.error("데이터 로드 실패: 데이터를 가져올 수 없습니다. (소스 데이터 오류 또는 API 문제)")
    st.stop()

detector = run_training_v48(detector, df)

# [DEBUG] 데이터 로드 확인
# st.success(f"데이터 로드 완료 (Shape: {df.shape})")
# st.write("컬럼 목록:", df.columns.tolist())

# 사이드바 정보
st.sidebar.info(f"""
**모델 정보**
- 레이어: 7개
- 피처: 22개
- 학습 기간: 2002-2023
- 검증 기간: 2024-2026
""")

if st.sidebar.button("🔄 데이터 갱신"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Structural Risk Monitor")

# 페이지 네비게이션
page = st.sidebar.radio(
    "페이지 선택",
    ["🏠 Home - 현재 위험", "📈 Backtest 결과", "🎯 피처 신호", "🔬 모델 진단", "ℹ️ About"]
)

# ==========================================
# Page 1: Home - 현재 위험 평가
# ==========================================
if page == "🏠 Home - 현재 위험":
    st.title("🛡️ Structural Risk Engine (Dual-Core)")
    
    # [NEW] 탭 분리
    tab1, tab2 = st.tabs(["🚀 메인: 폭락 감지 모델", "🔒 서브: 사모신용(Private Credit)"])
    
    # ==========================================================================
    # [TAB 1] 기존 메인 대시보드 (AI 예측)
    # ==========================================================================
    with tab1:
        st.subheader("📉 Market Crash Probability (AI Model)")
        
        # 현재 평가
        current = detector.get_current_assessment(df)
        
        # 상단 메트릭
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_emoji = {"Normal": "🟢", "Elevated": "🟡", "High": "🔴"}
            st.metric(
                "위험 등급",
                current['risk_level'],
                delta=None,
                delta_color="inverse"
            )
            st.markdown(f"### {risk_emoji.get(current['risk_level'], '⚪')}")
        
        with col2:
            st.metric(
                "폭락 확률 (20일)",
                f"{current['probability']:.1%}"
            )
            # 게이지 차트
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current['probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if current['probability'] > 20 else "orange" if current['probability'] > 10 else "green"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 20], 'color': "yellow"},
                        {'range': [20, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': detector.threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_gauge, width="stretch")
        
        with col3:
            # [수정] 권장 주식 비중 -> 과열 압력 (HMM Strain)
            hmm_strain_val = df['hmm_strain'].iloc[-1] if 'hmm_strain' in df.columns else 0.0
            
            st.metric(
                "과열 압력 (HMM Strain)",
                f"{hmm_strain_val:.2f}",
                delta="주의" if hmm_strain_val > 1.0 else "안정",
                delta_color="inverse"
            )
            
            # 게이지 차트 (Strain)
            fig_strain = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hmm_strain_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 3]}, # 보통 Z-score 3 이상이면 극단적
                    'bar': {'color': "darkred" if hmm_strain_val > 1.5 else "orange" if hmm_strain_val > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1.5], 'color': "yellow"},
                        {'range': [1.5, 3], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.5 # Critical Threshold
                    }
                }
            ))
            fig_strain.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_strain, width="stretch")
        
        st.markdown("---")
        
        # 주요 신호 강도
        st.subheader("📡 현재 신호 강도 (Top 5)")
        current_features = df.drop('crash', axis=1).iloc[-1]
        
        # [수정] 모니터링 핵심 지표: private_credit, hmm_strain 제거 (요청사항)
        base_features = [
            'volatility', 'bond_stress', 'eco_surprise', 
            'fx_carry', 'net_liquidity' 
        ]
        
        feature_values = {feat: current_features[feat] for feat in base_features if feat in current_features}
        
        fig_bar = go.Figure([go.Bar(
            x=list(feature_values.keys()),
            y=list(feature_values.values()),
            marker_color=['red' if v > 1 else 'orange' if v > 0.5 else 'green' for v in feature_values.values()],
            text=[f"{v:.2f}" for v in feature_values.values()],
            textposition='auto'
        )])
        fig_bar.update_layout(
            title="신호 강도 (Z-score)",
            xaxis_title="Feature",
            yaxis_title="값",
            height=400
        )
        st.plotly_chart(fig_bar, width="stretch")

    # ==========================================================================
    # [TAB 2] 사모신용 전용 모니터링 페이지
    # ==========================================================================
    with tab2:
        st.subheader("☠️ Private Credit Stress Monitor")
        st.markdown("""
        **"보이지 않는 위험"을 감시합니다.** 사모대출 시장(TCPC)과 공모 하이일드 채권(HYG)의 괴리를 통해 구조적 위기 징후를 포착합니다.
        """)
        
        # 사모신용 지표 계산
        with st.spinner('사모신용 지표 계산 중...'):
            try:
                pc_indicators = detector.get_private_credit_indicators(start_date='2023-01-01')
                
                if not pc_indicators.empty:
                    # 최근 값 가져오기
                    current_discount = pc_indicators['discount_to_nav'].iloc[-1]
                    discount_5d = pc_indicators['discount_to_nav_5d'].iloc[-1]
                    current_spread = pc_indicators['yield_spread'].iloc[-1]
                    spread_50d = pc_indicators['yield_spread_50d'].iloc[-1]
                    tcpc_yield = pc_indicators['tcpc_div_yield'].iloc[-1]
                    hyg_yield = pc_indicators['hyg_div_yield'].iloc[-1]
                    
                    # ======================
                    # 1. 메트릭 카드 (상단)
                    # ======================
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Discount to NAV 메트릭
                        discount_delta = current_discount - discount_5d
                        discount_status = "↑ 할인 확대" if discount_delta > 0 else "↓ 할인 축소" if discount_delta < 0 else "→ 변동없음"
                        
                        st.metric(
                            "📊 Discount to NAV",
                            f"{current_discount:.2f}%",
                            delta=f"5일 평균: {discount_5d:.2f}%",
                            help="TCPC 주가가 순자산가치(NAV) 대비 얼마나 할인되어 거래되고 있는지 나타냅니다. 양수: 할인, 음수: 프리미엄"
                        )
                        
                        # 할인율 상태 표시
                        if current_discount > 15:
                            st.error("⚠️ 극심한 할인 (>15%): 유동성 위기 신호")
                        elif current_discount > 10:
                            st.warning("🔶 높은 할인 (>10%): 투자자 신뢰 하락")
                        elif current_discount > 5:
                            st.info("ℹ️ 보통 할인 (5-10%): 정상 범위")
                        else:
                            st.success("✅ 낮은 할인 (\u003c5%): 양호")
                    
                    with col2:
                        # Yield Spread 메트릭
                        spread_delta = current_spread - spread_50d
                        spread_status = "↑ 스프레드 확대" if spread_delta > 0 else "↓ 스프레드 축소" if spread_delta < 0 else "→ 변동없음"
                        
                        st.metric(
                            "💰 Yield Spread (TCPC - HYG)",
                            f"{current_spread:.2f}%",
                            delta=f"50일 평균: {spread_50d:.2f}%",
                            help="TCPC의 배당수익률이 HYG보다 얼마나 높은지 나타냅니다. 스프레드가 확대되면 사모대출 위험 프리미엄 증가"
                        )
                        
                        # 스프레드 상태 표시
                        if current_spread > 8:
                            st.error("⚠️ 극대 스프레드 (>8%): 사모시장 경색")
                        elif current_spread > 5:
                            st.warning("🔶 높은 스프레드 (>5%): 위험 프리미엄 증가")
                        elif current_spread > 2:
                            st.info("ℹ️ 보통 스프레드 (2-5%): 정상 범위")
                        else:
                            st.success("✅ 낮은 스프레드 (\u003c2%): 안정적")
                    
                    st.markdown("---")
                    
                    # ======================
                    # 2. 배당수익률 비교
                    # ======================
                    st.subheader("📈 배당수익률 비교 (Trailing 12M)")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("TCPC (사모)", f"{tcpc_yield:.2f}%")
                    col2.metric("HYG (공모)", f"{hyg_yield:.2f}%")
                    
                    st.markdown("---")
                    
                    # ======================
                    # 3. 차트 2개 (Discount to NAV, Yield Spread)
                    # ======================
                    
                    # 차트 1: Discount to NAV
                    st.subheader("📉 Discount to NAV 추이 (최근 1년)")
                    
                    fig_discount = go.Figure()
                    
                    # 실제 할인율
                    fig_discount.add_trace(go.Scatter(
                        x=pc_indicators.index,
                        y=pc_indicators['discount_to_nav'],
                        mode='lines',
                        name='Discount to NAV',
                        line=dict(color='steelblue', width=2)
                    ))
                    
                    # 5일 이동평균
                    fig_discount.add_trace(go.Scatter(
                        x=pc_indicators.index,
                        y=pc_indicators['discount_to_nav_5d'],
                        mode='lines',
                        name='5일 평균',
                        line=dict(color='orange', width=1, dash='dash')
                    ))
                    
                    # 위험 임계선
                    fig_discount.add_hline(y=0, line_dash="solid", line_color="gray", annotation_text="NAV (0%)")
                    fig_discount.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="주의 (10%)")
                    fig_discount.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="위험 (15%)")
                    
                    fig_discount.update_layout(
                        title='TCPC Discount to NAV (%)',
                        xaxis_title='날짜',
                        yaxis_title='할인율 (%)',
                        height=400,
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_discount, use_container_width=True)
                    
                    # 차트 2: Yield Spread
                    st.subheader("💹 Relative Yield Spread 추이 (최근 1년)")
                    
                    fig_spread = go.Figure()
                    
                    # 실제 스프레드
                    fig_spread.add_trace(go.Scatter(
                        x=pc_indicators.index,
                        y=pc_indicators['yield_spread'],
                        mode='lines',
                        name='Yield Spread',
                        line=dict(color='darkred', width=2)
                    ))
                    
                    # 50일 이동평균
                    fig_spread.add_trace(go.Scatter(
                        x=pc_indicators.index,
                        y=pc_indicators['yield_spread_50d'],
                        mode='lines',
                        name='50일 평균',
                        line=dict(color='coral', width=1, dash='dash')
                    ))
                    
                    # 위험 임계선
                    fig_spread.add_hline(y=2, line_dash="dash", line_color="gray", annotation_text="정상 (2%)")
                    fig_spread.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="주의 (5%)")
                    fig_spread.add_hline(y=8, line_dash="dash", line_color="red", annotation_text="위험 (8%)")
                    
                    fig_spread.update_layout(
                        title='TCPC - HYG Yield Spread (%)',
                        xaxis_title='날짜',
                        yaxis_title='스프레드 (%)',
                        height=400,
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_spread, use_container_width=True)
                    
                    # ======================
                    # 4. 해석 가이드
                    # ======================
                    st.info("""
                    **💡 해석 가이드:**
                    
                    **Discount to NAV (할인율)**
                    * **양수 (+)**: TCPC가 NAV보다 저렴하게 거래 (할인)
                    * **할인율 \u003e 10%**: 투자자들이 사모대출 자산의 가치를 의심 중
                    * **할인율 \u003e 15%**: 유동성 위기 또는 신용 사건 임박 가능성
                    
                    **Yield Spread (수익률 스프레드)**
                    * **스프레드 확대**: 사모대출의 위험 프리미엄 증가 (투자자가 더 높은 수익률 요구)
                    * **스프레드 \u003e 5%**: 사모시장이 공모시장보다 위험하다고 판단
                    * **스프레드 \u003e 8%**: 사모펀드 환매 중단, 신용 동결 가능성
                    
                    **위기 시나리오**  
                    이 두 지표가 동시에 악화되고 + 메인 탭의 'Net Liquidity'가 마를 때 = **즉시 탈출 신호**
                    """)
                    
                else:
                    st.error("사모신용 데이터를 계산할 수 없습니다. (TCPC 또는 HYG 데이터 부족)")
                    
            except Exception as e:
                st.error(f"사모신용 지표 계산 오류: {e}")
                import traceback
                st.code(traceback.format_exc())


# ==========================================
# Page 2: Backtest 결과
# ==========================================
elif page == "📈 Backtest 결과":
    st.title("📈 백테스트 성능 분석")
    st.markdown("---")
    
    # 성능 메트릭 (상단)
    if hasattr(detector, 'backtest_results'):
        results = detector.backtest_results
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC", f"{results.get('auc', 0):.3f}")
        col2.metric("Recall", f"{results.get('recall', 0)*100:.1f}%")
        col3.metric("Precision", f"{results.get('precision', 0)*100:.1f}%")
        col4.metric("F1 Score", f"{results.get('f1', 0):.3f}")
    else:
        st.warning("백테스트 결과가 없습니다. 모델을 먼저 학습하세요.")
    
    st.markdown("---")

    # [NEW] 종합 분석 차트 (Full History)
    st.subheader("🔍 종합 분석 (Full History)")
    
    if hasattr(detector, 'backtest_results') and 'X_full' in results:
        X_full = results['X_full']
        y_full = results['y_full']
        y_pred_proba_full = results['y_pred_proba_full']
        test_start_date = results.get('test_start_date')
        
        # SPY 데이터 로드
        with st.spinner("차트 데이터 준비 중..."):
            try:
                spy_data = yf.download('SPY', start=X_full.index[0], progress=False)
                if isinstance(spy_data.columns, pd.MultiIndex):
                    spy_data.columns = spy_data.columns.get_level_values(0)
                if spy_data.index.tz is not None:
                    spy_data.index = spy_data.index.tz_localize(None)
                spy_close = spy_data['Close'] if 'Close' in spy_data.columns else spy_data.iloc[:, 0]
                spy_close = spy_close.reindex(X_full.index, method='ffill') # Align dates
            except:
                spy_close = pd.Series(np.nan, index=X_full.index)

        # Plot 생성
        fig_full = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.04,
            subplot_titles=("SPY Price + Crash Zones", "Crash Probability", "Layer Signals (Z-scores)", "Synchronized Stress Duration"),
            row_heights=[0.3, 0.2, 0.25, 0.25]
        )
        
        # 1. SPY Price
        fig_full.add_trace(go.Scatter(x=spy_close.index, y=spy_close, mode='lines', name='SPY', line=dict(color='black')), row=1, col=1)
        
        # Crash Zones (Red Background)
        y_series = pd.Series(y_full, index=X_full.index)
        crash_starts = y_series[(y_series == 1) & (y_series.shift(1) != 1)].index
        crash_ends = y_series[(y_series == 1) & (y_series.shift(-1) != 1)].index
        
        if len(crash_starts) > 0 and len(crash_ends) > 0:
             if len(crash_ends) > 0 and len(crash_starts) > 0 and crash_ends[0] < crash_starts[0]: 
                 crash_ends = crash_ends[1:]
             if len(crash_starts) > len(crash_ends): 
                 crash_ends = crash_ends.append(pd.Index([X_full.index[-1]]))
             
             for start, end in zip(crash_starts, crash_ends):
                 fig_full.add_vrect(
                     x0=start, x1=end, 
                     fillcolor="red", opacity=0.3, layer="below", line_width=0, 
                     row=1, col=1
                 )

        # 2. Probability
        thr = results.get('threshold', 0.5)
        fig_full.add_trace(go.Scatter(x=X_full.index, y=y_pred_proba_full, mode='lines', name='Prob', line=dict(color='darkred', width=1)), row=2, col=1)
        fig_full.add_hline(y=thr, line_dash="dash", line_color="red", row=2, col=1, annotation_text=f"Threshold {thr:.3f}")
        
        # 3. Signals (Top 7 Layout)
        # [수정] 차트에 표시할 핵심 지표 리스트 업데이트 (Top 7)
        columns_to_plot = [
            'volatility',        # 공포 지수 (VIX)
            'bond_stress',       # 채권 발작 (MOVE)
            'eco_surprise',      # 실물 경기 충격
            'fx_carry',          # [NEW] 환율/자본 유출 (2025 핵심)
            'private_credit',    # [NEW] 사모신용 붕괴 (TCPC)
            'net_liquidity',     # [NEW] 연준 유동성
            'hmm_strain'         # [NEW] 과열 압력 게이지
        ]
        
        # 색상 팔레트 (7개)
        colors_7 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf']
        
        for col, color in zip(columns_to_plot, colors_7):
            if col in X_full.columns:
                fig_full.add_trace(go.Scatter(x=X_full.index, y=X_full[col], mode='lines', name=col, line=dict(width=1, color=color)), row=3, col=1)
        fig_full.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        # 4. Stress Duration
        if 'sync_stress_duration' in X_full.columns:
            fig_full.add_trace(go.Scatter(
                x=X_full.index, y=X_full['sync_stress_duration'], 
                mode='lines', name='Duration', 
                fill='tozeroy', line=dict(color='#ffbb78', width=1)
            ), row=4, col=1)

        # Test Start Line
        if test_start_date:
            for i in range(1, 5):
                fig_full.add_vline(x=test_start_date, line_dash="dash", line_color="blue", row=i, col=1)

        fig_full.update_layout(height=1000, title_text="Structural Risk Analysis (Full History)", showlegend=True)
        st.plotly_chart(fig_full, width="stretch")
    
    st.markdown("---")
    

    # 2열 레이아웃 생성
    col1, col2 = st.columns(2)
    
    # [Left] Confusion Matrix
    with col1:
        st.subheader("Confusion Matrix")
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            # Heatmap
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, 
                x=['Predicted Normal', 'Predicted Crash'],
                y=['Actual Normal', 'Actual Crash'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}", 
                textfont={"size": 16},
                showscale=False
            ))
            
            fig_cm.update_layout(
                yaxis=dict(autorange="reversed"), # To match standard matrix layout
                margin=dict(t=30, b=30),
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # [Right] 신호 발생 통계
    with col2:
        st.subheader("📅 연도별 신호 통계")
        if 'X_test' in results and 'y_test' in results and 'y_pred' in results:
            try:
                X_idx = results['X_test'].index
                y_test = results['y_test']
                y_pred = results['y_pred']
                
                # 데이터프레임 생성
                stats_df = pd.DataFrame({'Actual': y_test, 'Pred': y_pred}, index=X_idx)
                stats_df['Year'] = stats_df.index.year
                
                # 집계
                years = sorted(stats_df['Year'].unique())
                tp_list = []
                fp_list = []
                fn_list = []
                
                for y in years:
                    sub = stats_df[stats_df['Year'] == y]
                    tp_list.append(len(sub[(sub['Actual']==1) & (sub['Pred']==1)]))
                    fp_list.append(len(sub[(sub['Actual']==0) & (sub['Pred']==1)]))
                    fn_list.append(len(sub[(sub['Actual']==1) & (sub['Pred']==0)]))
                
                # 차트
                fig_stats = go.Figure()
                fig_stats.add_trace(go.Bar(name='TP (정확)', x=years, y=tp_list, marker_color='green'))
                fig_stats.add_trace(go.Bar(name='FP (과민)', x=years, y=fp_list, marker_color='orange'))
                fig_stats.add_trace(go.Bar(name='FN (놓침)', x=years, y=fn_list, marker_color='red'))
                
                fig_stats.update_layout(
                    barmode='stack', 
                    xaxis_title="연도", 
                    yaxis_title="건수",
                    margin=dict(t=30, b=30),
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_stats, use_container_width=True)
                
            except Exception as e:
                st.error(f"통계 차트 생성 오류: {e}")

# ==========================================
# Page 3: 피처 신호
# ==========================================
elif page == "🎯 피처 신호":
    st.title("🎯 피처 신호 분석")
    st.markdown("---")
    
    # 피처 선택
    # 피처 선택
    available_features = [
        # 1. Base Features
        'volatility', 'bond_stress', 'eco_surprise', 'momentum', 'liquidity', 'fx_carry', 'net_liquidity',
        # 2. HMM Features (Pressure Cooker)
        'hmm_overheated', 'hmm_strain', 'hmm_strain_vel', 'strain_x_drain',
        # 3. Context Amplifiers
        'context_bond_stress', 'context_liquidity_drain', 'context_momentum_crash',
        'context_fx_shock', 'context_vol_shock', 'context_private_credit'
    ]
    selected_features = st.multiselect(
        "분석할 피처 선택",
        available_features,
        default=['hmm_strain', 'context_fx_shock', 'context_private_credit']
    )
    
    if selected_features:
        for feat in selected_features:
            st.subheader(f"📈 {feat}")
            
            # 시계열 차트
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[feat],
                mode='lines',
                name=feat,
                line=dict(color='steelblue', width=2)
            ))
            
            # Crash 라벨 표시
            crash_dates = df[df['crash'] == 1].index
            for crash_date in crash_dates:
                fig.add_vline(x=crash_date, line_width=1, line_dash="dash", line_color="red", opacity=0.3)
            
            fig.update_layout(
                title=f"{feat} Signal Over Time",
                xaxis_title="Date",
                yaxis_title="값 (Z-score)",
                height=300,
                hovermode='x unified'
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("피처를 선택하세요")

# ==========================================
# Page 4: 모델 진단
# ==========================================
elif page == "🔬 모델 진단":
    st.title("🔬 모델 진단")
    st.markdown("---")
    
    # Feature Importance
    st.subheader("🎯 Feature Importance (Top 15)")
    
    if hasattr(detector, 'model') and detector.model is not None:
        importances = None
        
        # 1. Try to get pre-calculated importances from backtest_results (Preferred for Ensemble)
        if hasattr(detector, 'backtest_results') and 'importances' in detector.backtest_results:
            importances = detector.backtest_results['importances']
            
        # 2. Try to get from model directly (Single models like XGBoost/RF)
        elif hasattr(detector.model, 'feature_importances_'):
            X = df.drop('crash', axis=1)
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': detector.model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
        if importances is not None:
            fig_imp = go.Figure([go.Bar(
                x=importances['importance'],
                y=importances['feature'],
                orientation='h',
                marker_color='steelblue'
            )])
            fig_imp.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=500
            )
            st.plotly_chart(fig_imp, width="stretch")
            
            # 테이블
            st.dataframe(importances, width="stretch")
        else:
             st.info("현재 모델(Ensemble)은 Feature Importance를 직접 제공하지 않거나, 계산된 결과가 없습니다.")

    else:
        st.warning("모델이 학습되지 않았습니다.")

# ==========================================
# Page 5: About
# ==========================================
# ==========================================
# Page 5: About
# ==========================================
elif page == "ℹ️ About":
    st.header("🧠 구조적 위험 탐지 시스템 (Structural Risk Detector 2026)")
    st.markdown("### \"Sheikh Sadik (2024) 논문 기반 - 21일 조기경보 시스템\"")
    
    st.markdown("---")

    st.markdown("""
    ## 1. 시스템 아키텍처 개요

    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    INPUT LAYERS (원시 신호)                          │
    ├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────┤
    │ Layer 1  │ Layer 2  │ Layer 3  │ Layer 3.5│ Layer 3.8│ Layer 3.9   │
    │변동성구조│채권스트레스│경제서프라이즈│모멘텀지표│유동성미세구조│FX캐리리스크│
    └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────┬──────┘
         │          │          │          │          │            │
         ▼          ▼          ▼          ▼          ▼            ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              STRUCTURAL FEATURES (구조적 특성)                       │
    ├─────────────────┬─────────────────┬─────────────────────────────────┤
    │ Paper Features  │ Absorption Ratio│ HMM Regime Detection            │
    │ (Skew/Kurt/Corr)│ (시스템 리스크) │ (Normal/Fragile/Stress)         │
    └────────┬────────┴────────┬────────┴────────────┬────────────────────┘
             │                 │                     │
             ▼                 ▼                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              PRESSURE COOKER LOGIC (압력밥솥 로직)                   │
    │  HMM Strain (누적 압력) + Context Interaction Features               │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              PATH-DEPENDENT FEATURES (경로 의존 변수)                │
    │  Duration (지속기간) + Acceleration (가속도) + Sync Stress           │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              DOUBLE-FILTER FEATURE SELECTION                        │
    │  Point Biserial Correlation ∩ Mutual Information                    │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              ENSEMBLE MODEL (앙상블 모델)                            │
    │  Random Forest + XGBoost → Soft Voting                              │
    └─────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  21일 폭락 확률  │
                        │  (0% ~ 100%)    │
                        └─────────────────┘
    ```

    ---

    ## 2. Layer 1: 변동성 구조 (`get_volatility_structure`)

    **목적**: VIX 기반으로 시장의 공포와 불안정성을 측정

    ### 핵심 지표 4가지:

    | 지표 | 계산 방식 | 의미 |
    |------|----------|------|
    | **Term Ratio** | VIX(1개월) / VIX3M(3개월) | >1이면 백워데이션 (단기 공포 > 장기) |
    | **Backwardation** | Term Ratio > 1.0 여부 | 백워데이션 발생 시 1, 아니면 0 |
    | **Skew Stress** | (SKEW - 100) / 10 | 풋옵션 수요 증가 → 꼬리 위험 인식 |
    | **RV Regime** | 5일 RV / 60일 RV | 단기 변동성이 장기 대비 급등 여부 |

    ---

    ## 3. Layer 2: 채권 스트레스 (`get_bond_stress_divergence`)

    **목적**: 채권시장의 스트레스와 신용 위험을 측정

    ### 핵심 지표 4가지:

    | 지표 | 계산 방식 | 의미 |
    |------|----------|------|
    | **MOVE-VIX Divergence** | MOVE Z-score - VIX Z-score | 채권 변동성이 주식보다 먼저 반응 (선행지표) |
    | **SOFR Stress** | SOFR 3개월 평균 - 3개월 국채 | 단기 자금시장 경색 |
    | **HY Stress** | 하이일드 스프레드 Z-score | 정크본드 위험 프리미엄 |
    | **Inversion Stress** | -(10Y - 2Y).clip(upper=0) | 수익률 곡선 역전 (경기침체 신호) |

    ---

    ## 4. Layer 3: 경제 서프라이즈 (`get_economic_surprise`)

    **목적**: 거시경제 데이터의 "가속도"를 측정

    - **UNRATE**: 실업률
    - **CPIAUCSL**: CPI
    - **INDPRO**: 산업생산지수

    **의미**: 단순히 "나빠졌다"가 아니라 "나빠지는 속도가 빨라졌다"를 포착

    ---

    ## 5. Layer 3.5: 모멘텀 지표 (`get_momentum_indicators`)

    **목적**: 기술적 분석 신호로 과열/과매도 탐지

    | 지표 | 계산 방식 | 의미 |
    |------|----------|------|
    | **RSI Divergence** | 가격 신고점 + RSI 저고점 | 베어리시 다이버전스 (숨겨진 약세) |
    | **MACD Stress** | MACD 히스토그램 Z-score | 음수일수록 약세 모멘텀 |
    | **Price Deviation** | (Price - MA200) / MA200 | 장기 추세 대비 이탈 정도 |

    ---

    ## 6. Layer 3.8: 유동성 미세구조 (`get_liquidity_indicators`)

    **목적**: 무료 OHLCV 데이터로 시장 미세구조(유동성) 추정

    | 지표 | 계산 방식 | 의미 |
    |------|----------|------|
    | **Amihud Illiquidity** | |수익률| / 거래대금 | 같은 거래량에 가격이 많이 움직임 = 유동성 부족 |
    | **Corwin-Schultz Spread** | High/Low 기반 스프레드 추정 | 매수-매도 스프레드 (거래비용) |
    | **VVIX Divergence** | VVIX Z-score - VIX Z-score | "변동성의 변동성"이 먼저 튀는지 |

    ---

    ## 7. Layer 3.9: FX 캐리 리스크 (`get_fx_carry_risk`)

    **목적**: 환율 변동성으로 글로벌 충격 감지
    **의미**: 엔캐리 청산 같은 글로벌 충격 포착 (USD/JPY, DXY 변동성 측정)

    ---

    ## 8. Layer 3.95: 순유동성 (`get_net_liquidity`)

    **핵심 공식**: `Net Liquidity = Fed Balance Sheet - TGA - RRP`
    **해석**: 순유동성이 줄어들면 → 시장에 돈이 마름 → 위험

    ---

    ## 9. Paper Features (Sheikh Sadik 2024 정렬)

    ### 9.1 고차 모멘트 (`get_paper_features`)
    | 지표 | 의미 |
    |------|------|
    | **Skewness** | 음수 Skew = 왼쪽 꼬리 위험 (급락 가능성) |
    | **Kurtosis** | 높은 Kurtosis = 꼬리가 두꺼움 (극단적 사건) |
    | **Correlation** | 주식과 국채/VIX가 같이 움직이면(동조화) 시스템 위기 |

    ### 9.2 Absorption Ratio (`get_absorption_ratio`)
    - **PCA(주성분분석)**. 9개 섹터가 2개의 거대 요인에 의해 몇 %나 설명되는가? 
    - 높을수록 **동조화(위기 전조)**.

    ---

    ## 10. HMM 기반 국면 탐지 (`get_market_regime_hmm`)

    **목적**: 시장을 3가지 상태로 분류 (비선형적)

    | 상태 | 정의 | 특징 |
    |------|------|------|
    | **Normal (0)** | 평온 | 낮은 변동성, 정상 Skew/Corr |
    | **Fragile/Overheated (1)** | 과열/살얼음 | 낮은 Vol인데 Skew 악화, Corr 상승 (가장 위험!) |
    | **Stress (2)** | 스트레스 | 변동성 폭발 (폭락 진행 중) |

    ---

    ## 11. 압력밥솥 로직 (Pressure Cooker)

    **핵심 아이디어**: Fragile 상태에서 압력이 쌓이다가, 외부 충격이 오면 폭발
    
    ### Context Interaction Features:
    - **Bond Stress Trigger**: 채권 충격 × 누적 압력
    - **FX Shock Trigger**: 환율 충격 × 누적 압력

    ---

    ## 12. 전체 Feature 목록 (최종)

    ### 원시 신호 (Base)
    - Volatility, Bond Stress, Eco Suprise
    - Momentum, Liquidity, FX Carry, Net Liquidity

    ### 구조적/파생 변수 (Structural/Derived)
    - **HMM State**: Normal / Fragile / Stress
    - **Interpretation**: HMM Strain (누적 압력), Stress Duration (지속 기간)
    - **Crisis Context**: Context Bond/FX/Vol Features

    ### 논문 변수 (Paper)
    - Skewness (66, 252), Kurtosis
    - Absorption Ratio, SPY-TLT Correlation

    ---

    ## 13. 핵심 설계 철학 요약

    | 원칙 | 구현 |
    |------|------|
    | **선행성** | MOVE-VIX Divergence, VVIX, Absorption Ratio |
    | **구조적 위험** | HMM으로 "겉은 멀쩡한데 속이 썩는" 상태 포착 |
    | **경로 의존성** | Duration, Acceleration으로 "지속"과 "가속" 측정 |
    | **비선형 상호작용** | Context Features로 "압력 × 충격" 결합 |
    | **Recall 우선** | 10배 Class Weight, 85% Recall 목표 임계값 |
    | **Crisis Focus** | 2000, 2008, 2020년 위기에 샘플 가중치 3배 |

    이 시스템의 궁극적 목표는 **"놓치는 폭락을 최소화"**하는 것입니다.  
    Precision이 낮아져도(오경보가 많아도) **Recall을 높여서 실제 폭락을 절대 놓치지 않는 전략**입니다.
    """)
