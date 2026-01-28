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
def load_detector_v42():
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
def load_data_v42(_detector):
    """데이터 로드 (모델 학습 제외)"""
    with st.spinner('데이터 로딩 중...'):
        # [AUTO] 매일 날짜 자동 갱신
        today = datetime.now().strftime('%Y-%m-%d')
        df = _detector.prepare_training_data(start_date='2002-01-01', end_date=today)
    return df

@st.cache_resource
def run_training_v42(_detector, df):
    """모델 학습 (별도 캐시)"""
    if df is None or df.empty:
        return _detector

    with st.spinner('모델 학습 및 백테스트 중...'):
        # [AUTO] 검증 구간 자동 설정 (최근 1년)
        # split_date = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        # 하지만 안정성을 위해 고정된 날짜 사용 권장 (2023-01-01)
        # [TUNING] Golden Ratio Tuning 시에는 더 긴 검증 구간이 필요할 수 있음
        _detector.train_model(df, split_date='2023-01-01')
    return _detector

# 모델 및 데이터 로드
detector = load_detector_v42()
if detector is None:
    st.stop()
df = load_data_v42(detector)

if df is None:
    st.error("데이터 로드 실패: 데이터를 가져올 수 없습니다. (소스 데이터 오류 또는 API 문제)")
    st.stop()

detector = run_training_v42(detector, df)

# [DEBUG] 데이터 로드 확인
# st.success(f"데이터 로드 완료 (Shape: {df.shape})")
# st.write("컬럼 목록:", df.columns.tolist())

# 사이드바 정보
st.sidebar.info(f"""
**모델 정보**
- 레이어: 7개
- 피처: 22개
- 학습 기간: 2002-2022
- 검증 기간: 2023-2026
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
    st.title("📊 실시간 위험 평가")
    st.markdown("---")
    
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
        st.metric(
            "권장 주식 비중",
            f"{current['equity_weight']:.0%}"
        )
        # 파이 차트
        fig_pie = go.Figure(data=[go.Pie(
            labels=['주식', '현금/채권'],
            values=[current['equity_weight'], 1 - current['equity_weight']],
            marker_colors=['steelblue', 'lightgray']
        )])
        fig_pie.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_pie, width="stretch")
    
    st.markdown("---")
    
    # 주요 신호 강도
    st.subheader("📡 현재 신호 강도")
    current_features = df.drop('crash', axis=1).iloc[-1]
    base_features = ['volatility', 'bond_stress', 'eco_surprise', 'momentum', 'liquidity', 'fx_carry', 'net_liquidity']
    
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
        
        # 3. Signals
        for col, color in zip(['volatility', 'bond_stress', 'eco_surprise'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
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
        'context_bond_stress', 'context_liquidity_drain', 'context_momentum_crash'
    ]
    selected_features = st.multiselect(
        "분석할 피처 선택",
        available_features,
        default=['hmm_strain', 'context_bond_stress', 'context_liquidity_drain']
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
        X = df.drop('crash', axis=1)
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': detector.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
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
        st.warning("모델이 학습되지 않았습니다.")

# ==========================================
# Page 5: About
# ==========================================
elif page == "ℹ️ About":
    st.title("ℹ️ Structural Risk Monitor")
    st.markdown("---")
    
    st.markdown("""
    ## 📌 모델 개요
    
    Structural Risk Detector 2026은 **7개 레이어**로 구성된 다차원 시장 위험 탐지 시스템입니다.
    
    ### 🧩 7개 Feature Layers
    
    1. **Volatility Structure** (VIX 구조)
       - VIX 수준 및 변화율
       - 역사적 분위수 대비 현재 위치
    
    2. **Bond Stress Divergence** (채권 스트레스)
       - 10년물-2년물 스프레드
       - 국채 vs SOFR 스프레드
    
    3. **Economic Surprise** (경제 서프라이즈)
       - Unemployment rate vs recession threshold
    
    4. **Momentum Indicators** (모멘텀)
       - RSI, MACD, Price vs MA200
    
    5. **Liquidity Microstructure** (미세구조 유동성)
       - Amihud Illiquidity
       - Corwin-Schultz Spread
       - VVIX Divergence
    
    6. **FX Carry Risk** (환율 캐리 위험)
       - USD/JPY 변동성
       - 엔화-주가 상관관계
    
    7. **Net Liquidity** (순유동성)
       - Fed Balance Sheet - TGA - RRP
       - 일일 유동성 변화율
    
    ### 📊 성능 (2023-2026 검증)
    - **AUC**: 0.403
    - **Recall**: 6% (1/18 crashes detected)
    - **조기경보 성공**: SVB (2023), Black Monday (2024), 3월 폭락 (2025)
    
    ### 🔗 데이터 소스
    - Yahoo Finance (SPY, VIX, VVIX, JPY=X)
    - FRED API (TGA, RRP, Fed BS, Economic data)
    """)
