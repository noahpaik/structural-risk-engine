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
        **"보이지 않는 위험"을 감시합니다.** 이 지표는 AI 학습에는 빠져있지만, **구조적 위기의 '트리거'**가 될 수 있으므로 별도 관찰이 필요합니다.
        """)
        
        # 1. 최신 상태 표시
        if 'private_credit' in df.columns:
            curr_pc_stress = df['private_credit'].iloc[-1]
            curr_pc_context = df['context_private_credit'].iloc[-1]
            
            col1, col2 = st.columns(2)
            col1.metric("Private Credit Stress (Z-score)", f"{curr_pc_stress:.2f}", 
                        delta="위험" if curr_pc_stress > 1.0 else "안정", delta_color="inverse")
            col2.metric("Weighted Impact (Context)", f"{curr_pc_context:.2f}")
            
            # 2. 사모신용 전용 차트 그리기
            # TCPC(사모) vs HYG(공모) 괴리율 시각화
            fig_pc = go.Figure()
            
            # 메인: 사모신용 스트레스 지수
            fig_pc.add_trace(go.Scatter(
                x=df.index, y=df['private_credit'],
                mode='lines', name='Private Credit Stress',
                line=dict(color='red', width=2)
            ))
            
            # 보조: 시장 위험 임계선
            fig_pc.add_hline(y=1.0, line_dash="dash", line_color="orange", annotation_text="Warning (1.0)")
            fig_pc.add_hline(y=2.0, line_dash="dash", line_color="darkred", annotation_text="Critical (2.0)")
            
            fig_pc.update_layout(
                title='TCPC(Private) vs HYG(Public) Divergence Stress',
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig_pc, use_container_width=True)
            
            st.info("""
            **💡 해석 가이드:**
            * **스트레스 > 1.0:** 사모 대출 자산(TCPC)의 가격이 공모 채권(HYG)보다 비정상적으로 하락 중.
            * **스트레스 > 2.0:** 유동성 위기 징후. 사모펀드 환매 중단 가능성 염두.
            * 이 지표가 튀어 오를 때, 메인 탭의 '현금 유동성(Net Liquidity)'이 마르고 있다면 **즉시 탈출**하십시오.
            """)
        else:
            st.error("사모신용 데이터가 계산되지 않았습니다. back.py를 확인하세요.")

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
elif page == "ℹ️ About":
    st.header("🧠 Structural Risk Engine (Sheikh Sadik 2024 기반)")
    st.markdown("### \"시장은 가장 고요할 때 붕괴한다.\"")
    
    st.info("""
    **이 모델은 단순한 기술적 분석 도구가 아닙니다.**  
    금융시장의 **'구조적 취약성(Structural Fragility)'**을 탐지하여, 
    모두가 안심하는 상승장 속에서 **시스템 붕괴의 전조(Precursors)**를 찾아내는 데 특화되어 있습니다.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 🌪️ 불안정성의 역설 (Paradox of Instability)")
        st.write("""
        > *"Stability breeds Instability." - Hyman Minsky*
        
        기존 모델들은 **변동성(Volatility)**이 낮으면 '안전하다'고 판단합니다. 
        하지만 이 모델은 다릅니다. **낮은 변동성이 오래 지속되면**, 
        투자자들은 과도한 레버리지를 쓰게 되고, 시스템은 오히려 **'살얼음판(Fragile)'**이 됩니다.
        
        이 모델은 **Absorption Ratio(흡수 비율)**와 **Skewness(비대칭성)**를 통해 
        이 숨겨진 위험을 감지합니다.
        """)
        
        st.subheader("3. 🧲 흡수 비율 (Absorption Ratio)")
        st.write("""
        시장이 정상일 때, 주식/채권/원자재는 각자 따로 움직입니다.
        하지만 위기가 다가오면, 모든 자산이 **하나의 거대한 요인(공포)**에 의해 
        **동조화(Synchronization)**되기 시작합니다.
        
        이 비율이 급등하면, 사소한 충격에도 시스템 전체가 무너질 수 있습니다.
        """)
    
    with col2:
        st.subheader("2. 🏗️ 구조적 HMM (Structural HMM)")
        st.write("""
        단순히 가격이 오르고 내리는 것을 넘어, **'시장 구조'**를 3단계로 진단합니다.
        
        - **🟢 Normal (안정)**: 건전한 상승장. 편안하게 투자하세요.
        - **🟠 Fragile (살얼음판)**: 가격은 오르지만, 내부적으로 **Skew가 깨지고(하락 징후)** 자산 간 **동조화**가 심해진 상태. **가장 위험한 구간**입니다.
        - **🔴 Stress (붕괴)**: 이미 위기가 터진 상태. 변동성이 폭발합니다.
        """)
        
        st.subheader("4. 🔬 이중 필터 변수 선택 (Double Filter)")
        st.write("""
        수만 개의 데이터 중 '진짜 신호'만 걸러내기 위해 두 번 거릅니다.
        1.  **Point Biserial**: 선형적인 상관관계가 뚜렷한가?
        2.  **Mutual Information**: 비선형적인 정보량(Entropy)이 높은가?
        
        이 둘의 **교집합(Intersection)**에 해당하는 상위 50% 지표만 사용하여, 
        가짜 신호(Noise)에 속지 않습니다.
        """)
        
    st.markdown("---")
    
    st.header("🛠️ 상세 피처 명세서 (Data & Logic)")
    st.markdown("""
    이 모델이 사용하는 주요 피처들의 데이터 원천과 계산 로직입니다.
    
    | Layer / Feature | 입력 데이터 (Data Source) | 계산 로직 및 의도 (Logic & Rationale) |
    | :--- | :--- | :--- |
    | **1. Volatility** | `^VIX` (CBOE Volatility Index) | **공포 지수**. 252일 이동평균 대비 현재 수준(Z-score) 측정. 높을수록 시장 공포 극대화. |
    | **2. Bond Stress** | `^TNX` (10Y), `SHY` (2Y Proxy) | **장단기 금리차**. 수익률 곡선 역전(Inversion) 후 급격한 스티프닝(Steepening) 탐지. |
    | **3. Macro (Eco)** | `UNRATE` (실업률), `RECPRO` (침체확률) | **샴의 법칙(Sahm Rule)** 변형. 실업률 이동평균이 급격히 상승하는 구간 포착. |
    | **4. Momentum** | `SPY` (S&P 500 Price) | **RSI & MA Divergence**. 가격이 200일 이동평균선보다 얼마나 과도하게 벌어졌는지 측정. |
    | **5. Liquidity** | `SPY` (OHLCV High/Low/Vol) | **Amihud & Corwin-Schultz**. 거래량 대비 가격 변화폭. 수치가 튀면 "팔 사람은 많은데 살 사람이 없는" 상태. |
    | **6. FX Carry** | `JPY=X` (엔/달러 환율) | **캐리 트레이드 청산**. 엔화의 변동성이 급증하면 글로벌 자금 회수(Margin Call) 신호. |
    | **7. Net Liquidity** | `WALCL` (Fed 자산), `TGA`, `RRP` | **연준 순유동성**. `Fed 자산 - (재무부 계좌 + 역레포)`. 실제 시장에 풀려있는 달러 총량 측정. |
    | **8. Paper Features** | `SPY`, `TLT` (20Y Treasury) | **Skewness**: 수익률 분포가 왼쪽으로 찌그러짐(급락 위험). <br> **Correlation**: 주식과 국채가 같이 떨어지면(양의 상관관계) 시스템 위기. |
    | **9. Absorption Ratio** | 9개 섹터 ETF (`XLK`, `XLF` 등) | **PCA(주성분분석)**. 9개 섹터가 2개의 요인에 의해 몇 %나 설명되는가? 높을수록 **동조화(위기 전조)**. |
    | **10. Structural HMM** | `Vol`, `Skew`, `Corr`, `Absorb` | **비지도 학습**. 단순히 가격만 보는 게 아니라, 위 4가지 구조적 지표를 종합해 **'Fragile State'** 판별. |
    """)
    
    st.markdown("---")
    st.subheader("⚙️ System Architecture Overview")
    st.code("""
    [Input Data] 
      ├── Market Prices (SPY, TLT, VIX)
      ├── Macro Indicators (Unemployment, Yield Curve)
      └── Liquidity Metircs (Net Liquidity, FX Carry)
          ⬇️
    [Feature Engineering]
      ├── Structural Features (Skewness, Kurtosis)
      ├── Systemic Risk (Absorption Ratio, Correlation)
      └── HMM Regime Detection (Vol + Skew + Corr)
          ⬇️
    [Feature Selection]
      🔍 Double Filter (Point Biserial ∩ Mutual Info)
          ⬇️
    [Ensemble Model]
      🤖 Random Forest (Bagging) + XGBoost (Boosting)
      ⚖️ Voting Classifier (Soft Voting)
          ⬇️
    [Prediction]
      🚨 Crash Probability (0 ~ 100%)
    """, language='text')
