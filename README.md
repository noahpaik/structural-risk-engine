# 📉 Structural Risk Detector 2026

금융 시장의 구조적 위험(Structural Risk)을 실시간으로 감지하고, 폭락(Crash) 가능성을 예측하여 최적의 주식 비중을 제안하는 AI 기반 대시보드입니다.

## 🚀 주요 기능 (Features)

### 1. 📊 실시간 위험 평가 (Real-time Assessment)
현재 시장 상황이 안전한지, 위험한지를 한눈에 보여줍니다.
- **위험 등급 (Risk Level)**: `Normal` 🟢 -> `Elevated` 🟡 -> `High` 🔴 -> `Critical` 🚨 4단계로 구분됩니다.
- **폭락 확률 (Crash Probability)**: 향후 20일 내 시장이 급락할 확률입니다. (20% 이상이면 위험 신호)
- **권장 주식 비중 (Equity Weight)**: 위험도에 따른 안전한 주식 투자 비중을 제안합니다. (위험할수록 현금/채권 비중 확대)
- **신호 강도 (Signal Strength)**: 어떤 요인(채권, 변동성, 거시경제 등)이 현재 위험을 주도하고 있는지 Z-score로 보여줍니다.

### 2. 📈 백테스트 성능 분석 (Backtest Results)
과거부터 현재까지 모델이 얼마나 잘 맞았는지 검증합니다.
- **성능 지표**: AUC, Recall(재현율), Precision(정밀도), F1-Score 등을 확인합니다.
- **종합 분석 차트 (Full History)**:
    - **SPY Price**: 주가 흐름과 실제 폭락 구간(빨간 배경)을 비교합니다.
    - **Probability**: 모델이 예측한 확률과 임계값(Threshold)을 시계열로 보여줍니다.
    - **Signals**: 각 레이어별 위험 신호의 흐름을 추적합니다.

### 3. 📡 상세 신호 분석 (Feature Signals)
7가지 리스크 레이어 각각의 세부 상태를 진단합니다.

---

## ⚙️ 기술적 상세 (Technical Details)

이 시스템은 **7가지 리스크 레이어**를 분석하여 **XGBoost** 모델로 최종 판단을 내립니다.

### 🔍 1. 리스크 레이어 (Risk Layers)
각 레이어는 세부 지표들을 종합하여 Z-score(표준점수)로 변환됩니다.

| 레이어 (Layer) | 주요 지표 (Indicators) | 설명 |
| :--- | :--- | :--- |
| **1. Volatility** | `VIX Term Structure`, `SKEW`, `Realized Vol` | 옵션 시장에 반영된 공포 지수 및 꼬리 위험(Tail Risk) 측정 |
| **2. Bond Stress** | `MOVE Index`, `High Yield Spread`, `Yield Curve (10Y-2Y)` | 채권 시장의 변동성 및 신용 위험 (MOVE 결측 시 VIX/국채금리 인덱스로 보완) |
| **3. Eco Surprise** | `Unemployment`, `CPI`, `Ind. Production` (YoY 가속도) | 실업률, 물가, 산업생산 지표가 급격히 악화되는 충격(Surprise) 감지 |
| **4. Momentum** | `Price Trend`, `MA Divergence` | 주가의 추세가 꺾이거나 과매도/과매수 상태인지 분석 |
| **5. Liquidity** | `Market Depth`, `Bid-Ask Proxy` | 시장 미시구조(Microstructure) 데이터를 통한 유동성 고갈 감지 |
| **6. FX Carry** | `Currency Pairs`, `Carry Trade Return` | 환율 변동에 따른 
캐리 트레이드 청산 위험 |
| **7. Net Liquidity**| `Fed Balance Sheet`, `TGA`, `RRP` | 연준(Fed)의 유동성 공급/흡수 현황 추적 |

### 🧠 2. 모델 설정 (Model Configuration)
최근 시장 트렌드와 위기 상황을 더 민감하게 포착하도록 튜닝되었습니다.

- **알고리즘**: XGBoost Classifier
- **Crash 정의 (Target)**: 향후 20일 내 **-7% 이상 하락** (기존 -10%에서 완화하여 민감도 증가)
- **학습 가중치 (Sensitivity)**: 폭락 데이터(Class 1)에 **5배(5.0x)** 가중치를 부여하여 작은 신호도 놓치지 않도록 설정 (Aggressive Tuning)
- **임계값 (Threshold)**: F2-Score(Recall 중시)를 최대화하는 값을 동적으로 찾되, **최소 0.25** 이상으로 유지

---

## 🛠️ 실행 및 배포 (Deployment)

### 로컬 실행
```bash
streamlit run app.py
```

### Streamlit Cloud 배포
1. GitHub 업로드 후 Streamlit Cloud 연결
2. **Secrets** 설정 필수:
```toml
FRED_API_KEY = "발급받은_API_KEY"
```
