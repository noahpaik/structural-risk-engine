# 📉 Structural Risk Detector 2026

금융 시장의 구조적 위험(Structural Risk)을 실시간으로 감지하고, 폭락(Crash) 가능성을 예측하여 최적의 주식 비중을 제안하는 AI 기반 대시보드입니다.

> **Last Updated:** 2026-01-27
> - **Bond Stress Robustness**: MOVE 지수 누락 시 TLT 변동성 자동 대체 & VIX 기준 시계열 정렬 적용
> - **Model Tuning**: 과적합 방지를 위한 가중치 최적화 (8.0x) 및 고정 임계값(0.25) 적용
> - **Eco Surprise**: 지표 민감도 조정 (De-powering)

---

## 🚀 주요 기능 (Features)

### 1. 📊 실시간 위험 평가 (Real-time Assessment)
현재 시장 상황이 안전한지, 위험한지를 한눈에 보여줍니다.
- **위험 등급 (Risk Level)**: `Normal` 🟢 -> `Elevated` 🟡 -> `High` 🔴 -> `Critical` 🚨 4단계로 구분됩니다.
- **폭락 확률 (Crash Probability)**: 향후 20일 내 기장 급락(-7% 이상) 확률입니다. (25% 이상이면 위험 신호)
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
각 레이어는 세부 지표들을 종합하여 Z-score(표준점수)로 변환됩니다. 모든 데이터는 Timezone-Naive로 변환되어 정렬됩니다.

| 레이어 (Layer) | 주요 지표 (Indicators) | 설명 |
| :--- | :--- | :--- |
| **1. Volatility** | `VIX Term Structure`, `SKEW`, `Realized Vol` | 옵션 시장에 반영된 공포 지수 및 꼬리 위험(Tail Risk) 측정 |
| **2. Bond Stress** | `MOVE Index`, `High Yield Spread`, `Yield Curve` | **[Robust]** 채권 시장의 변동성 및 신용 위험.<br>- `^MOVE` 데이터 누락 시 **`TLT` 변동성**으로 자동 대체.<br>- VIX 날짜 기준으로 모든 채권 데이터 정렬(Alignment) |
| **3. Eco Surprise** | `Unemployment`, `CPI`, `Ind. Production` | **[De-powered]** 실업률, 물가, 산업생산 지표의 급격한 악화 감지.<br>- 과민 반응 방지를 위해 신호 강도 50% 축소 및 지속시간 Log 적용 |
| **4. Momentum** | `Price Trend`, `MA Divergence` | 주가의 추세가 꺾이거나 과매도/과매수 상태인지 분석 |
| **5. Liquidity** | `Market Depth`, `Bid-Ask Proxy` | 시장 미시구조(Microstructure) 데이터를 통한 유동성 고갈 감지 |
| **6. FX Carry** | `Currency Pairs`, `Carry Trade Return` | 환율 변동에 따른 캐리 트레이드 청산 위험 감지 |
| **7. Net Liquidity**| `Fed Balance Sheet`, `TGA`, `RRP` | 연준(Fed)의 유동성 공급/흡수 현황 추적 |

### 🧠 2. 모델 설정 (Model Configuration)
최근 시장 트렌드와 위기 상황을 더 민감하게 포착하도록 튜닝되었습니다.

- **알고리즘**: **XGBoost Classifier**
    - `max_depth`: 4 (과적합 방지)
    - `learning_rate`: 0.05
    - `scale_pos_weight`: **8.0** (폭락 데이터에 8배 가중치 학습)
- **Crash 정의 (Target)**: 향후 20일 내 **-7% 이상 하락**
- **임계값 (Threshold)**: **0.25 (Fixed)**. 확률이 25%를 넘으면 위험으로 간주.
- **출력 평활화 (Smoothing)**: 확률값의 노이즈를 줄이기 위해 **EMA 20 (20일 지수이동평균)** 적용.
- **데이터 보존**: `Validation` 시 최근 데이터를 보존하기 위해 `Left Join` 방식 사용 (Label Lag 문제 해결).

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
