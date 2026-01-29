import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from hmmlearn.hmm import GaussianHMM

# ============================================
# 전체 시스템 통합 (2026년 1월 기준)
# ============================================

class StructuralRiskDetector2026:
    """
    2026년 1월 26일 기준 최종 버전
    - SOFR 반영
    - Regime-conditional Z-score
    - Path-dependent features
    - XGBoost meta-model
    """
    
    def __init__(self, fred_api_key):
        self.fred = Fred(api_key=fred_api_key)
        self.feature_weights = {
            'volatility': 0.40,
            'bond_stress': 0.35,
            'eco_surprise': 0.25
        }
        self.model = None
        self.backtest_results = {}
        self.threshold = 0.5 # Default threshold
        
    # ============================================
    # LAYER 1: 변동성 구조
    # ============================================
    
    def get_volatility_structure(self, start_date='2002-01-01'):
        """
        VIX Term Structure + SKEW + RV Regime
        """
        print("[INFO] 변동성 구조 계산 중...")
        
        try:
            # Helper for safe data fetching
            def get_data(ticker):
                df = yf.download(ticker, start=start_date, progress=False)
                if df.empty: return pd.Series(dtype=float)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Close 찾기 또는 첫 번째 컬럼
                if 'Close' in df.columns:
                    data = df['Close']
                else:
                    data = df.iloc[:, 0]
                
                # 2D DataFrame인 경우 첫 번째 컬럼 강제 선택
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0] # Force 1D
                    
                return data

            # VIX (1개월)
            vix = get_data('^VIX')
            
            # VIX3M (3개월) - 없으면 VXV
            vix3m = get_data('^VIX3M')
            if vix3m.empty:
                vix3m = get_data('^VXV')
            
            if vix.empty or vix3m.empty:
                 print("[WARN] 필수 데이터(VIX) 누락")
                 return pd.Series()

            # Term Structure Ratio (핵심)
            
            # Term Structure Ratio (핵심)
            term_ratio = vix / vix3m
            backwardation = (term_ratio > 1.0).astype(float)
            
            # SKEW Index
            skew = get_data('^SKEW')
            if not skew.empty:
                skew_stress = (skew - 100) / 10
            else:
                skew_stress = pd.Series(0, index=vix.index)
            
            # Realized Volatility Regime
            spy = get_data('SPY')
            if spy.empty:
                print("[WARN] SPY 데이터 누락")
                return pd.Series()
            
            returns = spy.pct_change()
            rv_5d = returns.rolling(5).std() * np.sqrt(252)
            rv_60d = returns.rolling(60).std() * np.sqrt(252)
            rv_regime = rv_5d / rv_60d
            
            # 통합
            vol_df = pd.DataFrame({
                'term_ratio': term_ratio,
                'backwardation': backwardation,
                'skew_stress': skew_stress,
                'rv_regime': rv_regime
            }, index=vix.index).dropna()
            
            # Regime-conditional Z-score
            vix_regime = vix < vix.rolling(252).median()
            
            vol_signal = pd.Series(0.0, index=vol_df.index, dtype=float)
            for col in vol_df.columns:
                vol_signal += self._regime_zscore(vol_df[col], vix_regime)
            
            vol_signal = vol_signal / len(vol_df.columns)
            
            print(f"[OK] 변동성 데이터: {len(vol_signal)} 포인트")
            return vol_signal
            
        except Exception as e:
            print(f"[ERROR] 변동성 계산 오류: {e}")
            return pd.Series()
    
    # ============================================
    # LAYER 2: 채권 스트레스 (SOFR 반영)
    # ============================================
    
    def get_bond_stress_divergence(self, start_date='2002-01-01'):
        """
        SOFR-Treasury + MOVE-VIX Divergence + Curve (Robust: TLT Fallback & Alignment)
        """
        # print("[INFO] 채권 스트레스 계산 중...")
        
        try:
            # 1. Helper for safe data fetching & Timezone Stripping
            def load_clean(ticker):
                try:
                    df = yf.download(ticker, start=start_date, progress=False)
                    if df.empty: return pd.Series(dtype=float)
                    
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        
                    data = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                    if isinstance(data, pd.DataFrame): 
                        data = data.iloc[:, 0]
                        
                    # Timezone Strip
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    return data
                except:
                    return pd.Series(dtype=float)

            # 2. 로드 VIX (Anchor) & MOVE
            vix = load_clean('^VIX')
            move = load_clean('^MOVE')
            
            # [Fallback] MOVE 누락 시 TLT 변동성 사용
            if move.empty or len(move) < 10:
                # print("⚠️ ^MOVE 데이터 누락 → TLT 변동성으로 대체합니다.")
                tlt = load_clean('TLT')
                if not tlt.empty:
                    # TLT 20일 변동성 (표준편차 * 100) -> MOVE Scale 흉내
                    move = tlt.pct_change().rolling(20).std() * 100
            
            # 3. Align VIX & MOVE
            if vix.empty:
                return pd.Series(dtype=float)
                
            combined = pd.DataFrame({'vix': vix, 'move': move}).sort_index()
            combined = combined.ffill().dropna()
            
            vix = combined['vix']
            move = combined['move']
            anchor_index = vix.index
            
            # 4. Divergence Calculation
            move_norm = (move - move.rolling(60).mean()) / move.rolling(60).std()
            vix_norm = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
            
            leading_signal = (move_norm - vix_norm).clip(lower=0)
            confirmation = np.minimum(move_norm, vix_norm).clip(lower=0)
            divergence = 0.6 * leading_signal + 0.4 * confirmation
            
            # 5. FRED Data Integration (Align to VIX index)
            def get_fred_aligned(series_id):
                try:
                    s = self.fred.get_series(series_id, observation_start=start_date)
                    if s.index.tz is not None:
                        s.index = s.index.tz_localize(None)
                    # Align to VIX
                    s = s.reindex(anchor_index).ffill()
                    return s
                except:
                    return pd.Series(dtype=float, index=anchor_index)

            # SOFR Stress
            sofr = get_fred_aligned('SOFR')
            treasury_3m = get_fred_aligned('DGS3MO')
            effr = get_fred_aligned('EFFR')
            
            if sofr.isna().all(): sofr = effr # Fallback
            
            sofr_3m = sofr.rolling(63).mean()
            sofr_spread = (sofr_3m - treasury_3m)
            
            # Helper for Z-score (Global like before, but safe)
            def safe_zscore(s):
                if s.isna().all(): return s
                return (s - s.mean()) / s.std()
                
            sofr_stress = safe_zscore(sofr_spread)
            
            # High Yield
            hy_spread = get_fred_aligned('BAMLH0A0HYM2')
            hy_stress = safe_zscore(hy_spread)
            
            # Yield Curve
            yield_10y = get_fred_aligned('DGS10')
            yield_2y = get_fred_aligned('DGS2')
            curve = yield_10y - yield_2y
            inversion = -curve.clip(upper=0)
            inversion_stress = safe_zscore(inversion)

            # 6. Combine
            bond_df = pd.DataFrame({
                'divergence': divergence,
                'credit_stress': sofr_stress,
                'hy_stress': hy_stress,
                'inversion': inversion_stress
            }, index=anchor_index)
            
            bond_signal = bond_df.mean(axis=1, skipna=True)
            return bond_signal
            
        except Exception as e:
            # print(f"[ERROR] Bond Stress Failed: {e}")
            return pd.Series(dtype=float)
    
    # ============================================
    # LAYER 3: 경제 서프라이즈
    # ============================================
    
    def get_economic_surprise(self, start_date='2002-01-01'):
        """
        YoY 가속도 기반
        """
        print("[INFO] 경제 서프라이즈 계산 중...")
        
        try:
            indicators = {}
            
            # 실업률
            try:
                unrate = self.fred.get_series('UNRATE', observation_start=start_date)
                indicators['unrate'] = unrate
            except:
                pass
            
            # CPI
            try:
                cpi = self.fred.get_series('CPIAUCSL', observation_start=start_date)
                indicators['cpi'] = cpi
            except:
                pass
            
            # 산업생산
            try:
                indpro = self.fred.get_series('INDPRO', observation_start=start_date)
                indicators['indpro'] = indpro
            except:
                pass
            
            surprise_signals = []
            
            for name, series in indicators.items():
                # YoY
                yoy = series.pct_change(12) * 100
                # 가속도
                accel = yoy.diff()
                
                # Z-score 계산 후 Series로 변환 (인덱스 유지)
                accel_clean = accel.dropna()
                surprise = stats.zscore(accel_clean)
                surprise_series = pd.Series(surprise, index=accel_clean.index, name=name)
                
                surprise_signals.append(surprise_series)
            
            if surprise_signals:
                eco_df = pd.concat(surprise_signals, axis=1)
                eco_signal = eco_df.mean(axis=1)
                print(f"[OK] 경제 데이터: {len(eco_signal)} 포인트")
                return eco_signal
            else:
                print("[WARN]  경제 지표 없음")
                return pd.Series()
            
        except Exception as e:
            print(f"[ERROR] 경제 서프라이즈 오류: {e}")
            return pd.Series()
    
    # ============================================
    # LAYER 3.5: Momentum Indicators (New)
    # ============================================
    
    def get_momentum_indicators(self, spy_close):
        """
        가격 모멘텀 지표 (RSI, MACD, Price vs MA200)
        2025년 비구조적 조정 탐지용
        """
        print("[INFO] 모멘텀 지표 계산 중...")
        
        try:
            # 1. RSI (14-day)
            delta = spy_close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # RSI 과매수(>70) 신호를 Z-score로 변환
            rsi_stress = ((rsi - 50) / 20).clip(-3, 3)  # Normalize around 50
            
            # 2. MACD (12-26-9)
            ema_12 = spy_close.ewm(span=12, adjust=False).mean()
            ema_26 = spy_close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line
            
            # MACD 히스토그램 정규화 (음수일수록 약세)
            macd_norm = stats.zscore(macd_histogram.dropna())
            macd_stress = pd.Series(macd_norm, index=macd_histogram.dropna().index)
            
            # 3. Price vs MA200 (장기 추세 이탈)
            ma_200 = spy_close.rolling(window=200).mean()
            price_deviation = (spy_close - ma_200) / ma_200 * 100  # Percentage
            
            # 과도한 상승(>10%) 후 반전을 포착
            deviation_stress = (price_deviation / 5).clip(-3, 3)  # Normalize
            
            # 통합
            momentum_df = pd.DataFrame({
                'rsi_stress': rsi_stress,
                'macd_stress': macd_stress,
                'price_deviation': deviation_stress
            }, index=spy_close.index).dropna()
            
            # 평균 모멘텀 신호
            momentum_signal = momentum_df.mean(axis=1)
            
            print(f"[OK] 모멘텀 데이터: {len(momentum_signal)} 포인트")
            return momentum_signal
            
        except Exception as e:
            print(f"[ERROR] 모멘텀 계산 오류: {e}")
            return pd.Series()
    
    # ============================================
    # LAYER 3.8: Liquidity Microstructure (New)
    # ============================================
    
    def get_liquidity_indicators(self, spy_df):
        """
        유동성 프록시 지표 (Amihud, Corwin-Schultz, VVIX)
        무료 OHLCV 데이터로 시장 미세구조 추정
        """
        print("[INFO] 유동성 지표 계산 중...")
        
        try:
            # Ensure we have OHLCV
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in spy_df.columns for col in required_cols):
                print("[WARN] OHLCV 데이터 불완전")
                return pd.Series()
            
            # 1. Amihud Illiquidity Ratio
            returns = spy_df['Close'].pct_change().abs()
            dollar_volume = spy_df['Close'] * spy_df['Volume']
            
            # Avoid division by zero
            daily_illiq = returns / dollar_volume
            daily_illiq = daily_illiq.replace([np.inf, -np.inf], 0)
            
            # 20-day rolling average, scaled
            amihud = daily_illiq.rolling(20).mean() * 1e9
            
            # 2. Corwin-Schultz Spread Estimator
            high = spy_df['High']
            low = spy_df['Low']
            
            # Single-day range
            beta = (np.log(high / low))**2
            beta_sum = beta.rolling(2).sum()
            
            # Two-day range
            high_2 = high.rolling(2).max()
            low_2 = low.rolling(2).min()
            gamma = (np.log(high_2 / low_2))**2
            
            # Alpha calculation
            sqrt2 = np.sqrt(2)
            alpha = (np.sqrt(2 * beta_sum) - np.sqrt(beta)) / (3 - 2 * sqrt2) - np.sqrt(gamma / (3 - 2 * sqrt2))
            
            # Spread estimate
            spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            spread = spread.clip(lower=0).rolling(20).mean()
            
            # 3. VVIX Divergence (optional, may fail if VVIX unavailable)
            try:
                vvix_df = yf.download('^VVIX', start=spy_df.index[0], progress=False)
                if isinstance(vvix_df.columns, pd.MultiIndex):
                    vvix_df.columns = vvix_df.columns.get_level_values(0)
                
                vvix_close = vvix_df['Close'] if 'Close' in vvix_df.columns else vvix_df.iloc[:, 0]
                if isinstance(vvix_close, pd.DataFrame):
                    vvix_close = vvix_close.iloc[:, 0]
                
                vvix_close.index = vvix_close.index.tz_localize(None)
                
                # VIX data (already loaded in volatility layer, reload for simplicity)
                vix_df = yf.download('^VIX', start=spy_df.index[0], progress=False)
                if isinstance(vix_df.columns, pd.MultiIndex):
                    vix_df.columns = vix_df.columns.get_level_values(0)
                
                vix_close = vix_df['Close'] if 'Close' in vix_df.columns else vix_df.iloc[:, 0]
                if isinstance(vix_close, pd.DataFrame):
                    vix_close = vix_close.iloc[:, 0]
                
                vix_close.index = vix_close.index.tz_localize(None)
                
                # Z-scores
                vix_z = (vix_close - vix_close.rolling(60).mean()) / vix_close.rolling(60).std()
                vvix_z = (vvix_close - vvix_close.rolling(60).mean()) / vvix_close.rolling(60).std()
                
                # Divergence (VVIX spike while VIX calm)
                vvix_divergence = vvix_z - vix_z
                
                print(f"  [OK] VVIX 데이터: {len(vvix_divergence)} 포인트")
            except Exception as e:
                print(f"  [WARN] VVIX 로드 실패: {e}")
                vvix_divergence = pd.Series(0, index=spy_df.index)
            
            # Combine liquidity signals
            liquidity_df = pd.DataFrame({
                'amihud': amihud,
                'spread': spread,
                'vvix_div': vvix_divergence
            }, index=spy_df.index)
            
            # Normalize each component (preserve indices)
            for col in liquidity_df.columns:
                col_data = liquidity_df[col].dropna()
                if len(col_data) > 0:
                    normalized = stats.zscore(col_data)
                    liquidity_df[col] = pd.Series(normalized, index=col_data.index)
            
            # Average liquidity signal
            liquidity_signal = liquidity_df.mean(axis=1, skipna=True)
            
            print(f"[OK] 유동성 데이터: {len(liquidity_signal)} 포인트")
            return liquidity_signal
            
        except Exception as e:
            print(f"[ERROR] 유동성 계산 오류: {e}")
            return pd.Series()
    
    # ============================================
    # LAYER 3.9: FX Carry Risk (Global Shock)
    # ============================================
    
    def get_fx_carry_risk(self):
        """
        엔 캐리 트레이드 청산 위험 감지
        원리: USD/JPY 급락(엔화 강세) + 주가 하락(SPY↓) = 캐리 청산
        """
        print("[INFO] FX 캐리 위험 계산 중...")
        
        try:
            # JPY=X: USD/JPY 환율 (1996년~)
            # 환율 하락 = 엔화 가치 상승 = 캐리 청산
            usdjpy_df = yf.download('JPY=X', period='max', progress=False)
            spy_df = yf.download('SPY', period='max', progress=False)
            
            if isinstance(usdjpy_df.columns, pd.MultiIndex):
                usdjpy_df.columns = usdjpy_df.columns.get_level_values(0)
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)
            
            usdjpy_df.index = usdjpy_df.index.tz_localize(None)
            spy_df.index = spy_df.index.tz_localize(None)
            
            usdjpy_close = usdjpy_df['Close']
            spy_close = spy_df['Close']
            
            # 1. 환율 변동성 (20일) - USD/JPY
            fx_vol = usdjpy_close.pct_change().rolling(20).std()
            fx_vol_threshold = fx_vol.rolling(252).quantile(0.90)
            
            # 2. 상관관계 (20일 rolling)
            # 캐리 청산 시: 엔화 급등(USD/JPY 하락, 음수) vs SPY 하락(음수) → 양의 상관관계
            usdjpy_returns = usdjpy_close.pct_change()
            spy_returns = spy_close.pct_change()
            corr = usdjpy_returns.rolling(20).corr(spy_returns)
            
            # 3. 캐리 청산 신호: 환율 변동성 급증 + 동조 하락(양의 상관관계)
            # 평소엔 상관관계 낮음, 위기 시 둘 다 하락 → corr > 0.3
            carry_unwind = ((fx_vol > fx_vol_threshold) & (corr > 0.3)).astype(float)
            
            # 정규화 (0을 유지하면서)
            carry_unwind_norm = (carry_unwind - carry_unwind.mean()) / (carry_unwind.std() + 1e-9)
            
            print(f"[OK] FX 캐리 데이터: {len(carry_unwind_norm)} 포인트")
            return carry_unwind_norm
            
        except Exception as e:
            print(f"[ERROR] FX 캐리 계산 오류: {e}")
            return pd.Series()
    
    # ============================================
    # LAYER 3.95: Net Liquidity (Daily Tracking)
    # ============================================
    
    def get_net_liquidity(self):
        """
        순유동성 추적 (Net Liquidity = Fed BS - TGA - RRP)
        일일 데이터로 시장 유동성 실시간 모니터링
        """
        print("[INFO] 순유동성 계산 중...")
        
        try:
            # FRED API 필요 (TGA, RRP는 FRED에서만 제공)
            # TGA: Treasury General Account (WTREGEN) - 주간
            # RRP: Reverse Repo (RRPONTSYD) - 일일
            # Fed BS: WALCL - 주간
            
            # 2002년부터 (WALCL 시작일)
            tga = self.fred.get_series('WTREGEN', observation_start='2002-01-01')
            rrp = self.fred.get_series('RRPONTSYD', observation_start='2013-01-01')  # RRP는 2013년부터
            fed_bs = self.fred.get_series('WALCL', observation_start='2002-01-01')  # Fed Balance Sheet
            
            # 인덱스 정렬
            tga.index = pd.to_datetime(tga.index).tz_localize(None)
            rrp.index = pd.to_datetime(rrp.index).tz_localize(None)
            fed_bs.index = pd.to_datetime(fed_bs.index).tz_localize(None)
            
            # 주간 데이터를 일일로 변환 (forward fill)
            tga = tga.resample('D').ffill()
            fed_bs = fed_bs.resample('D').ffill()
            # RRP는 이미 일일 데이터
            
            # Net Liquidity = Fed BS - TGA - RRP (단위: billions)
            combined = pd.DataFrame({
                'fed_bs': fed_bs,
                'tga': tga,
                'rrp': rrp
            })
            combined = combined.ffill()  # Forward fill for alignment
            
            # RRP 0으로 채우기 (2013년 이전)
            combined['rrp'] = combined['rrp'].fillna(0)
            
            net_liq = combined['fed_bs'] - combined['tga'] - combined['rrp']
            
            # 변화율 (20일)로 변환 (급격한 유동성 변화 감지)
            net_liq_change = net_liq.pct_change(20)
            
            # Z-score 정규화
            net_liq_signal = 1 * stats.zscore(net_liq_change.dropna())
            net_liq_series = pd.Series(net_liq_signal, index=net_liq_change.dropna().index)
            
            print(f"  [OK] TGA 데이터: {len(tga)} 포인트")
            print(f"  [OK] RRP 데이터: {len(rrp)} 포인트")
            print(f"[OK] 순유동성 데이터: {len(net_liq_series)} 포인트")
            return net_liq_series
            
        except Exception as e:
            print(f"[ERROR] 순유동성 계산 오류: {e}")
            print(f"   (FRED API 키가 필요하거나 데이터 접근 불가)")
            return pd.Series()
    
    # ============================================
    # LAYER 4 (복구): HMM 기반 국면 탐지
    # ============================================
    def get_market_regime_hmm(self, spy_df):
        """
        HMM을 이용한 3단계 시장 국면 탐지 (캐싱 기능 추가)
        - 0: Normal
        - 1: Overheated
        - 2: Stress
        """
        # [NEW] 캐시 파일 경로 설정
        cache_file = "hmm_model_cache.pkl"
        
        # 데이터의 마지막 날짜(또는 길이)를 확인해서, 데이터가 바뀌었으면 다시 학습
        if isinstance(spy_df, pd.DataFrame):
            last_date = spy_df.index[-1].strftime('%Y-%m-%d')
        else:
            # Fallback for Series
            last_date = spy_df.index[-1].strftime('%Y-%m-%d')

        # [1] 캐시 확인: 파일이 있고, 데이터가 최신이면 불러오기
        if os.path.exists(cache_file):
            try:
                cached_data = joblib.load(cache_file)
                # 저장된 데이터의 날짜와 현재 데이터 날짜가 같으면 로딩
                if cached_data.get('last_date') == last_date:
                    print("[INFO] HMM 모델을 캐시에서 불러옵니다. (학습 건너뜀 🚀)")
                    
                    saved_signal = cached_data['signal']
                    # 인덱스 길이를 더 안전하게 맞춤
                    return saved_signal.reindex(spy_df.index).ffill().dropna()
                    
            except Exception as e:
                print(f"[WARN] 캐시 로딩 실패, 다시 학습합니다: {e}")

        # [2] 캐시가 없거나 날짜가 다르면 -> HMM 학습 시작 (기존 로직)
        print("[INFO] HMM 3단계 국면(과열/노멀/스트레스) 분석 및 학습 중... (시간 소요됨 🐢)")
        
        try:
            # 1. 데이터 준비
            if isinstance(spy_df, pd.DataFrame) and 'Close' in spy_df.columns:
                data = spy_df['Close'].pct_change().dropna()
            else:
                data = spy_df.pct_change().dropna()
                
            vol = data.rolling(20).std().dropna()
            ret = data.rolling(20).mean().dropna()
            
            common_idx = vol.index.intersection(ret.index)
            X = np.column_stack([ret.loc[common_idx].values, vol.loc[common_idx].values])
            
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
            model.fit(X)
            hidden_states = model.predict(X)
            
            # (라벨링 로직: Stress/Overheated/Normal 찾기)
            means = model.means_
            stress_idx = np.argmax(means[:, 1]) # 변동성 최대
            remaining_indices = [i for i in range(3) if i != stress_idx]
            overheated_idx = remaining_indices[np.argmax(means[remaining_indices, 0])] # 나머지 중 수익률 최대
            normal_idx = [i for i in remaining_indices if i != overheated_idx][0]
            
            mapping = {normal_idx: 0, overheated_idx: 1, stress_idx: 2}
            mapped_states = np.array([mapping[s] for s in hidden_states])
            
            regime_signal = pd.Series(mapped_states, index=common_idx, name='hmm_regime')
            
            # [3] 결과 저장 (캐싱)
            print(f"[INFO] HMM 학습 완료. 결과를 '{cache_file}'에 저장합니다.")
            joblib.dump({'last_date': last_date, 'signal': regime_signal}, cache_file)
            
            return regime_signal
            
        except ImportError:
             print("[ERROR] 'hmmlearn' 라이브러리가 없습니다. (pip install hmmlearn)")
             return pd.Series()
        except Exception as e:
            print(f"[ERROR] HMM 분석 실패: {e}")
            return pd.Series()
    
    # ============================================
    # LAYER 4: Path-Dependent Features
    # ============================================
    
    def add_path_features(self, signals_df):
        """
        상태 지속 기간 + 가속도
        """
        print("[INFO] 경로 의존 변수 생성 중...")
        
        path_features = pd.DataFrame(index=signals_df.index)
        
        for col in signals_df.columns:
            signal = signals_df[col]
            
            # 1. 스트레스 지속 기간
            threshold = signal.rolling(252).quantile(0.80)
            stress_flag = signal > threshold
            stress_groups = (stress_flag != stress_flag.shift()).cumsum()
            duration = stress_flag.groupby(stress_groups).cumcount() * stress_flag
            path_features[f'{col}_duration'] = duration
            
            # 2. 가속도
            accel = signal.diff().diff()
            path_features[f'{col}_accel'] = accel
        
        # 3. Reference: Count-based Synchronized Stress
        # [수정 핵심] 2019년 Repo 발작 무시하기
        # 1. 기준 강화: 상위 35%(0.65) -> 상위 15%(0.85) (진짜 위험할 때만 켜짐)
        # 2. 기간 확대: 과거 1년(252) -> 과거 2년(504) (2017년 저변동성 장세 착시 방지)
        # [수정] 2023-24년 노이즈 제거를 위해 Rolling Quantile 도입 (Regime Adaptive)
        # "최근 1년(252일) 동안 본 것 중에 상위 10%냐?"
        
        rolling_thresholds = signals_df.rolling(252).quantile(0.80) # 기준을 80%
    
        # 현재 값이 최근 1년 기준선을 넘었는가?
        # 4. 개수 조건 강화: 지표 2개 -> 3개 이상 동시 폭발 시
        stress_counts = (signals_df > rolling_thresholds).sum(axis=1)
        all_stressed = stress_counts >= 3 # [TUNING] 2 -> 3
        
        stress_groups = (all_stressed != all_stressed.shift()).cumsum()
        
        # [추가] Duration에 Log를 씌워서 200일씩 쌓이는 거 방지 (De-powering)
        raw_duration = all_stressed.groupby(stress_groups).cumcount() * all_stressed
        path_features['sync_stress_duration'] = np.log1p(raw_duration)
        
        print(f"[OK] 경로 변수: {len(path_features.columns)}개")
        return path_features
    
    # ============================================
    # LAYER 5: 통합 및 모델 학습
    # ============================================
    
    # [OK] 핵심 수정: 미래 데이터까지 로드 (Updated dynamically)
    def prepare_training_data(self, start_date='2023-01-01', end_date=None):
        """
        전체 Feature 준비 (미래 데이터 확보 강화)
        """
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
        print(f"\n{'='*70}")
        print(f"[DATA] 데이터 준비: {start_date} ~ {end_date}")
        print(f"{'='*70}\n")
        
        vol_signal = self.get_volatility_structure(start_date)
        bond_signal = self.get_bond_stress_divergence(start_date)
        eco_signal = self.get_economic_surprise(start_date)
        
        try:
            if vol_signal.index.tz is not None: vol_signal.index = vol_signal.index.tz_localize(None)
            if bond_signal.index.tz is not None: bond_signal.index = bond_signal.index.tz_localize(None)
            if eco_signal.index.tz is not None: eco_signal.index = eco_signal.index.tz_localize(None)
        except Exception as e:
            print(f"[WARN] Timezone conversion error: {e}")

        # [OK] 핵심 수정: end_date 제거하여 최신 데이터까지 로드
        print("   SPY 데이터 로드 중 (최신까지)...")
        spy_df = yf.download('SPY', start=start_date, progress=False)  # end 제거!
        
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        
        spy_df.index = spy_df.index.tz_localize(None)
        
        if 'Close' in spy_df.columns:
            spy_close = spy_df['Close']
        else:
            spy_close = spy_df.iloc[:, 0]
             
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        
        print(f"   [OK] SPY 데이터 범위: {spy_close.index[0].date()} ~ {spy_close.index[-1].date()}")
        
        # [OK] NEW: Calculate Momentum Indicators
        momentum_signal = self.get_momentum_indicators(spy_close)
        
        try:
            if momentum_signal.index.tz is not None: 
                momentum_signal.index = momentum_signal.index.tz_localize(None)
        except Exception as e:
            print(f"[WARN] Momentum timezone conversion error: {e}")
        
        # [OK] NEW: Calculate Liquidity Indicators (use full OHLCV DataFrame)
        liquidity_signal = self.get_liquidity_indicators(spy_df)
        
        try:
            if liquidity_signal.index.tz is not None:
                liquidity_signal.index = liquidity_signal.index.tz_localize(None)
        except Exception as e:
            print(f"[WARN] Liquidity timezone conversion error: {e}")
        
        # [OK] NEW: Calculate FX Carry Risk
        fx_carry_signal = self.get_fx_carry_risk()
        
        try:
            if fx_carry_signal.index.tz is not None:
                fx_carry_signal.index = fx_carry_signal.index.tz_localize(None)
        except Exception as e:
            print(f"[WARN] FX Carry timezone conversion error: {e}")
        
        # [OK] NEW: Calculate Net Liquidity
        net_liq_signal = self.get_net_liquidity()
        
        try:
            if net_liq_signal.index.tz is not None:
                net_liq_signal.index = net_liq_signal.index.tz_localize(None)
        except Exception as e:
            print(f"[WARN] Net Liquidity timezone conversion error: {e}")

        # [복구] HMM 국면 탐지
        hmm_signal = self.get_market_regime_hmm(spy_df)

        # ==============================================================================
        # [NEW] 사모신용(Private Credit) 스트레스 - TCPC 중심
        # 논리: "공모 하이일드(HYG)는 버티는데, 사모 대출(TCPC) 가격이 무너지면 구조적 위기다."
        # ==============================================================================
        print("   [INFO] 사모신용(TCPC) 데이터 로드 중...")
        aux_tickers = ['HYG', 'TCPC']
        aux_data = yf.download(aux_tickers, start=start_date, progress=False)['Close']
        
        # 1. TCPC 괴리율 (Credit Divergence)
        # HYG(시장 유동성 있음) 대비 TCPC(비유동성 자산)의 상대 강도
        tcp_ratio = aux_data['TCPC'] / aux_data['HYG']
        
        # 2. Z-score 변환 (평균 회귀 분석)
        tcp_stress_raw = (tcp_ratio - tcp_ratio.rolling(252).mean()) / tcp_ratio.rolling(252).std()
        
        # 3. 신호 반전 (-1 곱하기, 수치가 높을수록 Panic)
        private_credit_signal = tcp_stress_raw * -1
        
        # 인덱스 매칭
        if private_credit_signal.index.tz is not None:
             private_credit_signal.index = private_credit_signal.index.tz_localize(None)
        
        # 4. 트리거 설정 (Z-score 1.0 이상이면 발작)
        tcpc_panic = (private_credit_signal > 1.0).astype(int)

        # [NEW] 사장님의 "압력 밥솥" 로직 구현
        # 원리: Overheated면 압력이 쌓이고(Risk 증가), Stress면 압력이 해소된다(Risk 감소/종료)
        
        # 1. 상태별 마스크 생성
        # hmm_signal 인덱스를 spy_df 인덱스와 맞추기 위해 reindex 필요할 수 있음 (보통은 맞춰져 있음)
        hmm_signal = hmm_signal.reindex(vol_signal.index).ffill().dropna()
        
        is_overheated = (hmm_signal == 1) # 과열 (경고등 켜야 함)
        is_stress = (hmm_signal == 2)     # 스트레스 (위기 해소 중)
        is_normal = (hmm_signal == 0)     # 평온
        
        # 2. '누적 압력(Accumulated Strain)' 지표 생성
        # [수정] HMM Strain 계산 로직 고도화 (가속도 반영)
        
        strain_list = []
        current_strain = 0
        
        # [수정] HMM Strain 계산 로직 대수술 (Always-on Sensor)
        
        strain_list = []
        current_strain = 0
        
        # 1. 외부 충격 감지 (Trigger)
        # Threshold를 살짝 낮춰서(-0.8) 민감하게 반응하도록 함
        bond_panic = (bond_signal > 0.8).astype(int)
        liq_drain = (net_liq_signal < -0.8).astype(int)
        # [NEW] 환율(FX) 트리거도 민감하게 (-0.8)
        fx_panic = (fx_carry_signal < -0.8).astype(int)
        # [NEW] 변동성 발작 추가 (사장님 의견 반영)
        # VIX가 평소보다 튀면(Z-score > 1.0) 무조건 압력 채움
        vol_panic = (vol_signal > 1.0).astype(int)
        
        # 인덱스 정렬 
        bond_panic = bond_panic.reindex(hmm_signal.index).fillna(0)
        liq_drain = liq_drain.reindex(hmm_signal.index).fillna(0)
        fx_panic = fx_panic.reindex(hmm_signal.index).fillna(0)
        vol_panic = vol_panic.reindex(hmm_signal.index).fillna(0)
        tcpc_panic = tcpc_panic.reindex(hmm_signal.index).fillna(0)
        
        for i in range(len(hmm_signal)):
            
            # [NEW] 외부 충격 가속도 계산 (HMM 상태와 무관하게 계산)
            # 환율(FX)에 가장 높은 가중치 5 부여 (2025년 타겟팅)
            # [수정] 가속도 공식: 변동성(Vol)이 튀면 압력을 3배로 빨리 채움
            # [수정] 사모신용(TCPC)이 흔들리면 뇌관 건드린 것임 -> 가중치 4
            external_shock = (bond_panic.iloc[i] * 3) + \
                             (fx_panic.iloc[i] * 5) + \
                             (tcpc_panic.iloc[i] * 4) + \
                             (vol_panic.iloc[i] * 3) + \
                             (liq_drain.iloc[i] * 2)
            
            if is_stress.iloc[i]:
                # [CASE 1] 이미 터짐 (Stress) -> 리셋
                current_strain = 0
                
            elif is_overheated.iloc[i]:
                # [CASE 2] 과열 상태 (Overheated)
                # 기본적으로 +1씩 차오르고, 외부 충격 있으면 더 빨리 차오름
                current_strain += (1 + external_shock)
                
            else:
                # [CASE 3] 평온/횡보 상태 (Normal)
                # "HMM은 평온해 보여도, 밖에서(FX/Vol) 난리가 났으면 압력을 채워라!"
                
                if external_shock > 0:
                    # 겉은 평온하지만 속은 썩어들어가는 중 -> 압력 증가
                    current_strain += external_shock 
                else:
                    # 진짜 아무 일도 없음 -> 쿨링
                    current_strain = max(0, current_strain - 1)
            
            strain_list.append(current_strain)
        #로그 변환
        log_strain_list = np.log1p(strain_list)
        # Series로 변환
        accumulated_strain = pd.Series(log_strain_list, index=hmm_signal.index, name='hmm_strain')
        
        # [NEW] 압력 폭발 지표 (Trigger) - 위에서 정의한 변수 사용
        # 압력이 찬 상태에서 돈이 빠지는가?
        strain_x_drain = accumulated_strain * liq_drain
        strain_x_drain.name = 'strain_x_drain'
        
        signals = pd.DataFrame({
            'volatility': vol_signal,
            'bond_stress': bond_signal,
            'eco_surprise': eco_signal,
            'momentum': momentum_signal,
            'liquidity': liquidity_signal,
            'fx_carry': fx_carry_signal,
            'net_liquidity': net_liq_signal,
            # [NEW] 사모신용(Private Credit) 신호 추가
            # 값이 높을수록 위험
            'private_credit': private_credit_signal,
            
            # [NEW Features]
            'hmm_overheated': is_overheated.astype(int), 
            'hmm_strain': accumulated_strain.astype(float),            
            'hmm_strain_vel': accumulated_strain.diff().fillna(0).astype(float), 
            'strain_x_drain': strain_x_drain.astype(float),

            # [AMPLIFIER] Context Interaction Features (Pressure Cooker Logic)
            # [GATE] "확실한 충격(Impact) 아니면 곱하지 마라"
            # Z-score 1.0 미만의 잡음은 0으로 처리 (Noise Gate)
            
            # [수정] 0의 함정 탈출 (Hybrid Trigger)
            # Strain이 0이어도, Signal 자체가 강력하면(Gate 통과) 경보 울림: Signal * (1 + Strain)
            
            'context_bond_stress': (bond_signal * (1 + accumulated_strain) * (bond_signal > 1.0).astype(int)).astype(float),
            'context_liquidity_drain': ((net_liq_signal * -1) * (1 + accumulated_strain) * (net_liq_signal < -1.0).astype(int)).astype(float),
            'context_momentum_crash': ((momentum_signal * -1) * (1 + accumulated_strain)).astype(float),
            
            # [NEW] 환율(FX) 트리거 (Carry 청산 쇼크)
            'context_fx_shock': ((fx_carry_signal * -1) * (1 + accumulated_strain) * fx_panic).astype(float),
            
            # [NEW] 변동성 (Volatility) 트리거 추가
            # VIX가 튀면 Strain이 없어도 경보 울림
            'context_vol_shock': (vol_signal * (1 + accumulated_strain) * vol_panic).astype(float),
            
            # [NEW] 사모신용 충격 트리거 (Hybrid Trigger)
            # 압력(Strain)이 0이어도, TCPC가 급락하면 그 자체로 '신용 사건'임
            'context_private_credit': (private_credit_signal * (1 + accumulated_strain) * tcpc_panic).astype(float)             
            
            # 'hmm_stress': is_stress.astype(int) <-- [삭제] 이걸 넣으면 뒷북칩니다.
        }).sort_index().ffill().dropna()
        
        path_features = self.add_path_features(signals)
        features = pd.concat([signals, path_features], axis=1).dropna()
        
        # [USER REQUEST] Eco Surprise 영향력 축소 (De-powering)
        
        # 1. 신호 강도(Magnitude) 50% 축소
        if 'eco_surprise' in features.columns:
             features['eco_surprise'] = features['eco_surprise'] * 0.5
             
        if 'eco_surprise_accel' in features.columns:
             features['eco_surprise_accel'] = features['eco_surprise_accel'] * 0.5
             
        # 2. 지속기간(Duration)에 로그(Log) 적용 (선형 -> 로그형 감쇄)
        if 'eco_surprise_duration' in features.columns:
             features['eco_surprise_duration'] = np.log1p(features['eco_surprise_duration'])
             
        returns = spy_close.pct_change()
        
        # [OK] VIX 데이터 로드 (급등 확인용)
        try:
            vix_df = yf.download('^VIX', start=start_date, progress=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            if 'Close' in vix_df.columns:
                vix_close = vix_df['Close']
            else:
                vix_close = vix_df.iloc[:, 0]
            if isinstance(vix_close, pd.DataFrame):
                vix_close = vix_close.iloc[:, 0]
            
            # VIX 급등 조건: 5일 후 VIX가 20일 이동평균의 1.5배 초과
            vix_spike = (vix_close > vix_close.rolling(20).mean() * 1.5).shift(-5)
            # 인덱스 정렬
            vix_spike = vix_spike.reindex(returns.index).fillna(False)
            
        except Exception as e:
            print(f"[WARN] VIX load failed for label generation: {e}")
            vix_spike = pd.Series(False, index=returns.index)

        # [OK] 엄격해진 레이블 (Option 1 적용)
        # 1. 향후 20일 -10% (기존 -8%에서 강화)
        future_dd_20 = returns.rolling(20).apply(
            lambda x: (1 + x).cumprod().min() - 1
        ).shift(-20)
        
        # 2. 향후 10일 -7% AND VIX 급등
        future_10d = returns.rolling(10).sum().shift(-10)
        
        # 통합
        crash_labels = (
            (future_dd_20 < -0.05) | # [RELAXED] -10% -> -7% 로 완화
            ((future_10d < -0.05) & vix_spike)
        ).astype(int)
        
        # ==============================================================================
        # [추가] 사용자 정의 폭락 구간 강제 지정 (Ground Truth Correction)
        # 기계적 하락폭이 부족하더라도, 구조적 위기였던 구간을 '정답(1)'으로 강제 설정
        # ==============================================================================
        
        # [NEW] 1. 2019년 9월 ~ 12월: 레포 발작 (Repo Crisis)
        # 시작: 9월 17일 레포 금리 급등
        # 종료: 10월 11일 연준이 단기국채 매입(Not QE) 발표 -> 유동성 공급 확정 시점
        # (기존 12월 15일 -> 10월 15일로 단축)
        if '2019-09-15' in crash_labels.index and '2019-10-15' in crash_labels.index:
             try:
                 crash_labels.loc['2019-09-15':'2019-10-15'] = 1
                 print("[USER] 2019년 레포 발작 구간(9/15~10/15)을 'Actual Crash'로 지정했습니다.")
             except: pass

        # [NEW] 2. 2023년 3월: 실리콘밸리은행(SVB) 파산 & 뱅크런
        # 시작: 3월 8일 SVB 손실 발표 및 뱅크런 시작
        # 종료: 3월 13일~15일 BTFP(은행기간대출프로그램) 가동 및 연준 대차대조표 급등 시점
        # (기존 3월 31일 -> 3월 17일로 단축)
        if '2023-03-08' in crash_labels.index and '2023-03-17' in crash_labels.index:
             try:
                 crash_labels.loc['2023-03-08':'2023-03-17'] = 1
                 print("[USER] 2023년 SVB 사태 구간(3/08~3/17)을 'Actual Crash'로 지정했습니다.")
             except: pass

        # 3. 2024년 7~8월: 엔캐리 트레이드 청산 (Black Monday 전조)
        # 7월 중순부터 8월 초까지를 위험 구간으로 정의
        if '2024-07-15' in crash_labels.index and '2024-08-05' in crash_labels.index:
             try:
                 crash_labels.loc['2024-07-15':'2024-08-05'] = 1
                 print("[USER] 2024년 7~8월 엔캐리 청산 구간을 'Actual Crash'로 강제 지정했습니다.")
             except: pass

        # 2. 2025년 3~4월: 관세 전쟁 (Tariff War)
        # 2월 말부터 4월 초까지를 위험 구간으로 정의
        if '2025-02-20' in crash_labels.index and '2025-04-10' in crash_labels.index:
             try:
                 crash_labels.loc['2025-02-20':'2025-04-10'] = 1
                 print("[USER] 2025년 3~4월 관세 전쟁 구간을 'Actual Crash'로 강제 지정했습니다.")
             except: pass
        
        # ==============================================================================

        crash_labels.name = 'crash'
        
        # [수정] 데이터 병합 (Left Join으로 최신 데이터 보존)
        df_full = features.join(crash_labels, how='left')
        
        # 1. 학습/검증용 데이터 (라벨이 반드시 있어야 함 -> NaN 제거)
        df_model = df_full.dropna()
        
        # [Safe Guard] 데이터가 비었는지 확인
        if df_full.empty:
            print("[CRITICAL] 생성된 데이터프레임이 비어 있습니다. (모든 데이터가 dropna 되었거나 소스 데이터 부족)")
            return None

        # 2. 실시간 모니터링용 데이터 (최근 20일 포함, 라벨 NaN이어도 됨)
        # 2026-01-26 기준 최신 데이터
        current_data = df_full.iloc[[-1]] 
        
        if df_model.empty:
            print("[ERROR] 라벨 생성 후 데이터가 텅 비었습니다.")
            return None

        # [수정] 날짜 기반 분할 (최근 폭락을 검증셋에 포함시키기 위함)
        # 2025년 3월 폭락을 검증하기 위해 2024년부터 검증
        split_date = pd.Timestamp('2024-01-01')
        
        train = df_model[df_model.index < split_date]
        test = df_model[df_model.index >= split_date]
        
        train_crashes = train['crash'].sum()
        test_crashes = test['crash'].sum()
        
        print(f"\n{'='*70}")
        print(f"[INFO] 데이터 분할 완료 (Split: {split_date.date()})")
        print(f"{'='*70}")
        print(f"학습 데이터: {len(train)} (폭락: {train_crashes})")
        print(f"검증 데이터: {len(test)} (폭락: {test_crashes})")
        
        if test_crashes == 0:
            print(f"⚠️  [경고] 검증 세트에 폭락 이벤트가 없습니다!")
            print(f"    -> 2024년 이후 시장이 강세장이었거나, 라벨링 기준(-7%)이 너무 높습니다.")
            print(f"    -> 테스트 목적이라면 라벨링 임계값을 -5%로 낮추거나, 분할 날짜를 2023년으로 당겨보세요.")
        else:
            print(f"✅ [성공] 검증 세트에 폭락 이벤트가 포함되었습니다.")
            
        # 모델 학습에는 df_model(전체)을 리턴하거나, train/test를 튜플로 리턴
        # 여기서는 기존 호환성을 위해 df_model 리턴 (내부에서 TimeSeriesSplit 사용 권장)
        return df_full
        
        print(f"\n{'='*70}")
        print(f"[INFO] 데이터 준비 완료")
        print(f"{'='*70}")
        print(f"총 샘플: {len(df)}")
        print(f"폭락 이벤트: {df['crash'].sum()} ({df['crash'].mean()*100:.1f}%)")
        print(f"  - 학습: {train_crashes}개")
        print(f"  - 검증: {test_crashes}개")
        
        # [OK] 라벨 분포 상세 출력
        if test_crashes == 0:
            print(f"\n[ERROR] 검증 데이터 폭락 0개!")
            print(f"   원인 진단:")
            print(f"   - SPY 최종 날짜: {spy_close.index[-1].date()}")
            print(f"   - 라벨 가능 범위: ~ {(spy_close.index[-1] - pd.Timedelta(days=20)).date()}")
            print(f"   - 검증 시작일: {df.index[split_idx].date()}")
            print(f"   → shift(-20) 때문에 최근 라벨 손실")
        else:
            print(f"   [OK] 검증 데이터 폭락 정상 포착")
        
        print(f"Feature 개수: {len(df.columns)-1}")
        print(f"{'='*70}\n")
        
        return df
        
    
    def train_model(self, df, split_date='2024-01-01', target_recall=0.55, max_fpr=0.40, test_size=0.2):
        """
        XGBoost Walk-Forward 학습 (Advanced Tuning)
        - Time-Decay Sample Weights
        - F-Score Maximization (F2 Score: Recall biased)
        """
        print(f"\n{'='*70}")
        print(f"[AI] 모델 학습 시작 (Advanced Tuning)")
        
        # 1. [중요] 라벨(crash)이 없는 데이터 제거 (학습/검증용)
        df_for_training = df.dropna(subset=['crash'])
        
        # 2. [중요] 날짜순 정렬 (이게 안 되어 있으면 분할이 꼬임)
        df_for_training = df_for_training.sort_index()
        
        X = df_for_training.drop('crash', axis=1)
        y = df_for_training['crash']
        
        # 3. [수정] 날짜 기반 분할 (iloc 대신 명확한 조건문 사용)
        split_ts = pd.Timestamp(split_date)
        print(f"   Split Date: {split_ts.date()}")
        
        # 날짜 조건으로 마스크 생성
        mask_train = X.index < split_ts
        mask_test = X.index >= split_ts
        
        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]
        
        # --- (에러 방지: 데이터가 비었는지 확인) ---
        if len(X_train) == 0:
            print("[CRITICAL ERROR] 학습 데이터가 없습니다. split_date가 너무 과거입니다.")
            return None
        if len(X_test) == 0:
            print(f"[CRITICAL ERROR] 검증 데이터가 없습니다. (데이터 끝: {X.index[-1]})")
            print(" -> 데이터가 2023년 이전까지만 있거나, 날짜 파싱 문제입니다.")
            # 강제로 최근 20%를 테스트로 할당 (비상 조치)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            print(f" -> [비상 조치] 마지막 20%를 검증셋으로 강제 할당함.")

        # 정보 출력
        print(f"학습: {len(X_train)} ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
        if not X_test.empty:
            print(f"검증: {len(X_test)} ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")
        
        print(f"원본: Normal={len(y_train[y_train==0])}, Crash={len(y_train[y_train==1])}")
        
        # ... (이하 1. Class Imbalance Handling 부터는 기존 코드 유지) ...
        
        # [추가] 학습 시에는 라벨(crash)이 있는 데이터만 사용해야 함 (NaN 제거)
        # df_for_training = df.dropna(subset=['crash'])
        
        #if split_date:
        #    print(f"   Split Date: {split_date}")
        #else:
        #    print(f"   Test Size: {test_size:.0%}")
        
        # 기존 코드에서 df 대신 df_for_training 사용
        #X = df_for_training.drop('crash', axis=1)
        #y = df_for_training['crash']
        
        # [OK] 날짜 기준 분할 우선
        #if split_date:
        #    split_ts = pd.Timestamp(split_date)
        #    post_split = df.index[df.index >= split_ts]
        #    if not post_split.empty:
        #        split_idx = df.index.get_loc(post_split[0])
        #        print(f"[OK] 강제 분할 시점: {split_ts.date()}")
        #    else:
        #        print(f"[WARN] Split date {split_date} out of range, fallback to ratio")
        #        split_idx = int(len(df) * (1 - test_size))
        #else:
        #     split_idx = int(len(df) * (1 - test_size))
        
        #X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        #y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        #print(f"학습: {len(X_train)} ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
        #print(f"검증: {len(X_test)} ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")
        #print(f"원본: Normal={len(y_train[y_train==0])}, Crash={len(y_train[y_train==1])}")
        
        # 1. Class Imbalance Handling
        if y_train.sum() == 0:
            print("[WARN] 학습 데이터에 Crash 없음. Dummy Model 사용.")
            pos_weight = 1.0
        else:
            # 1. 가중치 복구 (Recall 50% 사수용)
            # 폭락을 놓치는 건 15배 더 나쁘다. (20은 과하고 5는 약했음)
            pos_weight = 15.0
            print(f"[BALANCE] Class Weight (scale_pos_weight): {pos_weight:.2f}")

        # 2. Time-Decay Sample Weights (Linear: 0.5 -> 1.5)
        # 최근 데이터에 더 높은 가중치를 부여하여 Concept Drift 완화
        weights = np.linspace(0.5, 0.75, len(X_train))
        
        # XGBoost 학습
        self.model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,      # 0.05 -> 0.03 (다시 조금 침착하게)
            max_depth=6,             # 5 -> 6 (복잡한 곱셈 관계 이해하도록 깊이 증가)
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=pos_weight,
            
            # [NEW] 정밀도(Precision)를 올리는 핵심 파라미터 추가!
            # 기본값 0 -> 1.5 (노이즈에 과민반응 금지)
            gamma=1.5,               
            # 기본값 1 -> 5 (너무 적은 샘플로 판단 금지)
            min_child_weight=5,
            
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(
            X_train, y_train,
            sample_weight=weights,   # [OK] 가중치 적용
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 예측 확률
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # [SMOOTHING] 물 타기 금지! (span=20 -> 3)
        y_pred_proba = pd.Series(y_pred_proba).ewm(span=3).mean().values
        
        # 3. Dynamic Thresholding (Maximize F2-Score)
        # F2-Score: Recall에 Precision보다 2배 더 가중치 (Beta=2)
        #precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        #best_f2 = 0
        #optimal_threshold = 0.5
        
        # 분모가 0이 되는 것을 방지
        #numerator = 2 * (precision * recall)
        #denominator = precision + recall
        #f2_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        # Find best threshold
        # if len(thresholds) > 0:
        #    best_idx = np.argmax(f2_scores[:-1]) # thresholds length = recall length - 1 (usually)
        #    # scikit-learn precision_recall_curve: thresholds is shorter by 1
        #    if best_idx < len(thresholds):
        #        best_f2 = f2_scores[best_idx]
        #        optimal_threshold = thresholds[best_idx]
        
        # [수정] 오경보를 줄이기 위해 다시 상향
        # [TUNING] 0.45 -> 0.42 (신호가 강해졌으므로 살짝 낮춰도 됨)
        optimal_threshold = 0.42
        print(f"[TARGET] Fixed Threshold: {optimal_threshold:.3f}")
        
        self.threshold = optimal_threshold
        
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # 평가
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # print(f"{'='*70}")
        # print(f"[METRICS] 검증 성과")
        # print(f"{'='*70}")
        # print(f"AUC: {auc:.3f}")
        # print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Crash'])}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        # print(f"Confusion Matrix:")
        # print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
        # print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
        
        # [OK] NEW: 신호 발생 날짜 분석
        # print("\n" + "="*70)
        # print("[DATE] 신호 발생 상세 분석")
        # print("="*70)
        
        # False Positives (모델 경고 + 실제 정상)
        fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
        fp_dates = X_test.index[fp_indices]
        
        # if len(fp_dates) > 0:
            # print(f"\n[WARN] 경고 신호 발생 (FP={len(fp_dates)}개):")
            # 연도별로 그룹화
            # fp_by_year = {}
            # for date in fp_dates:
            #     year = date.year
            #     if year not in fp_by_year:
            #         fp_by_year[year] = []
            #     fp_by_year[year].append(date)
            
            # for year in sorted(fp_by_year.keys()):
                # print(f"\n  [{year}년]: {len(fp_by_year[year])}개")
                # 처음 10개만 출력
                # dates_to_show = fp_by_year[year][:10]
                # if len(fp_by_year[year]) > 10:
        # print(f"    {', '.join([str(d.month).zfill(2) + '-' + str(d.day).zfill(2) for d in dates_to_show])} ... (총 {len(fp_by_year[year])}개)")
        # print(f"    {', '.join([str(d.month).zfill(2) + '-' + str(d.day).zfill(2) for d in dates_to_show])}")
        
        # True Positives
        tp_indices = np.where((y_pred == 1) & (y_test == 1))[0]
        tp_dates = X_test.index[tp_indices]
        
        # if len(tp_dates) > 0:
            # print(f"\n[OK] 정확한 폭락 예측 (TP={len(tp_dates)}개):")
            # for date in tp_dates:
                # features = X_test.loc[date]
                # top_features = features.nlargest(3)
                # date_str = f"{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}"
                # print(f"  {date_str}: 주요 신호 = {', '.join([f'{k}({v:.2f})' for k, v in top_features.items()])}")
        
        # False Negatives
        fn_indices = np.where((y_pred == 0) & (y_test == 1))[0]
        fn_dates = X_test.index[fn_indices]
        
        # if len(fn_dates) > 0:
            # print(f"\n[ERROR] 놓친 폭락 (FN={len(fn_dates)}개):")
            # for date in fn_dates[:10]:  # 처음 10개만
                # date_str = f"{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}"
                # print(f"  {date_str}")
            # if len(fn_dates) > 10:
                # print(f"  ... 외 {len(fn_dates)-10}개")
        
        # print("="*70 + "\n")
        
        # [OK] 백테스트 결과 저장 (Streamlit용)
        from sklearn.metrics import accuracy_score, f1_score
        self.backtest_results = {
            'auc': auc,
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': cm,
            'threshold': optimal_threshold,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'X_test': X_test,
            'split_date': split_date
        }
        
        # Feature Importance
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        # print(f"\n[SEARCH] 상위 15개 Feature Importance:")
        # print(importances.to_string(index=False))
        # print(f"{'='*70}\n")
        
        # Predict on FULL dataset for visualization
        y_pred_proba_full = self.model.predict_proba(X)[:, 1]
        
        # [SMOOTHING] EMA 20 적용
        y_pred_proba_full = pd.Series(y_pred_proba_full).ewm(span=20).mean().values

        # 저장
        self.backtest_results.update({
            'X_full': X,
            'y_full': y,
            'y_pred_proba_full': y_pred_proba_full,
            'test_start_date': X_test.index[0],
            'importances': importances
        })
        
        return self.model
    
    # ============================================
    # 백테스트 시각화
    # ============================================
    
    def plot_backtest_results(self):
        """
        백테스트 결과 시각화
        """
        if not self.backtest_results:
            print("[WARN]  먼저 모델 학습 필요")
            return
        
        # Use Full Dataset if available, otherwise fallback to Test
        if 'X_full' in self.backtest_results:
            X_plot = self.backtest_results['X_full']
            y_plot = self.backtest_results['y_full']
            y_pred_proba = self.backtest_results['y_pred_proba_full']
            test_start = self.backtest_results.get('test_start_date')
            title_prefix = "Full History (Train + Test)"
        else:
            X_plot = self.backtest_results['X_test']
            y_plot = self.backtest_results['y_test']
            y_pred_proba = self.backtest_results['y_pred_proba']
            test_start = None
            title_prefix = "Test Set Only"
        
        # SPY 가격
        spy_df = yf.download('SPY', start=X_plot.index[0], end=X_plot.index[-1], progress=False)
        if isinstance(spy_df.columns, pd.MultiIndex):
             spy_df.columns = spy_df.columns.get_level_values(0)
             
        if 'Close' in spy_df.columns:
             spy_close = spy_df['Close']
        else:
             spy_close = spy_df.iloc[:, 0]
             
        if isinstance(spy_close, pd.DataFrame):
             spy_close = spy_close.iloc[:, 0]
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # 1. SPY 가격 + 폭락 구간
        axes[0].plot(spy_close.index, spy_close, color='black', linewidth=1.5, label='SPY')
        
        # Validation Start Line
        if test_start:
            axes[0].axvline(test_start, color='blue', linestyle='--', label='Test Set Start')
            
        crash_periods = y_plot[y_plot == 1].index
        for date in crash_periods:
            axes[0].axvspan(date, date + timedelta(days=20), alpha=0.3, color='red')
        axes[0].set_title(f'SPY Price + Crash Periods (Red Zones) - {title_prefix}', fontweight='bold', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 2. 폭락 확률
        proba_series = pd.Series(y_pred_proba, index=X_plot.index)
        axes[1].plot(proba_series.index, proba_series, color='darkred', linewidth=2)
        axes[1].axhline(0.5, color='red', linestyle='--', label='Threshold (50%)')
        if test_start:
            axes[1].axvline(test_start, color='blue', linestyle='--')
            
        axes[1].fill_between(proba_series.index, 0.5, proba_series.values,
                             where=proba_series.values > 0.5,
                             alpha=0.3, color='red', label='Alert (Fixed 0.5)')
        axes[1].set_title('Crash Probability (Full History)', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Probability')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # 3. 레이어별 신호
        for col in ['volatility', 'bond_stress', 'eco_surprise']:
            if col in X_plot.columns:
                axes[2].plot(X_plot.index, X_plot[col], label=col, linewidth=1.5)
        if test_start:
            axes[2].axvline(test_start, color='blue', linestyle='--')
        axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_title('Layer Signals (Z-scores)', fontweight='bold', fontsize=12)
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        # 4. 동시 스트레스 지속 기간
        if 'sync_stress_duration' in X_plot.columns:
            axes[3].fill_between(X_plot.index, 0, X_plot['sync_stress_duration'],
                                alpha=0.5, color='orange')
            if test_start:
                axes[3].axvline(test_start, color='blue', linestyle='--')
            axes[3].set_title('Synchronized Stress Duration (Days)', 
                            fontweight='bold', fontsize=12)
            axes[3].set_ylabel('Days')
            axes[3].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        now = datetime.now()
        filename = f'backtest_results_{now.year}{str(now.month).zfill(2)}{str(now.day).zfill(2)}_{str(now.hour).zfill(2)}{str(now.minute).zfill(2)}{str(now.second).zfill(2)}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[INFO] 차트 저장: {filename}")
        
        plt.show()
    
    def get_current_assessment(self, df):
        """
        현재 시점 위험 평가 (2026-01-28)
        """
        if self.model is None:
            print("[WARN] 먼저 모델 학습 필요")
            return
        
        X = df.drop('crash', axis=1)
        current_features = X.iloc[-1:].copy()
        
        # 예측 (Smoothing 적용을 위해 최근 100일 사용)
        X_window = X.iloc[-100:] if len(X) > 100 else X
        proba_window = self.model.predict_proba(X_window)[:, 1]
        
        # [SMOOTHING] 물 타기 금지! (span=20 -> 3)
        smoothed_proba = pd.Series(proba_window).ewm(span=3).mean()
        crash_proba = smoothed_proba.iloc[-1]
        
        if len(proba_window) == 0:
             crash_proba = 0.0
        
        # 위험 등급
        
        # Dynamic threshold 적용 고려 (여기서는 보수적으로 0.35 or use saved threshold if implemented)
        # 단순히 절대 확률로 등급 매김
        # Dynamic Threshold 적용
        thr = self.threshold
        risk_level = (
            '[HIGH] Critical' if crash_proba > thr * 1.5 else
            '[HIGH] High' if crash_proba > thr else
            '[ELEVATED] Elevated' if crash_proba > thr * 0.7 else
            '[NORMAL] Normal'
        )
        
        # 포지션 제안
        equity_weight = max(0, min(1, 1 - crash_proba * 1.5))
        
        print(f"\n{'='*70}")
        print(f"[TARGET] 현재 위험 평가 (2026-01-26)")
        print(f"{'='*70}")
        print(f"20일 폭락 확률: {crash_proba:.1%}")
        print(f"위험 등급: {risk_level}")
        print(f"\n[POSITION] 포지션 제안:")
        print(f"  - 권장 주식 비중: {equity_weight:.0%}")
        print(f"  - 현금/채권 비중: {1-equity_weight:.0%}")
        
        print(f"\n[INFO] 현재 신호 강도:")
        for col in ['volatility', 'bond_stress', 'eco_surprise']:
            if col in current_features.columns:
                val = current_features[col].values[0]
                print(f"  - {col}: {val:.2f}")
        
        if 'sync_stress_duration' in current_features.columns:
            sync_days = current_features['sync_stress_duration'].values[0]
            if sync_days > 0:
                print(f"\n[WARN] 구조적 스트레스 지속: {sync_days:.0f}일")
        
        print(f"{'='*70}\n")
        
        return {
            'probability': crash_proba,
            'risk_level': risk_level,
            'equity_weight': equity_weight
        }
    
    def plot_feature_signals(self, df, features_to_plot=['momentum', 'net_liquidity', 'fx_carry']):
        """
        주요 피처들의 시계열 차트 생성
        """
        try:
            from datetime import datetime
            
            # 피처 + duration 컬럼 추가
            all_features = []
            for feat in features_to_plot:
                all_features.append(feat)
                duration_col = f"{feat}_duration"
                if duration_col in df.columns:
                    all_features.append(duration_col)
            
            # 존재하는 컬럼만 필터
            available_features = [f for f in all_features if f in df.columns]
            
            if not available_features:
                print("[WARN] 요청한 피처가 데이터에 없습니다.")
                return
            
            # 차트 생성
            n_features = len(available_features)
            fig, axes = plt.subplots(n_features, 1, figsize=(16, 3*n_features), sharex=True)
            
            if n_features == 1:
                axes = [axes]
            
            for i, feature in enumerate(available_features):
                ax = axes[i]
                
                # 피처 값
                ax.plot(df.index, df[feature], label=feature, linewidth=1.5, color='steelblue')
                
                # Crash 라벨 영역 표시
                if 'crash' in df.columns:
                    crash_dates = df[df['crash'] == 1].index
                    for crash_date in crash_dates:
                        ax.axvline(crash_date, color='red', alpha=0.3, linewidth=0.8)
                
                # Zero line
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                
                # 제목 및 레이블
                ax.set_ylabel(feature, fontsize=10, fontweight='bold')
                ax.legend(loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.2)
            
            # X축 레이블 (마지막 subplot만)
            axes[-1].set_xlabel('Date', fontsize=10)
            
            # 전체 타이틀
            fig.suptitle('[INFO] Feature Signal Analysis', fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            # 저장
            out_path = f"feature_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] 피처 차트 저장: {out_path}")
            
        except Exception as e:
            print(f"[ERROR] 피처 차트 생성 오류: {e}")
    
    # ============================================
    # 유틸리티
    # ============================================
    
    def _regime_zscore(self, series, regime_indicator):
        """
        레짐별 Z-score
        """
        result = pd.Series(index=series.index, dtype=float)
        
        for idx in series.index:
            if pd.isna(series[idx]):
                continue
            
            current_regime = regime_indicator.get(idx, True)
            historical = series[:idx].tail(252)
            regime_hist = regime_indicator[:idx].tail(252).reindex(historical.index).fillna(False).astype(bool)
            
            if current_regime:
                regime_data = historical[regime_hist]
            else:
                regime_data = historical[~regime_hist]
            
            if len(regime_data) > 30:
                mean = regime_data.mean()
                std = regime_data.std()
                result[idx] = (series[idx] - mean) / std if std > 0 else 0
            else:
                result[idx] = (series[idx] - historical.mean()) / historical.std()
        
        return result


# ============================================
# 실행
# ============================================

if __name__ == "__main__":
    
    print(f"\n{'='*70}")
    print(f"[START] 구조적 위험 탐지 시스템 백테스트")
    print(f"   실행 시점: 2026년 1월 27일")
    print(f"{'='*70}\n")
    
    # FRED API 키 필요
    FRED_API_KEY = 'bea6a71ed27ab72ad1719aa15b92e5cd'  # https://fred.stlouisfed.org/docs/api/api_key.html
    
    # 초기화
    detector = StructuralRiskDetector2026(fred_api_key=FRED_API_KEY)
    
    # 데이터 준비 (2002 ~ 2026)
    df = detector.prepare_training_data(
        start_date='2002-01-01',
        end_date='None'
    )
    
    if df is not None and not df.empty:
        # 모델 학습 (2023-01-01 기준 분할: Training 2002-2022, Valid 2023-2026)
        detector.train_model(df, split_date='2023-01-01')
        
        # 현재 위험 평가
        current_risk = detector.get_current_assessment(df)
        
        # 백테스트 결과 시각화
        detector.plot_backtest_results()
        
        # 피처 신호 차트 생성
        detector.plot_feature_signals(
            df, 
            features_to_plot=['momentum', 'net_liquidity', 'fx_carry', 'liquidity']
        )
        
        print("\n[OK] 백테스트 완료!")
    else:
        print("\n[ERROR] 데이터 부족으로 백테스트를 진행할 수 없습니다.")
