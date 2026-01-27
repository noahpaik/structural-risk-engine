import yfinance as yf
import pandas as pd
import numpy as np

# 2024년 7-9월 SPY 데이터 확인
spy_data = yf.download('SPY', start='2024-07-01', end='2024-09-30', progress=False)

# MultiIndex 처리
if isinstance(spy_data.columns, pd.MultiIndex):
    spy_data.columns = spy_data.columns.get_level_values(0)

spy_data.index = spy_data.index.tz_localize(None)
spy_close = spy_data['Close']

# 일별 수익률
returns = spy_close.pct_change()

# 향후 20일 최대 낙폭
future_dd_20 = returns.rolling(20).apply(
    lambda x: (1 + x).cumprod().min() - 1
).shift(-20)

print("="*70)
print("2024년 8월 Black Monday 진단")
print("="*70)
print(f"\n8월 5일 전후 데이터:\n")
print(f"{'Date':<12} {'Close':>8} {'Daily%':>8} {'20D_DD%':>9}")
print("-"*42)

for date in pd.date_range('2024-08-01', '2024-08-15'):
    if date in spy_close.index:
        close = spy_close.loc[date]
        daily = returns.loc[date] * 100 if date in returns.index else np.nan
        dd20 = future_dd_20.loc[date] * 100 if date in future_dd_20.index else np.nan
        print(f"{date.date()} {close:8.2f} {daily:8.2f} {dd20:9.2f}")

# VIX 확인
vix_data = yf.download('^VIX', start='2024-07-01', end='2024-09-30', progress=False)
if isinstance(vix_data.columns, pd.MultiIndex):
    vix_data.columns = vix_data.columns.get_level_values(0)
vix_data.index = vix_data.index.tz_localize(None)
vix_close = vix_data['Close']
vix_ma20 = vix_close.rolling(20).mean()

print(f"\n\nVIX 데이터 (8월 5일 전후):")
print(f"{'Date':<12} {'VIX':>8} {'MA20':>8} {'Spike':>6}")
print("-"*38)

for date in pd.date_range('2024-08-01', '2024-08-15'):
   if date in vix_close.index:
        vix = vix_close.loc[date]
        ma = vix_ma20.loc[date] if date in vix_ma20.index else np.nan
        spike = "YES" if vix > ma * 1.5 else "NO"
        print(f"{date.date()} {vix:8.2f} {ma:8.2f} {spike:>6}")

print("\n="*70)
print("결론: 8월 5일이 '폭락 라벨'을 받았는지 확인")
print("="*70)

if pd.Timestamp('2024-08-05') in future_dd_20.index:
    dd_val = future_dd_20.loc['2024-08-05']
    print(f"조건 1: 향후 20일 최대 낙폭 = {dd_val*100:.2f}% (기준: -10% 이하)")
    print(f"        -> {'통과' if dd_val < -0.10 else '실패 (V자 반등으로 20일 통산 약함)'}")
    
    vix_val = vix_close.loc['2024-08-05']
    vix_ma = vix_ma20.loc['2024-08-05']
    print(f"\n조건 2: VIX Spike = {vix_val:.1f} (MA20: {vix_ma:.1f}, 기준: {vix_ma*1.5:.1f})")
    print(f"        -> {'통과' if vix_val > vix_ma * 1.5 else '실패'}")
