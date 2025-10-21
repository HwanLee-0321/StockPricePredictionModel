#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

# 모든 컬럼을 다 보이게
pd.set_option('display.max_columns', None)

# 화면 너비에 맞춰 줄바꿈 없이 보기
pd.set_option('display.width', 200)

# 필요하면 각 컬럼 폭도 늘리기
pd.set_option('display.max_colwidth', None)

# 0) raw_data.csv에서 종목 코드 목록을 추출
#    low_memory=False로 mixed type 경고 억제
raw_df = pd.read_csv('raw_data.csv', usecols=['Ticker'], low_memory=False)
tickers = raw_df['Ticker'].dropna().astype(str).unique().tolist()

# 1) 파일 불러오기 (low_memory=False)
df = pd.read_csv('raw_data.csv', low_memory=False)

# 2) 날짜 컬럼 처리: datetime으로 변환 후 시간 제거
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.date
print("1-2단계 완료, 데이터 크기:", df.shape)

# 3) 전부 결측치인 컬럼 제거
drop_cols = ['PE Ratio', 'PB Ratio']
df.drop(columns=drop_cols, errors='ignore', inplace=True)
print("3단계 완료, 데이터 크기:", df.shape)

# ── 4) SMA_5/SMA_20 초기 결측치 구간 제거 ──
required_cols = [c for c in ['SMA_5', 'SMA_20'] if c in df.columns]
if required_cols:
    # 원본 NaN을 유지하기 위해 dropna는 건너뜁니다.
    # df = df.dropna(subset=required_cols, how='any')
    print(f"4단계 확인, {required_cols} 초기 결측 유지 → 데이터 크기: {df.shape}")
else:
    print("4단계: SMA_5/SMA_20 컬럼 없음, 건너뜀")

# ── 5) 나머지 결측치 보간 (interpolation) ──
# 모델에서 NaN 윈도우를 스킵하므로, 보간도 건너뛰고 그대로 둡니다.
# for col in ['Volatility', 'Change_pct']:
#     if col in df.columns:
#         df[col] = df[col].interpolate()
print("5단계: Volatility, Change_pct 결측 그대로 유지")

# ── 6) Dividend Yield 중앙값으로 대체 ──
# 학습 시 NaN을 건너뛰려면 fillna도 생략합니다.
# if 'Dividend Yield' in df.columns:
#     med = df['Dividend Yield'].median()
#     df['Dividend Yield'] = df['Dividend Yield'].fillna(med)
# elif 'Dividend_Yield' in df.columns:
#     med = df['Dividend_Yield'].median()
#     df['Dividend_Yield'] = df['Dividend_Yield'].fillna(med)
print("6단계: Dividend Yield 결측 그대로 유지")

# 7) 이상치 처리: Change_pct 클리핑
df.loc[:, 'Change_pct'] = df['Change_pct'].clip(-0.2, 0.2) if 'Change_pct' in df.columns else df.get('Change_pct')

# 8) 이상치 처리: Volume 로그 변환
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
df['Volume'] = df['Volume'].apply(lambda x: np.log1p(x) if x > 0 else 0)
print("7-8단계 이상치 처리 완료")

# 9) 최종 확인 및 저장
print("최종 데이터 크기:", df.shape)
print(df.head())
df.to_csv('preprocessed_train.csv', index=False)
print("파일 저장 완료: preprocessed_train.csv")

