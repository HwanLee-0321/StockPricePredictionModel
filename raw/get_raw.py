#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import yfinance as yf

# 사용할 티커 리스트
tickers = [
    "000270.KS", "000660.KS", "000810.KS", "005380.KS", "005490.KS",
    "005930.KS", "005935.KS", "009540.KS", "010130.KS", "011200.KS",
    "012330.KS", "012450.KS", "015760.KS", "028260.KS", "032830.KS",
    "033780.KS", "034020.KS", "035420.KS", "035720.KS", "042660.KS",
    "051910.KS", "055550.KS", "068270.KS", "086790.KS", "105560.KS",
    "138040.KS", "207940.KS", "259960.KS", "329180.KS", "373220.KS"
]

# 1) 다운로드 시 auto_adjust를 명시적으로 설정 (기본이 바뀌었으니 경고 안 나게)
fetch_kwargs = {
    'period':       'max',
    'progress':     False,
    'auto_adjust':  False   # 혹은 True, 원하시는 대로 명시적으로 지정
}

df_list = []
for ticker in tickers:
    try:
        df = yf.download(ticker, **fetch_kwargs)
        df.reset_index(inplace=True)
        df['Ticker'] = ticker

        info = yf.Ticker(ticker).info
        df['Name'] = info.get('longName', '')

        df['52_Week_High'] = df['Close'].rolling(252, min_periods=1).max()
        df['52_Week_Low']  = df['Close'].rolling(252, min_periods=1).min()
        df['SMA_5']        = df['Close'].rolling(5,   min_periods=1).mean()
        df['SMA_20']       = df['Close'].rolling(20,  min_periods=1).mean()
        df['Return']       = df['Close'].pct_change().fillna(0)
        df['Volatility']   = df['Return'].rolling(20, min_periods=1).std().fillna(0)
        df['Change_pct']   = df['Return'] * 100

        df['Market_Cap']     = info.get('marketCap',     np.nan)
        df['PER']            = info.get('trailingPE',    np.nan)
        df['PBR']            = info.get('priceToBook',   np.nan)
        df['Dividend_Yield'] = info.get('dividendYield', np.nan)
        df['Beta']           = info.get('beta',          np.nan)

        df_list.append(df)
    except Exception as e:
        print(f"[경고] {ticker} 처리 중 에러: {e}")

# 2) concat 후 NaN 채우기 — group_keys=False 로 multi-index·성능 경고 해소
raw_data = pd.concat(df_list, ignore_index=True)
raw_data.sort_values(['Ticker','Date'], inplace=True)

# raw_data = (
#     raw_data
#     .groupby('Ticker', group_keys=False)
#     .apply(lambda grp: grp.ffill().bfill())
#     .reset_index(drop=True)
# )

raw_data.to_csv('raw_data.csv', index=False)
print("raw_data.csv 생성 완료 (경고 없이 저장)") 

