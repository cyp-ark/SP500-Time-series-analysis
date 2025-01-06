import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Matplotlib에서 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows (맑은 고딕)
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS (애플고딕)
# plt.rcParams['font.family'] = 'NanumGothic'  # Linux (나눔고딕)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. 데이터 로드
ticker = "AAPL"
data = yf.download(ticker, start="2006-01-01", end="2024-01-01")[['Close']]

# 데이터 인덱스 설정
data = data.reset_index()
data.set_index('Date', inplace=True)

# 데이터 확인
print(data.head())

# 2. 시계열 데이터 시각화
plt.figure(figsize=(12, 6))
plt.plot(data, label='종가')
plt.title(f"{ticker} 주가 시계열 데이터")
plt.xlabel("날짜")
plt.ylabel("종가")
plt.legend()
plt.grid()
plt.show()

# 3. 로그 변환 데이터 시각화
data_log = np.log(data['Close'])

plt.figure(figsize=(12, 6))
plt.plot(data_log, label='로그 변환 데이터', color='orange')
plt.title(f"{ticker} 주가 로그 변환 데이터")
plt.xlabel("날짜")
plt.ylabel("로그 변환 값")
plt.legend()
plt.grid()
plt.show()

# 4. 로그 변환 데이터에 대한 ADF 테스트
adf_test_log = adfuller(data_log.dropna())
print("\n=== 로그 변환 데이터 ADF 테스트 ===")
print("ADF 통계량:", adf_test_log[0])
print("p-값:", adf_test_log[1])
print("기준값:", adf_test_log[4])
if adf_test_log[1] <= 0.05:
    print("로그 변환 데이터는 정상성(stationary)을 만족합니다.")
else:
    print("로그 변환 데이터는 정상성이 없습니다.")

# 5. 로그 변환 데이터 ACF/PACF 분석
plt.figure(figsize=(12, 6))
plot_acf(data_log.dropna(), lags=30, title="로그 변환 데이터 ACF")
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(data_log.dropna(), lags=30, title="로그 변환 데이터 PACF")
plt.show()


# 9. 원본 데이터에 대한 1차 차분
data_diff = data['Close'].diff().dropna()

plt.figure(figsize=(12, 6))
plt.plot(data_diff, label='1차 차분 데이터', color='blue')
plt.title(f"{ticker} 1차 차분 데이터")
plt.xlabel("날짜")
plt.ylabel("차분 값")
plt.legend()
plt.grid()
plt.show()

# 10. 1차 차분 데이터 ADF 테스트
adf_test_diff = adfuller(data_diff)
print("\n=== 1차 차분 데이터 ADF 테스트 ===")
print("ADF 통계량:", adf_test_diff[0])
print("p-값:", adf_test_diff[1])
print("기준값:", adf_test_diff[4])
if adf_test_diff[1] <= 0.05:
    print("1차 차분 데이터는 정상성(stationary)을 만족합니다.")
else:
    print("1차 차분 데이터는 정상성이 없습니다.")

# 11. 1차 차분 데이터 ACF/PACF 분석
plt.figure(figsize=(12, 6))
plot_acf(data_diff, lags=30, title="1차 차분 데이터 ACF")
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(data_diff, lags=30, title="1차 차분 데이터 PACF")
plt.show()
