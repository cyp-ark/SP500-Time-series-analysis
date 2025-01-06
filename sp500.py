import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Matplotlib에서 한국어 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows (맑은 고딕)
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS
# plt.rcParams['font.family'] = 'NanumGothic'  # Linux (나눔고딕)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 영어 섹터 이름과 한국어 섹터 이름 매핑
sector_names = {
    "XLK": "정보 기술",
    "XLF": "금융",
    "XLV": "헬스케어",
    "XLY": "자유 소비재",
    "XLE": "에너지",
    "XLI": "산업재",
    "XLB": "소재",
    "XLU": "유틸리티",
    "XLC": "통신 서비스",
    "XLRE": "부동산"
}

# S&P 500 섹터별 ETF 리스트
etfs = list(sector_names.keys())
data = {}

# ETF 데이터 다운로드 및 저장
for etf in etfs:
    etf_data = yf.download(etf, start="2010-01-01", end="2025-01-01")['Close']
    if etf_data.empty:
        print(f"No data available for {etf}")
        continue
    data[etf] = etf_data  # ETF 데이터를 딕셔너리에 추가

# 데이터프레임으로 병합
sector_data = pd.concat(data.values(), axis=1, keys=data.keys())

# 섹터 이름을 한국어로 변환
sector_data.rename(columns=sector_names, inplace=True)

# 결과 확인
print(sector_data.head())

# 결측치 처리
sector_data.fillna(method='ffill', inplace=True)
sector_data.fillna(method='bfill', inplace=True)

# 수익률 계산
returns = sector_data.pct_change().dropna()
print(returns.head())

# 데이터 시각화
sector_data.plot(figsize=(12, 6))
plt.title("S&P 500 섹터별 ETF 성과")
plt.xlabel("날짜")
plt.ylabel("ETF 가격")
plt.legend(title="섹터")
plt.show()
