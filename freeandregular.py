import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

# Windows용 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

# 파일 경로
file_path = r"C:\Users\Admin\Desktop\python_sql\data\installment.xlsx"

# 데이터 불러오기
df = pd.read_excel(file_path)

# Box Plot으로 적립방법별 최고금리 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='적립방법', y='최고금리', palette='pastel')
plt.title('적립방법에 따른 최고금리 분포', fontsize=16)
plt.xlabel('적립방법', fontsize=12)
plt.ylabel('최고금리 (%)', fontsize=12)
plt.show()

# Bar Plot으로 적립방법별 평균 최고금리 비교
plt.figure(figsize=(10, 6))
avg_interest = df.groupby('적립방법')['최고금리'].mean().reset_index()
sns.barplot(data=avg_interest, x='적립방법', y='최고금리', palette='muted')
plt.title('적립방법별 평균 최고금리', fontsize=16)
plt.xlabel('적립방법', fontsize=12)
plt.ylabel('평균 최고금리 (%)', fontsize=12)
plt.show()
