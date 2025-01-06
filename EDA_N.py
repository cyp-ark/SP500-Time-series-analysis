import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import openai

# OpenAI API 키 설정
openai.api_key = "your-api-key"

def sidebar_navigation():
    st.sidebar.title('Navigation')
    if st.sidebar.button('EDA'):
        st.session_state["session"] = "EDA_N"
    if st.sidebar.button('Stock'):
        st.session_state["session"] = "stock_N"

def decompose_and_plot(data):
    """시계열 데이터를 분해하고 결과를 Plotly로 시각화합니다."""
    decomposition = seasonal_decompose(data['Close'], model='additive', period=20)

    # 3x1 서브플롯 생성
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        row_heights=[0.3, 0.3, 0.3],
                        vertical_spacing=0.05)

    # Trend (추세)
    fig.add_trace(go.Scatter(
        x=data['Date'], y=decomposition.trend, mode='lines', name='Trend',
        line=dict(color='blue')
    ), row=1, col=1)

    # Seasonality (계절성)
    fig.add_trace(go.Scatter(
        x=data['Date'], y=decomposition.seasonal, mode='lines', name='Seasonality',
        line=dict(color='green')
    ), row=2, col=1)

    # Residual (잔차)
    fig.add_trace(go.Scatter(
        x=data['Date'], y=decomposition.resid, mode='lines', name='Residual',
        line=dict(color='red')
    ), row=3, col=1)

    # 레이아웃 업데이트
    fig.update_layout(
        height=800, width=900,
        title='Time Series Decomposition',
        xaxis_title='Date',
        yaxis_title='Value',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    return decomposition

def statistical_tests(series, name):
    """시계열 데이터에 통계적 검정을 적용합니다."""
    # ADF 검정
    adf_result = adfuller(series.dropna())
    adf_pvalue = adf_result[1]

    # Ljung-Box 테스트
    ljungbox_result = acorr_ljungbox(series.dropna(), lags=[10], return_df=True)
    lb_pvalue = ljungbox_result['lb_pvalue'].values[0]

    result = (
        f"**{name}**:\n"
        f"- ADF Test p-value: {adf_pvalue:.4f} ({'Stationary' if adf_pvalue < 0.05 else 'Non-Stationary'})\n"
        f"- Ljung-Box Test p-value: {lb_pvalue:.4f} ({'Random Residuals' if lb_pvalue >= 0.05 else 'Non-Random Residuals'})"
    )

    return result

def interpret_results(trend, seasonal, resid):
    """OpenAI API를 사용하여 시계열 분해 결과와 통계적 검정 결과를 해석합니다."""
    prompt = (
        "다음은 시계열 데이터를 분해한 결과와 통계적 검정 결과입니다. 데이터를 해석해 주세요:\n"
        f"1. Trend: \n{trend.describe().to_string()}\n"
        f"2. Seasonality: \n{seasonal.describe().to_string()}\n"
        f"3. Residual: \n{resid.describe().to_string()}"
    )

    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=300
    )

    return response['choices'][0]['text']

def eda_page():
    """EDA 페이지 구현"""
    st.title("Exploratory Data Analysis (EDA)")

    # 데이터 로드
    stock_code = st.text_input("분석할 주식 코드를 입력하세요", value="^GSPC")
    data = yf.download(stock_code, start="2020-01-01", end="2023-01-01").reset_index()

    if not data.empty:
        st.subheader("시계열 분해")

        # 시계열 분해 및 시각화
        decomposition = decompose_and_plot(data)

        st.subheader("통계적 검정 결과")
        trend_stats = statistical_tests(decomposition.trend, "Trend")
        seasonal_stats = statistical_tests(decomposition.seasonal, "Seasonality")
        resid_stats = statistical_tests(decomposition.resid, "Residual")

        st.markdown(trend_stats)
        st.markdown(seasonal_stats)
        st.markdown(resid_stats)

        st.subheader("AI 해석")
        if st.button("시계열 결과 해석 요청"):
            interpretation = interpret_results(
                decomposition.trend.dropna(),
                decomposition.seasonal.dropna(),
                decomposition.resid.dropna()
            )
            st.write(interpretation)
    else:
        st.warning("데이터를 불러올 수 없습니다. 올바른 주식 코드를 입력하세요.")

def run():
    st.set_page_config(layout='wide')

    # 사이드바 네비게이션
    sidebar_navigation()

    if st.session_state.get("session", "Stock") == "stock_N":
        st.write("Stock Page")
    elif st.session_state.get("session", "Stock") == "EDA_N":
        eda_page()

if __name__ == "__main__":
    run()