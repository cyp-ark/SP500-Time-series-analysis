import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import openai

# OpenAI API 키 설정
openai.api_key = "sk-proj-HdnXCBBy1vYyyjD6x6R5bRCgcSiAposccP6xiQyvbu-RGpSFiaT_5MAkR8rzDA1bLdR1r8EPbzT3BlbkFJwyy5uxSYoo-wVtAUw7e51l_JL9840GExLAKLr1oSTuSGS7psOgGvpaJ3MoH5BQnprpT"

# 상태 초기화
if "session" not in st.session_state:
    st.session_state["session"] = "EDA_N"
if "start_date" not in st.session_state:
    st.session_state["start_date"] = None
if "end_date" not in st.session_state: 
    st.session_state["end_date"] = None


def eda_page_streamlit():
    """EDA 페이지 구현"""
    st.title("Exploratory Data Analysis (EDA)")

    # 데이터 로드
    stock_code = st.text_input("분석할 주식 코드를 입력하세요", value="^GSPC")
    try:
        data = yf.download(stock_code).reset_index()
        if data.empty:
            st.warning("선택한 주식 코드에 해당하는 데이터가 없습니다.")
            return
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류 발생: {e}")
        return

    st.subheader("기간 선택")

    max_date = data['Date'].max()
    if "start_date" not in st.session_state or st.session_state["start_date"] is None:
        st.session_state["start_date"] = max_date - pd.DateOffset(years=3)
    if "end_date" not in st.session_state or st.session_state["end_date"] is None:
        st.session_state["end_date"] = max_date

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        if st.button("최근 1달"):
            st.session_state["start_date"] = max_date - pd.DateOffset(months=1)
            st.session_state["end_date"] = max_date
    with col2:
        if st.button("최근 3달"):
            st.session_state["start_date"] = max_date - pd.DateOffset(months=3)
            st.session_state["end_date"] = max_date
    with col3:
        if st.button("최근 6달"):
            st.session_state["start_date"] = max_date - pd.DateOffset(months=6)
            st.session_state["end_date"] = max_date
    with col4:
        if st.button("최근 1년"):
            st.session_state["start_date"] = max_date - pd.DateOffset(years=1)
            st.session_state["end_date"] = max_date
    with col5:
        if st.button("최근 3년"):
            st.session_state["start_date"] = max_date - pd.DateOffset(years=3)
            st.session_state["end_date"] = max_date
    with col6:
        if st.button("최근 5년"):
            st.session_state["start_date"] = max_date - pd.DateOffset(years=5)
            st.session_state["end_date"] = max_date
    with col7:
        with st.expander("직접 선택"):
            col1, col2 = st.columns(2)
            with col1:
                sday = st.date_input("시작 날짜", value=st.session_state["start_date"])
            with col2:
                eday = st.date_input("종료 날짜", value=st.session_state["end_date"])
            if st.button("적용"):
                st.session_state["start_date"] = sday
                st.session_state["end_date"] = eday

    start_date = st.session_state["start_date"]
    end_date = st.session_state["end_date"]
 
    if start_date > end_date:
        st.error("오류: 시작 날짜는 종료 날짜보다 앞서야 합니다.")
    else:
        filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]
        if filtered_data.empty:
            st.warning("선택한 날짜 범위에 데이터가 없습니다.")
        else:
            st.subheader("시계열 분해")
            decomposition = decompose_and_plot(filtered_data)

            st.subheader("통계적 검정 결과")
            trend_stats = statistical_tests(decomposition.trend, "Trend")
            seasonal_stats = statistical_tests(decomposition.seasonal, "Seasonality")
            resid_stats = statistical_tests(decomposition.resid, "Residual")

            st.markdown(trend_stats)
            st.markdown(seasonal_stats) 
            st.markdown(resid_stats)

            st.subheader("AI 해석 (LangChain)")
            if st.button("LangChain 기반 해석 요청"):
                interpretation = interpret_with_langchain(
                    decomposition.trend.dropna(), 
                    decomposition.seasonal.dropna(),
                    decomposition.resid.dropna()
                )
                st.write(interpretation)


def decompose_and_plot(data):
    """시계열 데이터를 분해하고 결과를 개별 그래프로 시각화합니다."""
    decomposition = seasonal_decompose(data['Close'], model='additive', period=20)

    # Trend (추세) 그래프
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=data['Date'], y=decomposition.trend, mode='lines', name='Trend',
        line=dict(color='blue')
    ))
    fig_trend.update_layout(title="Trend (추세)", xaxis_title="Date", yaxis_title="Value", height=400, width=900)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Seasonality (계절성) 그래프
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(
        x=data['Date'], y=decomposition.seasonal, mode='lines', name='Seasonality',
        line=dict(color='green')
    ))
    fig_seasonal.update_layout(title="Seasonality (계절성)", xaxis_title="Date", yaxis_title="Value", height=400, width=900)
    st.plotly_chart(fig_seasonal, use_container_width=True)

    # Residual (잔차) 그래프
    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(
        x=data['Date'], y=decomposition.resid, mode='lines', name='Residual',
        line=dict(color='red')
    ))
    fig_residual.update_layout(title="Residual (잔차)", xaxis_title="Date", yaxis_title="Value", height=400, width=900)
    st.plotly_chart(fig_residual, use_container_width=True)

    return decomposition

def statistical_tests(series, name):
    """시계열 데이터에 통계적 검정을 적용합니다."""
    adf_result = adfuller(series.dropna())
    adf_pvalue = adf_result[1]

    ljungbox_result = acorr_ljungbox(series.dropna(), lags=[10], return_df=True)
    lb_pvalue = ljungbox_result['lb_pvalue'].values[0]

    result = (
        f"**{name}**:\n"
        f"- ADF Test p-value: {adf_pvalue:.4f} ({'Stationary' if adf_pvalue < 0.05 else 'Non-Stationary'})\n"
        f"- Ljung-Box Test p-value: {lb_pvalue:.4f} ({'Random Residuals' if lb_pvalue >= 0.05 else 'Non-Random Residuals'})"
    )

    return result

def interpret_with_langchain(trend, seasonal, resid):
    """LangChain을 사용하여 시계열 결과를 해석합니다."""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)

    prompt = PromptTemplate(
        input_variables=["trend", "seasonal", "residual"],
        template="""
        아래는 시계열 데이터를 분해한 결과입니다. 각각의 구성 요소를 해석하고 중요한 통찰력을 제공합니다:
        1. Trend 데이터 요약: {trend}
        2. Seasonality 데이터 요약: {seasonal}
        3. Residual 데이터 요약: {residual}

        해석:
        - 추세(Trend)의 의미와 방향
        - 계절성(Seasonality)의 패턴
        - 잔차(Residual)의 무작위성 및 이상치 여부
        """)

    chain = prompt | llm
    
    response = chain.run({
        "trend": trend.describe().to_string(),
        "seasonal": seasonal.describe().to_string(),
        "residual": resid.describe().to_string()
    })

    return response

def sidebar_navigation():
    st.sidebar.title("Navigation")
    if st.sidebar.button("EDA"):
        st.session_state["session"] = "EDA_N"
    if st.sidebar.button("Stock"):
        st.session_state["session"] = "stock_N"

def run():
    st.set_page_config(layout="wide")

    sidebar_navigation()

    if st.session_state["session"] == "EDA_N":
        eda_page_streamlit()
    else:
        st.write("Stock Page")

if __name__ == "__main__":
    run()
