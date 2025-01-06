import streamlit as st
import stock
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import openai

# OpenAI API 키 설정
openai.api_key = "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"

def sidebar_navigation():
    st.sidebar.title('Navigation')
    if st.sidebar.button('EDA'):
        st.session_state["session"] = "EDA_N"
    if st.sidebar.button('Stock'):  
        st.session_state["session"] = "stock_N" 
 
def decompose_and_plot(data): 
    """시계열 데이터를 분해하고 결과를 Plotly로 시각화합니다."""
    decomposition = seasonal_decompose(data['Close'], model='additive', period=20)

    # 시각화
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['Date'], y=decomposition.trend, name='Trend', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=decomposition.seasonal, name='Seasonality', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data['Date'], y=decomposition.resid, name='Residual', line=dict(color='red')))

    fig.update_layout(title='Time Series Decomposition', xaxis_title='Date', yaxis_title='Value', height=600)
    st.plotly_chart(fig, use_container_width=True)

    return decomposition


def interpret_results(trend, seasonal, resid):
    """OpenAI API를 사용하여 시계열 분해 결과를 해석합니다."""
    prompt = (
        "다음은 시계열 데이터를 분해한 결과입니다. 이를 바탕으로 데이터를 해석해 주세요.\n"
        "1. Trend: \n" + trend.describe().to_string() + "\n"
        "2. Seasonality: \n" + seasonal.describe().to_string() + "\n"
        "3. Residual: \n" + resid.describe().to_string()
    )

    # 최신 ChatCompletion 방식 사용
    response = openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 유능한 증권 데이터 분석가이자 통계학 전문가이자 교육자입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300 
    )

    return response['choices'][0]['message']['content'] 


def eda_page():
    """EDA 페이지 구현"""
    st.title("Exploratory Data Analysis (EDA)")

    # 데이터 로드
    stock_code = st.text_input("분석할 주식 코드를 입력하세요", value="^GSPC")
    data = stock.load_data(stock_code)

    if data is not None:
        st.subheader("시계열 분해")

        # 시계열 분해 및 시각화
        decomposition = decompose_and_plot(data)

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
        stock.main()
    elif st.session_state.get("session", "Stock") == "EDA_N":
        eda_page()

if __name__ == "__main__":
    run()    