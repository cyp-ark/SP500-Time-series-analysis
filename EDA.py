import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

def load_data(stock):
    data = yf.download(stock)
    data.columns = [' '.join(col)[:-(len(stock)+1)] for col in data.columns]
    data.reset_index(inplace=True)
    
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def log_scale(data):
    for col in data.columns[1:]:
        data['log_'+col] = np.log1p(data[col])
    return data

def show_stock_name(symbol):
    try:
        stock_name = yf.Ticker(symbol).info['longName']
        st.markdown(f"""
            <h1>
            EDA of {stock_name}
            </h1>
            """, unsafe_allow_html=True)
    except KeyError:
        return "종목 코드가 잘못되었거나 이름을 찾을 수 없습니다."
    
def show_data_description(data):
    description = data.describe()
    st.table(description)
    
def show_data_histogram(data):
    name_columns = data.select_dtypes(include=[np.number]).columns
    num_columns = len(name_columns)
    # 서브플롯 생성
    fig = make_subplots(
        rows=1, cols=num_columns,  # 열의 수에 맞춰 서브플롯 생성
        subplot_titles=name_columns,  # 각 서브플롯 제목 설정
        shared_yaxes=True  # y축을 공유하여 시각화
    )

    # 각 열에 대해 히스토그램 추가
    for i, column in enumerate(name_columns):
        fig.add_trace(
            go.Histogram(x=data[column].dropna(), name=column),  # dropna()로 결측치 처리
            row=1, col=i+1  # 1행, 각 열에 해당하는 위치에 추가
        )
    
    return fig

def show_decomposition(data, col, log):
    if log: col = 'log_'+col
    else: col = col
        
    decomposition = seasonal_decompose(data[col], model='additive', period=12)
    
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    residual = decomposition.resid.dropna()
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[col], mode='lines', name='Close'), row=1, col=1)
    
    # 트렌드 서브플롯
    fig.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend'),
                row=2, col=1)

    # 계절성 서브플롯
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonal'),
                row=3, col=1)

    # 잔차 서브플롯
    fig.add_trace(go.Scatter(x=residual.index, y=residual, mode='lines', name='Residual'),
                row=4, col=1)
    
    fig.update_layout(
        yaxis=dict(title=col),
        yaxis2=dict(title='Trend'),
        yaxis3=dict(title='Seasonal'),
        yaxis4=dict(title='Residual'),
    )
    
    return fig, data[col], trend, seasonal, residual

def main():
    stock = st.sidebar.text_input("주식 코드를 입력하세요", value='^GSPC')
    
    if stock is not None:
        show_stock_name(stock)
        st.divider()
        
        st.subheader('Raw Data')
        data = load_data(stock)
        
        ################
        # 날짜 범위 선택
        ################
        
        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()
        # 날짜 범위 슬라이더
        
        date_range = st.sidebar.slider(
            'Select a date range',
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),  # 기본값: 전체 범위
            format="YYYY-MM-DD"
        )
        filtered_data = data[
            (data['Date'] >= pd.to_datetime(date_range[0])) &
            (data['Date'] <= pd.to_datetime(date_range[1]))
        ]
        
        ################
        ### show data
        ################
        
        st.dataframe(filtered_data, use_container_width=True)
        
        ################
        ### show data description
        ################
        
        #st.subheader('Data Description')
        #show_data_description(filtered_data)
        
        st.subheader('Data Histogram')
        fig = show_data_histogram(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
        
        filtered_data = log_scale(filtered_data)
        st.subheader('Time-series decomposition and visualization')
        
        col1, col2 = st.columns(2)
        with col1:
            col = st.selectbox('Column', filtered_data.columns[1:6], index=0)
        with col2:
            log = st.checkbox('Log Scale', value=False)
        
        fig, signal, trend, seasonal, residual = show_decomposition(filtered_data,col,log=log)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader('Stationary Test')
        
        
        

if __name__ == "__main__":
    main()