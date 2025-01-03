import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data(stock):
    """데이터를 로드하고 전처리합니다."""
    data = yf.download(stock)
    data.columns = [' '.join(col)[:-(len(stock)+1)] for col in data.columns]
    data.reset_index(inplace=True)
    
    data['Date'] = pd.to_datetime(data['Date'])
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_60'] = data['Close'].rolling(window=60).mean()
    data['SMA_120'] = data['Close'].rolling(window=120).mean()
    
    # Calculate RSI and add it to the data
    data['RSI'] = calculate_rsi(data)
    return data

def calculate_rsi(data, period=14):
    """RSI 계산 함수"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def filter_data_by_date(data, start_date, end_date):
    """주어진 날짜 범위에 맞게 데이터를 필터링합니다."""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

def create_plot(data, show_sma_5, show_sma_20, show_sma_60, show_sma_120):
    """Close 가격, SMA, Volume을 포함한 Plotly 서브플롯 그래프를 생성합니다."""
    
    # 2x1 서브플롯 생성
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.1)

    # Close 가격 라인 추가
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'], mode='lines', name='Close Price',
        line=dict(color='blue', width=1.5)
    ), row=1, col=1)

    # 각 SMA 체크박스에 따라 라인 추가
    if show_sma_5:
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['SMA_5'], mode='lines', name='5-Day SMA',
            line=dict(color='#DD552B', width=1)
        ), row=1, col=1)
    if show_sma_20:
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['SMA_20'], mode='lines', name='20-Day SMA',
            line=dict(color='#BA69D1', width=1)
        ), row=1, col=1)
    if show_sma_60:
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['SMA_60'], mode='lines', name='60-Day SMA',
            line=dict(color='#6E9BD9', width=1)
        ), row=1, col=1)
    if show_sma_120:
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['SMA_120'], mode='lines', name='120-Day SMA',
            line=dict(color='#9BBB6B', width=1)
        ), row=1, col=1)
        
    # Volume 컬러 설정
    volume_colors = ['#5174E3']  # 첫 번째 값은 기본적으로 파란색
    for i in range(1, len(data)):
        if data['Volume'].iloc[i] > data['Volume'].iloc[i-1]:
            volume_colors.append('#E05659')  # 이전 값보다 크면 빨간색
        else:
            volume_colors.append('#5174E3')  # 이전 값보다 작으면 파란색

    # Volume 바플롯 추가
    fig.add_trace(go.Bar(
        x=data['Date'], y=data['Volume'], name='Volume',
        marker=dict(color=volume_colors)
    ), row=2, col=1)
    
    # RSI 라인 추가
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['RSI'], mode='lines', name='RSI',
        line=dict(color='#EB6239', width=1)
    ), row=3, col=1)
    
    # RSI의 30, 70 수준을 나타내는 수평선 추가
    fig.add_shape(
        type="line",
        x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=30, y1=30,
        line=dict(color="gray", width=1),
        row=3, col=1
    )
    fig.add_shape(
        type="line",
        x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=70, y1=70,
        line=dict(color="gray", width=1),
        row=3, col=1
    )

    # 레이아웃 설정
    fig.update_layout(
        height=600, width=800,
        
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Volume'),
        xaxis3=dict(title='Date'),  # 하단 x축
        yaxis3=dict(title='RSI',
                    tickvals=[30, 70]),  # y축 설정 (RSI)
    )

    return fig


def main():
    """Streamlit 앱의 메인 함수입니다."""
    st.title("S&P 500 Visualization with Moving Averages")
    
    stock = st.text_input("주식 코드를 입력하세요", value='^GSPC')
    
    if stock is not None:
        data = load_data(stock)

        # 날짜 범위 필터링
        max_date = data['Date'].max()
        start_date = max_date - pd.DateOffset(years=3)
        end_date = data['Date'].max()
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        with col1:
            if st.button('최근 1달'):
                start_date = max_date - pd.DateOffset(months=1)
                end_date = data['Date'].max()
        with col2:
            if st.button('최근 3달'):
                start_date = max_date - pd.DateOffset(months=3)
                end_date = data['Date'].max()
        with col3:
            if st.button('최근 6달'):
                start_date = max_date - pd.DateOffset(months=6)
                end_date = data['Date'].max()
        with col4:
            if st.button('최근 1년'):
                start_date = max_date - pd.DateOffset(years=1)
                end_date = data['Date'].max()
        with col5:
            if st.button('최근 3년'):
                start_date = max_date - pd.DateOffset(years=3)
                end_date = data['Date'].max()
        with col6:
            if st.button('최근 5년'):
                start_date = max_date - pd.DateOffset(years=5)
                end_date = data['Date'].max()
        with col7:
            with st.popover('직접 선택'):
                col1, col2, col3 = st.columns(3)
                with col1 : sday = st.date_input("시작 날짜")
                with col2 : eday = st.date_input("종료 날짜")
                with col3 :
                    if st.button("적용"):
                        start_date = sday
                        end_date = eday
        

        if start_date > end_date:
            st.error("오류: 시작 날짜는 종료 날짜보다 앞서야 합니다.")
        else:
            filtered_data = filter_data_by_date(data, start_date, end_date)
            if filtered_data.empty:
                st.warning("선택한 날짜 범위에 데이터가 없습니다.")
            else:
                # 각 SMA에 대한 체크박스 추가
                col1, col2, col3, col4 = st.columns(4)
                with col1 : show_sma_5 = st.checkbox("5일 단순이동평균", value=False)
                with col2 : show_sma_20 = st.checkbox("20일 단순이동평균", value=False)
                with col3 : show_sma_60 = st.checkbox("60일 단순이동평균", value=False)
                with col4 : show_sma_120 = st.checkbox("120일 단순이동평균", value=False)
                
                fig = create_plot(filtered_data, show_sma_5, show_sma_20, show_sma_60, show_sma_120)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("CSV 파일을 업로드하여 시각화할 수 있습니다.")

if __name__ == "__main__":
    main()
