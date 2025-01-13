##########################################
### Stock 파트(주가 창 시각화) 구현 ###
##########################################


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
    
    data['SMA_20_STD'] = data['Close'].rolling(window=20).std()
    
    data['Upper Band'] = data['SMA_20'] + (data['SMA_20_STD'] * 2)
    data['Lower Band'] = data['SMA_20'] - (data['SMA_20_STD'] * 2)
    
    # Calculate RSI and add it to the data
    data['RSI'] = calculate_rsi(data)
    data['RSI_SMA'] = data['RSI'].rolling(window=5).mean()
    
    data['Cross'] = 0  # 교차 여부를 저장하는 열 (1: 상승, -1: 하락)
    for i in range(1, len(data)):
        if data['RSI'].iloc[i] > data['RSI_SMA'].iloc[i] and data['RSI'].iloc[i-1] <= data['RSI_SMA'].iloc[i-1]:
            data.loc[i, 'Cross'] = 1  # 상승 교차
        elif data['RSI'].iloc[i] < data['RSI_SMA'].iloc[i] and data['RSI'].iloc[i-1] >= data['RSI_SMA'].iloc[i-1]:
            data.loc[i, 'Cross'] = -1  # 하락 교차
    
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

def show_stock_name(symbol):
    try:
        stock_name = yf.Ticker(symbol).info['longName']
        st.markdown(f"""
            <h1>
            {stock_name}
            </h1>
            """, unsafe_allow_html=True)
    except KeyError:
        return "종목 코드가 잘못되었거나 이름을 찾을 수 없습니다."

def show_stock_price(data):
    yesterday_close = data['Close'].iloc[-2]
    today_close = data['Close'].iloc[-1]
    change = today_close - yesterday_close
    change_percent = (change / yesterday_close) * 100

    color = 'red' if change > 0 else 'blue' if change < 0 else 'black'
    symbol = '▲' if change > 0 else '▼' if change < 0 else ''
    
    st.markdown(f"""
            <div style="display: flex; align-items: baseline;">
                <h1 style='color:{color}; display:inline;'> {data['Close'].iloc[-1]:.2f} </h1>
                <h3 style='color:{color}; display:inline;'> {symbol} {abs(change):.2f} </h3>
                <h3 style='color:{color}; display:inline;'> {change_percent:.2f}% </h3>
            </div>
            """, unsafe_allow_html=True)


def create_plot(data, show_sma_5, show_sma_20, show_sma_60, show_sma_120, show_bollinger):
    """Close 가격, SMA, Volume을 포함한 Plotly 서브플롯 그래프를 생성합니다."""
    
    # 3x1 서브플롯 생성
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
        
    # 볼린저밴드
    if show_bollinger:
        # 상단 밴드 추가
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Upper Band'], mode='lines', name='Upper Band',
            line=dict(color='green', width=1, dash='dash')
        ),row=1, col=1)

        # 하단 밴드 추가
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Lower Band'], mode='lines', name='Lower Band',
            line=dict(color='green', width=1, dash='dash')
        ),row=1, col=1)
        
    ################
    ### Volume
    ################
        
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
    
    ################
    ### RSI
    ################
    
    # RSI 라인
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['RSI'], mode='lines', name='RSI',
        line=dict(color='#EB6239', width=1)
    ), row=3, col=1)
    
    # RSI SMA 라인
    fig.add_trace(go.Scatter(
            x=data['Date'], y=data['RSI_SMA'], mode='lines', name='RSI SMA',
            line=dict(color='#78A1DA', width=1)
    ), row=3, col=1)
    
    # 교차점에 화살표 추가
    for i, row in data.iterrows():
        if row['Cross'] == -1:  # 상승 교차
            fig.add_annotation(
                x=row['Date'], y=row['RSI'],
                text='', showarrow=True, arrowhead=1, arrowsize=1,
                arrowcolor='#78A1DA', ax=0, ay=-10,
                xref='x3', yref='y3'
            )
        elif row['Cross'] == 1:  # 하락 교차
            fig.add_annotation(
                x=row['Date'], y=row['RSI'],
                text='', showarrow=True, arrowhead=1, arrowsize=1,
                arrowcolor='#EB6239', ax=0, ay=10,
                xref='x3', yref='y3'
            )
    
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
                    tickvals=[30, 70]),  # y축 설정 (RSI),
        showlegend=False
    )

    return fig


def main():
    """Streamlit 앱의 메인 함수입니다."""
    stock = st.sidebar.text_input("주식 코드를 입력하세요", value='^GSPC')
    
    if stock is not None:
        data = load_data(stock)
        
        show_stock_name(stock)
        show_stock_price(data)
        
        st.divider()

        # 날짜 범위 필터링
        max_date = data['Date'].max()
        start_date = max_date - pd.DateOffset(years=1)
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
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1 : show_sma_5 = st.checkbox("5일 단순이동평균", value=False)
                with col2 : show_sma_20 = st.checkbox("20일 단순이동평균", value=True)
                with col3 : show_sma_60 = st.checkbox("60일 단순이동평균", value=False)
                with col4 : show_sma_120 = st.checkbox("120일 단순이동평균", value=False)
                with col5 : show_bollinger = st.checkbox("볼린저 밴드", value=True)
                
                fig = create_plot(filtered_data, show_sma_5, show_sma_20, show_sma_60, show_sma_120, show_bollinger)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("CSV 파일을 업로드하여 시각화할 수 있습니다.")

if __name__ == "__main__":
    main()
 