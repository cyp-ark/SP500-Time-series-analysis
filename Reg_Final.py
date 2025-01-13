import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
def load_data(stock):
    data = yf.download(stock)
    data.columns = [' '.join(col)[:-(len(stock)+1)] for col in data.columns]
    data.reset_index(inplace=True)
    
    data['Date'] = pd.to_datetime(data['Date'])
    return data
def show_stock_name(symbol):
    try:
        stock_name = yf.Ticker(symbol).info['longName']
        st.markdown(f"""
            <h1>
            Regression of {stock_name}
            </h1>
            """, unsafe_allow_html=True)
    except KeyError:
        return "종목 코드가 잘못되었거나 이름을 찾을 수 없습니다."
def LR_analysis(data, train_date_range, test_data_range, log=False, method="linear", alpha=0.1):
    if 'index' not in data.columns:
        data.reset_index(drop=False, inplace=True)
    train_date_range = [pd.to_datetime(date) for date in train_date_range]
    test_data_range = [pd.to_datetime(date) for date in test_data_range]
    X_train = np.array(data[(data['Date'] >= train_date_range[0]) & (data['Date'] <= train_date_range[1])].index).reshape(-1, 1)
    X_test = np.array(data[(data['Date'] >= test_data_range[0]) & (data['Date'] <= test_data_range[1])].index).reshape(-1, 1)
    y_train = np.array(data[(data['Date'] >= train_date_range[0]) & (data['Date'] <= train_date_range[1])]['Close'])
    y_test = np.array(data[(data['Date'] >= test_data_range[0]) & (data['Date'] <= test_data_range[1])]['Close'])
   
    if log:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
    
    if method == "linear":
        model = LinearRegression()
    elif method == "ridge":
        model = Ridge(alpha=alpha)
    elif method == "lasso":
        model = Lasso(alpha=alpha)
    elif method == "sgd":
        model = SGDRegressor(max_iter=1000, tol=1e-3, alpha=0.01, learning_rate='constant', eta0=0.01)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if log:
        y_train = np.expm1(y_train)
        y_test = np.expm1(y_test)
        y_train_pred = np.expm1(y_train_pred)
        y_test_pred = np.expm1(y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train.flatten(), y=(y_train), mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=X_train.flatten(), y=(y_train_pred), mode='lines', name='Train Pred'))
    fig.add_trace(go.Scatter(x=X_test.flatten(), y=(y_test), mode='lines', name='Test'))
    fig.add_trace(go.Scatter(x=X_test.flatten(), y=(y_test_pred), mode='lines', name='Test Pred'))
        
    return fig, mse_train, mse_test, r2_train, r2_test
        
def main():
    stock = st.sidebar.text_input("주식 코드를 입력하세요", value='^GSPC')
    
    if stock is not None:
        show_stock_name(stock)
        st.divider()
        data = load_data(stock)
        
        ################
        # 날짜 범위 선택
        ################
        
        min_date = data['Date'].min().date()
        q4_date = data['Date'].quantile(0.8).date()
        max_date = data['Date'].max().date()
        
        # 날짜 범위 슬라이더
        train_date_range = st.sidebar.slider(
            'Select a train date range',
            min_value=min_date,
            max_value=max_date,
            value=(min_date, q4_date - pd.Timedelta(days=1)),  # 기본값: 전체 범위
            format="YYYY-MM-DD",
            key="train_date_range"
        )
        test_data_range = st.sidebar.slider(
            'Select a test date range',
            min_value=min_date,
            max_value=max_date,
            value=(q4_date, max_date),  # 기본값: 전체 범위
            format="YYYY-MM-DD",
            key="test_data_range"
        )
        
        if train_date_range[1] >= test_data_range[0]:
            st.error("Test data should be later than train data")
            return
        else :
            ######################
            ### Linear Regression
            ######################
            st.subheader("Linear Regression")
            log = st.sidebar.checkbox("Log Scale", key="log")
            fig, mse_train, mse_test, r2_train, r2_test = LR_analysis(data, train_date_range, test_data_range, log=log, method="linear")
            st.write(f"Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}")
            st.write(f"Train R2: {r2_train:.2f}, Test R2: {r2_test:.2f}")
            st.plotly_chart(fig)
             
            ######################
            ### Ridge and Lasso
            ###################### 
            st.subheader("Lasso Regression")
            fig, mse_train_lasso, mse_test_lasso, r2_train_lasso, r2_test_lasso = LR_analysis(data, train_date_range, test_data_range, log=log, method="lasso")
            st.write(f"Train MSE: {mse_train_lasso:.2f}, Test MSE: {mse_test_lasso:.2f}")
            st.write(f"Train R2: {r2_train_lasso:.2f}, Test R2: {r2_test_lasso:.2f}")
            st.plotly_chart(fig)
            
            st.subheader("Ridge Regression")
            fig, mse_train, mse_test, r2_train, r2_test = LR_analysis(data, train_date_range, test_data_range, log=log, method="ridge")
            st.write(f"Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}")
            st.write(f"Train R2: {r2_train:.2f}, Test R2: {r2_test:.2f}")
            st.plotly_chart(fig)
            
            st.subheader("SGD Regression")
            fig, mse_train, mse_test, r2_train, r2_test = LR_analysis(data, train_date_range, test_data_range, log=log, method="sgd")
            st.write(f"Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}")
            st.write(f"Train R2: {r2_train:.2f}, Test R2: {r2_test:.2f}")
            st.plotly_chart(fig)
            
            
            
        
if __name__ == "__main__":
    main()