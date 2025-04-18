##########################################
### EDA 파트 구현 ###
##########################################



import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose 
import plotly.express as px
from scipy.stats import shapiro, kstest
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import openai
from langchain.chains import LLMChain 
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import requests  
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


def load_data(stock):
    data = yf.download(stock)
    data.columns = [' '.join(col)[:-(len(stock)+1)] for col in data.columns]
    data.reset_index(inplace=True)
    
    data['Date'] = pd.to_datetime(data['Date']) 
    return data
 

def log_scale(data):
    for col in data.columns[1:]:
        if 'log_'+ col not in data.columns:
            data['log_'+col] = np.log1p(data[col])
        else:
            pass
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


def perform_normality_test(data):
    """정규성 검정을 수행하고 결과 반환"""
    results = {}
    for column in data.columns:
        if data[column].dtype in [np.float64, np.int64]:  # 숫자형 데이터만 처리
            # Shapiro-Wilk Test
            stat, p_value = shapiro(data[column].dropna())
            results[column] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05  # 귀무가설: 정규성을 따름
            }
    return results


def interpret_decomposition_with_ai(signal, trend, seasonal, residual, model, col, stock):
    """
    시계열 분해 결과를 LangChain API를 통해 AI에게 전달하여 해석을 받는 함수
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

    # LangChain Prompt Template 정의
    prompt = PromptTemplate(
        input_variables=["signal", "trend", "seasonal", "residual", "model", "col", "stock"],
        template="""
        다음은 특정 주식 데이터의 시계열 분해 결과입니다:

        - 원본 신호 데이터(signal): {signal}
        - 추세(trend): {trend}
        - 계절성(seasonal): {seasonal}
        - 잔차(residual): {residual}

        분해 모델: {model}
        분석 대상 컬럼: {col}
        주식 코드: {stock}

        위 데이터를 기반으로 다음 질문에 답변해주세요:
        1. 추세 데이터(trend)에서 나타나는 주요 패턴은 무엇인가요?
        2. 계절성 데이터(seasonal)는 어떤 주기를 가지고 있나요?
        3. 잔차(residual) 데이터를 통해 알 수 있는 신호의 노이즈 특성은 무엇인가요?
        4. 전체적으로 데이터의 주요 특징을 요약해주세요.
        """
    )

    # LangChain Chain 설정
    chain = LLMChain(llm=llm, prompt=prompt)

    # LangChain 호출하여 해석 받기
    interpretation = chain.run({
        "signal": signal.to_json(),  # JSON 형식으로 변환하여 전달
        "trend": trend.to_json(),
        "seasonal": seasonal.to_json(),
        "residual": residual.to_json(),
        "model": model,
        "col": col,
        "stock": stock
    })

    return interpretation

def show_combined_histogram_with_tests(data):
    # 기본 데이터와 로그 데이터 분리
    name_columns = [col for col in data.columns if not col.startswith('log_') and col != 'Date']
    log_columns = [col for col in data.columns if col.startswith('log_')]

    # 정규성 검정 (원본 데이터)
    original_test_results = []
    for column in name_columns:
        stat, p_value = shapiro(data[column].dropna())
        is_normal = "Yes" if p_value > 0.05 else "No"
        original_test_results.append({
            "Column": column,
            "p-value": p_value,
            "Normal": is_normal
        })

    # 정규성 검정 (로그 데이터)
    log_test_results = []
    for column in log_columns:
        stat, p_value = shapiro(data[column].dropna())
        is_normal = "Yes" if p_value > 0.05 else "No"
        log_test_results.append({
            "Column": column,
            "p-value": p_value,
            "Normal": is_normal
        })

    # 원본 데이터 히스토그램 생성
    original_fig = make_subplots(
        rows=1, cols=5,  # 1행: 히스토그램
        subplot_titles=name_columns,
        horizontal_spacing=0.05
    )
    
    colors = ["#0000FF", "#87CEEB", "#008000", "#FFA500", "#FF0000"]
            
    for i, column in enumerate(name_columns): 
        original_fig.add_trace(
            go.Histogram(
                x=data[column].dropna(),
                name=f"{column}",
                marker_color=colors[i % len(colors)]  # 색상 리스트에서 순환
            ),
            row=1, col=i + 1
        )

    original_fig.update_layout(
        height=400,
        title_text="Original Data Histograms",
        showlegend=False
    )

    # 텍스트 박스 생성 (Original 데이터)
    original_text_boxes = make_subplots(rows=1, cols=5, horizontal_spacing=0.05)
    for i, result in enumerate(original_test_results):
        if result['p-value'] < 0.05:
            conclusion = "유의수준 0.05 하에서 정규분포라고 볼 수 없다."
        else:
            conclusion = "유의수준 0.05 하에서 정규분포라고 볼 수 있다."
        
        text = f"<b>p-value: {result['p-value']:.4f}<br>{conclusion}</b>"
        
        original_text_boxes.add_annotation(
            text=text,
            x=0.5, y=0.5,  # 중앙에 배치
            xref=f"x{i + 1}", yref="paper", 
            showarrow=False,
            font=dict(size=16, color="black"),
            align="center"
        )
    ''' 
    original_text_boxes = make_subplots(rows=1, cols=5, horizontal_spacing=0.05)
    for i, result in enumerate(original_test_results):
        text = f"<b>p-value: {result['p-value']:.4f}<br>Normal: {result['Normal']}</b>"
        original_text_boxes.add_annotation(
            text=text,
            x=0.5, y=0.5, 
            xref=f"x{i + 1}", yref="paper",
            showarrow=False,
            font=dict(size=16, color="black"),
            align="center" 
        ) 
    '''

    # 텍스트 박스 레이아웃 설정 (Original 데이터)
    original_text_boxes.update_layout( 
        height=150,  # 텍스트 박스 높이
        margin=dict(l=0, r=0, t=0, b=0),  # 여백 제거
        plot_bgcolor="white",  # 배경색 흰색
        paper_bgcolor="white",  # 캔버스 배경 흰색
        xaxis=dict( 
            visible=False,  # x축 숨김
            showgrid=False,  # 그리드 숨김
            zeroline=False  # 0선 숨김
        ), 
        yaxis=dict(
            visible=False,  # y축 숨김
            showgrid=False,  # 그리드 숨김
            zeroline=False  # 0선 숨김
        ),
        showlegend=False  # 범례 숨김
    )
    '''
    # 각 subplot에 테두리 추가
    for i in range(5):  # subplot 개수 (1x5)
        original_text_boxes.add_shape(
            type="rect",  # 사각형
            x0=0.2 * i,  # subplot 시작 x 좌표 
            x1=0.2 * (i + 1),  # subplot 끝 x 좌표
            y0=0,  # subplot 시작 y 좌표
            y1=1,  # subplot 끝 y 좌표
            xref="paper",  # x축 기준 (paper는 전체 영역)
            yref="paper",  # y축 기준 (paper는 전체 영역)
            line=dict(color="grey", width=2),  # 테두리 색상과 두께
        )
    ''' 

    # 로그 데이터 히스토그램 생성
    log_fig = make_subplots(
        rows=1, cols=5,  # 1행: 히스토그램
        subplot_titles=[f"{col} (Log-scaled)" for col in log_columns],
        horizontal_spacing=0.05
    )

    colors = ["#0000FF", "#87CEEB", "#008000", "#FFA500", "#FF0000"]
   
    for i, column in enumerate(log_columns): 
        log_fig.add_trace(
            go.Histogram(
                x=data[column].dropna(),
                name=f"{column}",
                marker_color=colors[i % len(colors)]  # 색상 리스트에서 순환
            ),
            row=1, col=i + 1
        )

    log_fig.update_layout(
        height=400,
        title_text="Log-transformed Data Histograms",
        showlegend=False
    )

    # 텍스트 박스 생성 (Log 데이터)
    log_text_boxes = make_subplots(rows=1, cols=5, horizontal_spacing=0.05)
    for i, result in enumerate(log_test_results):
        if result['p-value'] < 0.05:
            conclusion = "유의수준 0.05 하에서 정규분포라고 볼 수 없다."
        else:
            conclusion = "유의수준 0.05 하에서 정규분포라고 볼 수 있다."
        
        text = f"<b>p-value: {result['p-value']:.4f}<br>{conclusion}</b>"
        
        log_text_boxes.add_annotation(
            text=text,
            x=0.5, y=-0.3,  # x: 중앙, y: 아래쪽 위치 조정
            xref=f"x{i + 1}",  # subplot x축 기준
            yref=f"y{i + 1}",  # subplot y축 기준
            showarrow=False, 
            font=dict(size=16, color="black"),
            align="center"
        ) 

    # 텍스트 박스 레이아웃 설정 (Log 데이터)
    log_text_boxes.update_layout(
        height=150,  # 텍스트 박스 높이
        margin=dict(l=0, r=0, t=0, b=0),  # 여백 제거
        plot_bgcolor="white",  # 배경색 흰색
        paper_bgcolor="white",  # 캔버스 배경 흰색
        xaxis=dict(
            visible=False,  # x축 숨김
            showgrid=False,  # 그리드 숨김
            zeroline=False  # 0선 숨김
        ),
        yaxis=dict(
            visible=False,  # y축 숨김
            showgrid=False,  # 그리드 숨김
            zeroline=False  # 0선 숨김
        ),
        showlegend=False  # 범례 숨김
    )
    '''
    # 각 subplot에 테두리 추가
    for i in range(5):  # subplot 개수 (1x5)
        log_text_boxes.add_shape(
            type="rect",  # 사각형
            x0=0.2 * i,  # subplot 시작 x 좌표 
            x1=0.2 * (i + 1),  # subplot 끝 x 좌표
            y0=0,  # subplot 시작 y 좌표
            y1=1,  # subplot 끝 y 좌표
            xref="paper",  # x축 기준 (paper는 전체 영역)
            yref="paper",  # y축 기준 (paper는 전체 영역)
            line=dict(color="grey", width=2),  # 테두리 색상과 두께
        )
    '''
        
    return original_fig, original_text_boxes, log_fig, log_text_boxes


def show_decomposition(data, col, log, model):
    if log == 'log':
        data[col] = np.log1p(data[col])  # 로그 스케일 처리 
    else:
        data[col] = data[col]
         
    # 선택한 모델에 따라 분해 수행
    decomposition = seasonal_decompose(data[col], model=model, period=12)
    
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()  
    residual = decomposition.resid.dropna()
    
    # Plotly를 사용해 분해 결과 시각화
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=data['Date'], y=data[col], mode='lines', name='Original'), row=1, col=1)
    
    # 트렌드 서브플롯
    fig.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend'), row=2, col=1)

    # 계절성 서브플롯
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonal'), row=3, col=1)

    # 잔차 서브플롯
    fig.add_trace(go.Scatter(x=residual.index, y=residual, mode='lines', name='Residual'), row=4, col=1)
    
    fig.update_layout(
        yaxis=dict(title=col),
        yaxis2=dict(title='Trend'),
        yaxis3=dict(title='Seasonal'),
        yaxis4=dict(title='Residual'),
        height=800,
        title_text=f"Time Series Decomposition ({model.capitalize()} Model)"
    )
    
    return fig, data[col], trend, seasonal, residual



def visualize_smoothing(data): 
    """
    단순 이동평균, 가중 이동평균, 지수평활화를 시각화하며,
    단순 이동평균의 Window Size 및 Centered 옵션 조합에 따른 Loss Function 값을 계산 및 시각화.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px 
    import streamlit as st
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

    # Adjusted R² 계산 함수 추가
    def adjusted_r2_score(actual, predicted, n, k):
        r2 = r2_score(actual, predicted)
        return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

    # **1. 변수 선택**
    original_columns = [col for col in data.columns[1:] if not col.startswith('log_')]
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        column = st.selectbox("Select a variable for smoothing", original_columns)
    with col2:
        log_option = st.selectbox("Log Scale", ["original", "log"], index=0)
    with col3:
        transform_option = st.selectbox(
            "Select Data Transformation", ["original data", "1st differencing", "2nd differencing"], index=0
        )
    with col4:
        # Loss Function 선택
        loss_function = st.selectbox("Select Loss Function", ["RMSE", "MSE", "MAE", "MAPE", "R2"], index=0)

    # 데이터 선택 및 변환 적용
    selected_column = column
    if log_option == "log":
        selected_column = f"log_{column}"
        if selected_column not in data.columns:
            data[selected_column] = np.log1p(data[column])  # 로그 변환
    else:
        selected_column = column

    # 선택한 데이터 변환
    transformed_data = data[selected_column].copy()
    if transform_option == "1st differencing":
        transformed_data = transformed_data.diff()
    elif transform_option == "2nd differencing":
        transformed_data = transformed_data.diff().diff()

    # NaN 처리
    transformed_data = transformed_data.dropna()

    dates = data['Date']
    original_data = transformed_data

    # Train/Test Split
    train_size = int(len(original_data) * 0.8)
    train_data = original_data[:train_size]
    test_data = original_data[train_size:]

    # **2. 단순 이동평균 파라미터 선택 및 Loss Function 계산 통합**
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode 
    st.markdown("<h3 style='font-size:20px; color:black;'>Moving Average Parameters</h3>", unsafe_allow_html=True)

    # Window Size 범위 설정
    col6, col7 = st.columns(2)
    with col6:
        min_window = st.number_input("Minimum Window Size", min_value=1, max_value=500, value=2, step=1)
    with col7:
        max_window = st.number_input("Maximum Window Size", min_value=min_window, max_value=500, value=20, step=1)

    # Loss 계산 (Train 데이터 기준)
    results = []

    for window_size in range(min_window, max_window + 1):
        moving_avg = train_data.rolling(window=window_size).mean().shift(-1)
        valid_indices = ~moving_avg.isna()

        if valid_indices.sum() == 0:  # 유효 데이터가 없으면 건너뛰기
            continue

        actual = train_data[valid_indices]
        predicted = moving_avg[valid_indices]

        if loss_function == "RMSE":
            loss = np.sqrt(mean_squared_error(actual, predicted))
        elif loss_function == "MSE":
            loss = mean_squared_error(actual, predicted)
        elif loss_function == "MAE":
            loss = mean_absolute_error(actual, predicted)
        elif loss_function == "MAPE":
            loss = mean_absolute_percentage_error(actual, predicted)
        elif loss_function == "R2":
            loss = r2_score(actual, predicted)  

        results.append({
            "Window Size": window_size, 
            "Loss": loss
        })

    # 결정계수면 역으로 결과 정렬 및 최적 파라미터 선택
    if loss_function == 'R2':
        results_df = pd.DataFrame(results).sort_values(by="Loss", ascending=False)
    else:
        results_df = pd.DataFrame(results).sort_values(by="Loss", ascending=True)


    # 최적 파라미터 적용하여 Test 데이터 예측       
    best_params = results_df.iloc[0]
    best_window = int(best_params["Window Size"])
    
    # **Ag-Grid 설정 및 UI 배치**  
    col_loss_table, col_loss_chart = st.columns([3, 7])

    with col_loss_table:
        gb = GridOptionsBuilder.from_dataframe(results_df)
        gb.configure_selection('multiple', use_checkbox=True) # , pre_selected_rows=list(range(5))

        # **첫 행 강조 (lightgreen 배경색 설정)**
        custom_css = {
            ".ag-row:first-child": {
                "background-color": "lightgreen !important;"
            }
        }

        # **열 크기 자동화**
        gb.configure_default_column(resizable=True, autoSize=True)
        gb.configure_grid_options(domLayout='normal')  # 스크롤 가능하도록 설정
        grid_options = gb.build()

        # Ag-Grid 초기 선택 설정
        initial_selection = results_df.head(5).to_dict('records')
        grid_response = AgGrid(
            results_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            theme='streamlit',
            custom_css=custom_css,
            selected_rows=initial_selection
        )

        # 선택된 데이터 가져오기
        selected_rows = grid_response['selected_rows']
        selected_results = pd.DataFrame(selected_rows)

    # **상위 5개 행 디폴트 시각화 및 선택된 조합의 이동평균 계산**
    with col_loss_chart:
        fig = go.Figure()

        # 상위 5개 행 기본 시각화 (점선)
        visualized_windows = set()
        
        for _, row in results_df.head(5).iterrows():
            window_size = int(row['Window Size'])
            visualized_windows.add(window_size)
            test_moving_avg = test_data.rolling(window=window_size).mean()

            fig.add_trace(go.Scatter(
                x=dates[train_size:],
                y=test_moving_avg,
                mode="lines+markers",
                name=f"Window Size: {window_size}",
                line=dict(dash="dot")
            ))

        # 선택된 데이터 추가 시각화 (실선)
        for _, row in selected_results.iterrows():
            window_size = int(row['Window Size'])
            if window_size not in visualized_windows:
                test_moving_avg = test_data.rolling(window=window_size).mean()

                fig.add_trace(go.Scatter(
                    x=dates[train_size:],
                    y=test_moving_avg,
                    mode="lines+markers",
                    name=f"Window Size: {window_size}",
                    line=dict(dash="dot")
                ))

        # 테스트 데이터 추가 (검정색 실선)
        fig.add_trace(go.Scatter(
            x=dates[train_size:],
            y=test_data,
            mode="lines",
            name="Test Data",
            line=dict(color="black", dash="solid")
        ))

        # 그래프 설정 및 출력
        fig.update_layout(title="Test Predictions for Selected Moving Averages",
                        xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    # 테스트 데이터에 적합한 이동 평균 계산 
    test_moving_avg = test_data.rolling(window=best_window).mean().shift(-1)
    valid_indices_test = ~test_moving_avg.isna()

    # 유효한 테스트 데이터와 이동 평균 선택
    actual_test = test_data[valid_indices_test]
    predicted_test = test_moving_avg[valid_indices_test]

    # 테스트 데이터 Loss 계산
    if loss_function == "RMSE":
        test_loss = np.sqrt(mean_squared_error(actual_test, predicted_test))
    elif loss_function == "MSE":
        test_loss = mean_squared_error(actual_test, predicted_test)
    elif loss_function == "MAE":
        test_loss = mean_absolute_error(actual_test, predicted_test)
    elif loss_function == "MAPE":
        test_loss = mean_absolute_percentage_error(actual_test, predicted_test)
    elif loss_function == "R2":
        test_loss = r2_score(actual_test, predicted_test)

    # 테스트 데이터 Loss 출력
    st.write(f"Best Window Size: {best_window}")
    st.write(f"Test Data Loss ({loss_function}): {test_loss:.4f}")

    
    
    # **6. 가중 이동평균 (Weighted Moving Average) 파라미터 선택 및 Loss Function 계산**
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

    st.markdown("<h3 style='font-size:20px; color:black;'>Weighted Moving Average Parameters</h3>", unsafe_allow_html=True)
    col8, col9 = st.columns(2) 

    with col8:
        min_wma_window = st.number_input("Minimum Window Size for WMA", min_value=1, max_value=500, value=2, step=1)
    with col9:
        max_wma_window = st.number_input("Maximum Window Size for WMA", min_value=min_wma_window, max_value=500, value=20, step=1)

    weight_methods = ["Linear", "Reverse", "Exponential", "Triangular"]

    wma_results = []
    for weight_type in weight_methods:
        for window_size in range(min_wma_window, max_wma_window + 1):
            # 가중치 계산
            if weight_type == "Linear":
                weights = np.arange(1, window_size + 1).astype(float)
            elif weight_type == "Reverse":
                weights = np.arange(window_size, 0, -1).astype(float)
            elif weight_type == "Exponential":
                k = 0.5
                weights = np.exp(-k * (window_size - np.arange(1, window_size + 1))).astype(float)
            elif weight_type == "Triangular":
                weights = 1 - np.abs(np.linspace(-1, 1, window_size))

            weights /= weights.sum()  # 가중치 정규화

            # WMA 계산
            train_wma = train_data.rolling(window=window_size).apply(
                lambda x: np.dot(x, weights[:len(x)]), raw=True
            ).shift(-1)

            valid_indices = ~train_wma.isna()

            if valid_indices.sum() == 0:
                continue

            actual = train_data[valid_indices]
            predicted = train_wma[valid_indices]

            if loss_function == "RMSE":
                loss = np.sqrt(mean_squared_error(actual, predicted))
            elif loss_function == "MSE":
                loss = mean_squared_error(actual, predicted)
            elif loss_function == "MAE":
                loss = mean_absolute_error(actual, predicted)
            elif loss_function == "MAPE":
                loss = mean_absolute_percentage_error(actual, predicted)
            elif loss_function == "R2":
                loss = r2_score(actual, predicted)  

            wma_results.append({"Window Size": window_size, "Weight Type": weight_type, "Loss": loss})

    # 결정계수면 역으로
    if loss_function == 'R2':
        wma_results_df = pd.DataFrame(wma_results).sort_values(by="Loss", ascending=False)
    else:
        wma_results_df = pd.DataFrame(wma_results).sort_values(by="Loss", ascending=True)

    # **Ag-Grid로 데이터프레임 및 초기 선택**
    col_wma_table, col_wma_chart = st.columns([3, 7])

    with col_wma_table:
        gb = GridOptionsBuilder.from_dataframe(wma_results_df)
        gb.configure_selection('multiple', use_checkbox=True)
        gb.configure_default_column(resizable=True, autoSize=True)
        gb.configure_grid_options(domLayout='normal')  # 스크롤 가능하도록 설정

        # 첫 행 강조 (lightgreen 배경)
        custom_css = {
            ".ag-row:first-child": {
                "background-color": "lightgreen !important;"
            }
        }

        grid_options = gb.build()

        # 상위 5개 초기 선택
        initial_selection = wma_results_df.head(5).to_dict('records')
        grid_response = AgGrid(
            wma_results_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            theme='streamlit',
            selected_rows=initial_selection,
            custom_css=custom_css
        )

        # 선택된 데이터 가져오기
        selected_rows = grid_response['selected_rows']
        selected_results = pd.DataFrame(selected_rows)

    # **그래프 시각화**
    with col_wma_chart:
        fig = go.Figure()

        # 상위 5개 데이터 시각화 (점선)
        for _, row in wma_results_df.head(5).iterrows():
            window_size = int(row['Window Size'])
            weight_type = row['Weight Type']

            # WMA 계산
            weights = None
            if weight_type == "Linear":
                weights = np.arange(1, window_size + 1).astype(float)
            elif weight_type == "Reverse":
                weights = np.arange(window_size, 0, -1).astype(float)
            elif weight_type == "Exponential":
                k = 0.5
                weights = np.exp(-k * (window_size - np.arange(1, window_size + 1))).astype(float)
            elif weight_type == "Triangular":
                weights = 1 - np.abs(np.linspace(-1, 1, window_size))

            weights /= weights.sum()  # 가중치 정규화

            test_wma = test_data.rolling(window=window_size).apply(
                lambda x: np.dot(x, weights[:len(x)]), raw=True
            )

            fig.add_trace(go.Scatter(
                x=dates[train_size:],
                y=test_wma,
                mode="lines+markers",
                name=f"Default WMA Window: {window_size}, Weight: {weight_type}",
                line=dict(dash="dot")  # 점선으로 표시
            ))

        # 선택된 데이터 추가 시각화 (실선)
        for _, row in selected_results.iterrows():
            window_size = int(row['Window Size'])
            weight_type = row['Weight Type']

            # WMA 계산
            weights = None
            if weight_type == "Linear":
                weights = np.arange(1, window_size + 1).astype(float)
            elif weight_type == "Reverse":
                weights = np.arange(window_size, 0, -1).astype(float)
            elif weight_type == "Exponential":
                k = 0.5
                weights = np.exp(-k * (window_size - np.arange(1, window_size + 1))).astype(float)
            elif weight_type == "Triangular":
                weights = 1 - np.abs(np.linspace(-1, 1, window_size))

            weights /= weights.sum()  # 가중치 정규화

            test_wma = test_data.rolling(window=window_size).apply(
                lambda x: np.dot(x, weights[:len(x)]), raw=True
            )

            fig.add_trace(go.Scatter(
                x=dates[train_size:],
                y=test_wma,
                mode="lines+markers",
                name=f"WMA Window: {window_size}, Weight: {weight_type}",
                line=dict(dash="dot")
            ))

        # 테스트 데이터 추가 (검정색 실선)
        fig.add_trace(go.Scatter(
            x=dates[train_size:],
            y=test_data,
            mode="lines",
            name="Test Data",
            line=dict(color="black", dash="solid")
        ))

        # 그래프 설정
        fig.update_layout(
            title="WMA Loss by Window Size and Weight Type",
            xaxis_title="Date",
            yaxis_title=f"{loss_function}",
            legend_title="Weight Type"
        )
        st.plotly_chart(fig, use_container_width=True)
 
    # **최적 파라미터 및 Test Loss 출력**
    # 테스트 데이터에 대한 WMA 계산
    best_params2 = wma_results_df.iloc[0]
    best_wma_window = int(best_params2['Window Size'])
    best_weight_type = best_params2['Weight Type']

    # 가중치 계산 
    if best_weight_type == "Linear":
        weights = np.arange(1, best_wma_window + 1).astype(float)
    elif best_weight_type == "Reverse":
        weights = np.arange(best_wma_window, 0, -1).astype(float)
    elif best_weight_type == "Exponential":
        k = 0.5
        weights = np.exp(-k * (best_wma_window - np.arange(1, best_wma_window + 1))).astype(float)
    elif best_weight_type == "Triangular":
        weights = 1 - np.abs(np.linspace(-1, 1, best_wma_window))

    weights /= weights.sum()  # 가중치 정규화

    # 테스트 데이터에 대한 WMA 계산
    test_wma = test_data.rolling(window=best_wma_window).apply(
        lambda x: np.dot(x, weights[:len(x)]), raw=True
    ).shift(-1)

    # 유효 데이터 필터링
    valid_indices_test = ~test_wma.isna()
    actual_test = test_data[valid_indices_test]
    predicted_test = test_wma[valid_indices_test]

    # Loss 계산
    if loss_function == "RMSE":
        test_loss = np.sqrt(mean_squared_error(actual_test, predicted_test))
    elif loss_function == "MSE":
        test_loss = mean_squared_error(actual_test, predicted_test)
    elif loss_function == "MAE":
        test_loss = mean_absolute_error(actual_test, predicted_test)
    elif loss_function == "MAPE":
        test_loss = mean_absolute_percentage_error(actual_test, predicted_test)
    elif loss_function == "R2":
        test_loss = r2_score(actual_test, predicted_test)

    # Loss 값 출력
    best_params2 = wma_results_df.iloc[0] 
    st.write(f"Best Window Size: {int(best_params2['Window Size'])}")
    st.write(f"Best Weight Type: {best_params2['Weight Type']}")
    st.write(f"Test Data Loss ({loss_function}): {test_loss:.4f}")


 
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st  

    # **7. 지수평활 (Exponential Smoothing) 파라미터 선택 및 Loss Function 계산**
    st.markdown("<h3 style='font-size:20px; color:black;'>Exponential Smoothing Parameters</h3>", unsafe_allow_html=True)

    col_min_alpha, col_max_alpha, col_alpha_step = st.columns(3)
    with col_min_alpha: 
        min_alpha = st.number_input("Minimum Alpha", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    with col_max_alpha:
        max_alpha = st.number_input("Maximum Alpha", min_value=min_alpha, max_value=1.0, value=0.5, step=0.01)
    with col_alpha_step:
        alpha_step = st.number_input("Alpha Step", min_value=0.01, max_value=1.0, value=0.01, step=0.01)
 
    # Generate alpha values
    alpha_values = np.arange(min_alpha, max_alpha + alpha_step, alpha_step)

    # Placeholder for ES results
    es_results = []

    # Calculate Loss for Each Alpha
    for alpha in alpha_values:
        train_es = train_data.ewm(alpha=alpha, adjust=False).mean().shift(-1)
        valid_indices = ~train_es.isna()

        if valid_indices.sum() == 0:
            continue

        actual = train_data[valid_indices]
        predicted = train_es[valid_indices]

        if loss_function == "RMSE":
            loss = np.sqrt(mean_squared_error(actual, predicted))
        elif loss_function == "MSE":
            loss = mean_squared_error(actual, predicted)
        elif loss_function == "MAE":
            loss = mean_absolute_error(actual, predicted)
        elif loss_function == "MAPE":
            loss = mean_absolute_percentage_error(actual, predicted)
        elif loss_function == "R2":
            loss = r2_score(actual, predicted)

        es_results.append({"Alpha": alpha, "Loss": loss})

    # Sort Results
    if loss_function == 'R2':
        es_results_df = pd.DataFrame(es_results).sort_values(by="Loss", ascending=False)
    else:
        es_results_df = pd.DataFrame(es_results).sort_values(by="Loss", ascending=True)

    es_results_df["Alpha"] = es_results_df["Alpha"].round(2)

    # Display Results with AgGrid
    col_es_table, col_es_chart = st.columns([3, 7])

    with col_es_table:
        gb = GridOptionsBuilder.from_dataframe(es_results_df)
        gb.configure_selection('multiple', use_checkbox=True)
        gb.configure_default_column(resizable=True, autoSize=True)
        gb.configure_grid_options(domLayout='normal')
        
        # **첫 행 강조 (lightgreen 배경색 설정)**
        custom_css = {
            ".ag-row:first-child": {
                "background-color": "lightgreen !important;"
            }
        }

        initial_selection = es_results_df.head(5).to_dict('records')
        grid_response = AgGrid(
            es_results_df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            theme='streamlit',
            custom_css = custom_css,
            selected_rows=initial_selection
        )

        selected_rows = grid_response['selected_rows']
        selected_results = pd.DataFrame(selected_rows)

    # Plot Results
    with col_es_chart:
        fig_es = go.Figure()

        # Plot top 5 results (dotted line)
        for _, row in es_results_df.head(5).iterrows():
            alpha = row['Alpha']
            test_es = test_data.ewm(alpha=alpha, adjust=False).mean()

            fig_es.add_trace(go.Scatter(
                x=dates[train_size:],
                y=test_es,
                mode="lines+markers",
                name=f"Alpha: {alpha:.2f} (Default)",
                line=dict(dash="dot")
            ))

        # Plot selected results (solid line)
        for _, row in selected_results.iterrows():
            alpha = row['Alpha']
            test_es = test_data.ewm(alpha=alpha, adjust=False).mean()

            fig_es.add_trace(go.Scatter(
                x=dates[train_size:],
                y=test_es,
                mode="lines+markers",
                name=f"Alpha: {alpha:.2f} (Selected)",
                line=dict(dash="dot")
            ))

        # Plot test data (black solid line)
        fig_es.add_trace(go.Scatter(
            x=dates[train_size:],
            y=test_data,
            mode="lines",
            name="Test Data",
            line=dict(color="black", dash="solid")
        ))

        # Update layout
        fig_es.update_layout(
            title="Exponential Smoothing Loss by Alpha",
            xaxis_title="Date",
            yaxis_title=f"{loss_function}",
            legend_title="Alpha Values"
        )

        st.plotly_chart(fig_es, use_container_width=True)

    # Best Parameters Output

    best_alpha = es_results_df.iloc[0]['Alpha']

    test_es = test_data.ewm(alpha=best_alpha, adjust=False).mean().shift(-1)
    valid_indices_test = ~test_es.isna()

    # 유효한 테스트 데이터와 예측 값 선택
    actual_test = test_data[valid_indices_test]
    predicted_test = test_es[valid_indices_test]

    # 테스트 데이터 Loss 계산
    if loss_function == "RMSE":
        test_loss = np.sqrt(mean_squared_error(actual_test, predicted_test))
    elif loss_function == "MSE":
        test_loss = mean_squared_error(actual_test, predicted_test)
    elif loss_function == "MAE":
        test_loss = mean_absolute_error(actual_test, predicted_test)
    elif loss_function == "MAPE":
        test_loss = mean_absolute_percentage_error(actual_test, predicted_test)
    elif loss_function == "R2":
        test_loss = r2_score(actual_test, predicted_test)

    # 테스트 데이터 Loss 출력
    st.write(f"Best Parameters for ES: Alpha={best_alpha}")
    st.write(f"Test Data Loss ({loss_function}): {test_loss:.4f}")

    
    
    
    import pandas as pd
    import numpy as np 
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
    from statsmodels.tsa.arima.model import ARIMA
    import plotly.graph_objects as go
    import streamlit as st  

            
    
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go 
    import streamlit as st
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

    # ARIMA Parameters
    st.markdown("<h3 style='font-size:20px; color:black;'>ARIMA Parameters (Rolling Update)</h3>", unsafe_allow_html=True)

    # ARIMA parameter ranges
    col_p_range, col_q_range = st.columns(2)
    with col_p_range:
        max_p = st.number_input("Maximum AR term (p)", min_value=0, max_value=10, value=3, step=1)
    with col_q_range:
        max_q = st.number_input("Maximum MA term (q)", min_value=0, max_value=10, value=3, step=1)

    # Setting differencing order (d)
    if 'transform_option' in locals() and transform_option == "original data":
        d = 0
    elif 'transform_option' in locals() and transform_option == "1st differencing":
        d = 1
    elif 'transform_option' in locals() and transform_option == "2nd differencing":
        d = 2
    else:
        d = 1

    # Placeholder for ARIMA results
    aic_results = []

    # Train ARIMA models and calculate AIC and loss
    with st.spinner("Training ARIMA model..."):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=len(test_data))

                    # Calculate loss
                    if loss_function == "RMSE":
                        loss = np.sqrt(mean_squared_error(test_data, forecast))
                    elif loss_function == "MSE":
                        loss = mean_squared_error(test_data, forecast)
                    elif loss_function == "MAE":
                        loss = mean_absolute_error(test_data, forecast)
                    elif loss_function == "MAPE":
                        loss = mean_absolute_percentage_error(test_data, forecast)
                    elif loss_function == "R2":
                        loss = r2_score(test_data, forecast)

                    aic_results.append({"p": p, "q": q, "AIC": fitted_model.aic}) 
                except Exception as e:
                    aic_results.append({"p": p, "q": q, "AIC": np.nan})  

    # Convert results to DataFrame
    aic_results_df = pd.DataFrame(aic_results).sort_values(by="AIC", ascending=True)

    # Display ARIMA results with AgGrid
    col_results_table, col_results_chart = st.columns([3, 7])

    with col_results_table:
        gb = GridOptionsBuilder.from_dataframe(aic_results_df)
        gb.configure_selection('multiple', use_checkbox=True)
        gb.configure_default_column(resizable=True, autoSize=True)
        gb.configure_grid_options(domLayout='normal')
        
        # **첫 행 강조 (lightgreen 배경색 설정)**
        custom_css = {
            ".ag-row:first-child": {
                "background-color": "lightgreen !important;"
            }
        }

        initial_selection = aic_results_df.head(3).to_dict('records')  # Select top 3 AIC models by default
        grid_response = AgGrid(
            aic_results_df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            theme='streamlit',
            custom_css = custom_css,
            selected_rows=initial_selection
        )

        selected_rows = grid_response['selected_rows']
        selected_results = pd.DataFrame(selected_rows)

    # Perform rolling predictions
    def forecast_one_step(fitted_model):
        forecast = fitted_model.get_forecast(steps=1)
        fc = forecast.predicted_mean.values[0] 
        return fc

    # Plot predictions for selected models and rolling predictions
    with col_results_chart:
        fig = go.Figure()

        # Plot test data
        fig.add_trace(go.Scatter(
            x=dates[train_size:], 
            y=test_data, 
            mode="lines", 
            name="Test Data", 
            line=dict(color="black")
        ))

        # Perform rolling predictions for top 3 models (default display)
        for _, row in aic_results_df.head(3).iterrows():
            p, q = int(row['p']), int(row['q'])
            model = ARIMA(train_data, order=(p, d, q))
            fitted_model = model.fit()

            rolling_pred = []
            rolling_model = fitted_model
            for new_ob in test_data:
                fc = forecast_one_step(rolling_model)
                rolling_pred.append(fc)
                rolling_model = rolling_model.append([new_ob], refit=False)

            fig.add_trace(go.Scatter(
                x=dates[train_size:],
                y=rolling_pred,
                mode="lines+markers",
                name=f"ARIMA(p={p}, d={d}, q={q}) [Default]",
                line=dict(dash="dot")
            ))

        # Perform rolling predictions for user-selected models
        for _, row in selected_results.iterrows():
            p, q = int(row['p']), int(row['q'])
            model = ARIMA(train_data, order=(p, d, q))
            fitted_model = model.fit()

            rolling_pred = []
            rolling_model = fitted_model
            for new_ob in test_data:
                fc = forecast_one_step(rolling_model)
                rolling_pred.append(fc)
                rolling_model = rolling_model.append([new_ob], refit=False)

            fig.add_trace(go.Scatter(
                x=dates[train_size:],
                y=rolling_pred,
                mode="lines+markers",
                name=f"ARIMA(p={p}, d={d}, q={q}) [Selected]",
                line=dict(dash="dot")
            ))

        # Update layout
        fig.update_layout(
            title="ARIMA Predictions with Rolling Updates",
            xaxis_title="Date",
            yaxis_title="Values",
            legend_title="Models"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Evaluate final model performance
    best_params = aic_results_df.iloc[0]
    st.write(f"Best Parameters of ARIMA: p={best_params['p']}, d={d}, q={best_params['q']}")

    model = ARIMA(train_data, order=(int(best_params['p']), d, int(best_params['q'])))
    fitted_model = model.fit()

    y_pred = []
    for new_ob in test_data:
        fc = forecast_one_step(fitted_model)
        y_pred.append(fc)
        fitted_model = fitted_model.append([new_ob], refit=False)
    '''
    # Final rolling predictions
    fig_final = go.Figure()
    fig_final.add_trace(go.Scatter(x=dates[:train_size], y=train_data, mode="lines", name="Train Data", line=dict(color="blue")))
    fig_final.add_trace(go.Scatter(x=dates[train_size:], y=test_data, mode="lines", name="Test Data", line=dict(color="orange")))
    fig_final.add_trace(go.Scatter(x=dates[train_size:], y=y_pred, mode="lines", name="Rolling Predictions", line=dict(color="green")))
    fig_final.update_layout(title="Final Rolling Predictions")
    st.plotly_chart(fig_final, use_container_width=True)
    '''

    # Calculate loss for the best model
    if loss_function == "RMSE":
        test_loss = np.sqrt(mean_squared_error(test_data, y_pred))
    elif loss_function == "MSE":
        test_loss = mean_squared_error(test_data, y_pred)
    elif loss_function == "MAE":
        test_loss = mean_absolute_error(test_data, y_pred)
    elif loss_function == "MAPE":
        test_loss = mean_absolute_percentage_error(test_data, y_pred)
    elif loss_function == "R2":
        test_loss = r2_score(test_data, y_pred)

    st.write(f"Test Loss ({loss_function}): {test_loss:.4f}")
        




    # 9. 최적 파라미터 조합 시각화
    st.markdown("<h3 style='font-size:20px; color:black;'>Best Parameter Smoothing Method</h3>", unsafe_allow_html=True)

    # 기존 기능 시각화
    fig = go.Figure()

    # 원 데이터 시각화 
    fig.add_trace(
        go.Scatter(x=dates, y=original_data, mode='lines', name='Original', line=dict(color='blue'))
    )
 
    # (1) 단순 이동평균 최적 파라미터 적용 시각화
    best_moving_avg = train_data.rolling(window=best_window).mean()
    fig.add_trace(
        go.Scatter(
            x=dates[:len(best_moving_avg)],
            y=best_moving_avg,
            mode='lines',
            name=f'MA (Window={best_window})',
            line=dict(color='orange')
        )
    )

    # (2) 가중 이동평균 최적 파라미터 적용 시각화
    best_wma_type = wma_results_df.iloc[0]['Weight Type']
    best_wma_window = int(wma_results_df.iloc[0]['Window Size'])

    # 가중치 계산
    if best_wma_type == 'Linear':
        weights = np.arange(1, best_wma_window + 1).astype(float)
    elif best_wma_type == 'Reverse':
        weights = np.arange(best_wma_window, 0, -1).astype(float)
    elif best_wma_type == 'Exponential':
        k = 0.5
        weights = np.exp(-k * (best_wma_window - np.arange(1, best_wma_window + 1))).astype(float)
    elif best_wma_type == 'Triangular':
        weights = 1 - np.abs(np.linspace(-1, 1, best_wma_window))
    weights /= weights.sum()

    # 가중 이동평균 계산
    best_wma = train_data.rolling(window=best_wma_window).apply(lambda x: np.dot(x, weights), raw=True)
    fig.add_trace(
        go.Scatter(
            x=dates[:len(best_wma)],
            y=best_wma,
            mode='lines',
            name=f'WMA ({best_wma_type}, Window={best_wma_window})',
            line=dict(color='green')
        )
    )

    # (3) 지수평활 최적 파라미터 적용 시각화
    best_alpha = es_results_df.iloc[0]['Alpha']
    best_es = train_data.ewm(alpha=best_alpha, adjust=False).mean()
    fig.add_trace(
        go.Scatter(
            x=dates[:len(best_es)],
            y=best_es,
            mode='lines',
            name=f'ES (Alpha={best_alpha})',
            line=dict(color='red')
        )
    )
    
    # (4) ARIMA 모델의 학습 데이터 적합값
    arima_fitted_values = fitted_model.fittedvalues
 
    # 학습 데이터 적합값 시각화
    fig.add_trace(
        go.Scatter(
            x=dates[:len(arima_fitted_values)],
            y=arima_fitted_values,
            mode='lines',
            name=f'ARIMA Fit (p={best_params["p"]}, d={d}, q={best_params["q"]})',
            line=dict(color='purple')
        )  
    )


    # Test 데이터 최적 파라미터 적용 시각화
    # (1) 단순 이동평균   
    best_test_moving_avg = test_data.rolling(window=best_window).mean()
    fig.add_trace(
        go.Scatter(
            x=dates[len(train_data):len(train_data)+len(best_test_moving_avg)],
            y=best_test_moving_avg,
            mode='lines',
            name=f'MA Test (Window={best_window})',
            line=dict(color='orange', dash='dot')
        )
    )

    # (2) 가중 이동평균
    best_test_wma = test_data.rolling(window=best_wma_window).apply(lambda x: np.dot(x, weights), raw=True)
    fig.add_trace(
        go.Scatter(
            x=dates[len(train_data):len(train_data)+len(best_test_wma)],
            y=best_test_wma,
            mode='lines',
            name=f'WMA Test ({best_wma_type}, Window={best_wma_window})',
            line=dict(color='green', dash='dot')
        )
    )

    # (3) 지수평활 
    best_test_es = test_data.ewm(alpha=best_alpha, adjust=False).mean()
    fig.add_trace(
        go.Scatter(
            x=dates[len(train_data):len(train_data)+len(best_test_es)],
            y=best_test_es, 
            mode='lines',
            name=f'ES Test (Alpha={best_alpha})',
            line=dict(color='red', dash='dot')
        )
    ) 

    # (4) ARIMA 모델 예측값 (테스트 데이터) 
    fig.add_trace(
        go.Scatter(
            x=dates[len(train_data):len(train_data) + len(y_pred)],
            y=y_pred,
            mode='lines',
            name=f'ARIMA Forecast (p={best_params["p"]}, d={d}, q={best_params["q"]})',
            line=dict(color='purple', dash='dash')
        )
    )
    # 그래프 출력
    st.plotly_chart(fig, use_container_width=True)
    
    
    
def plot_acf_pacf(data, title):
    """ACF 및 PACF 그래프 생성"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_acf(data.dropna(), ax=axes[0], lags=40, title=f"ACF - {title}")
    plot_pacf(data.dropna(), ax=axes[1], lags=40, title=f"PACF - {title}")
    
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    
    st.pyplot(fig)
 


def perform_adf_test(data, title):
    """ADF 검정 수행"""
    result = adfuller(data.dropna())
    st.write(f"#### ADF Test - {title}")
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")
    st.write("Critical Values:")
    for key, value in result[4].items():
        st.write(f"   {key}: {value:.4f}")
    if result[1] <= 0.05:
        st.write("Result: 데이터가 Stationary 상태입니다 (p-value <= 0.05).")
    else: 
        st.write("Result: 데이터가 Non-Stationary 상태입니다 (p-value > 0.05).")



def stationary_test(data, default_col):
    """Stationary Test: ACF, ADF, KPSS 수행 with Selectbox, Results Table, and Hypothesis"""

    import numpy as np
    import pandas as pd
    import streamlit as st 
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt

    # Rolling mean and std visualization function
    def plot_rolling(data, roll_size):
        # 이동평균함수(rolling) - 평균, 표준편차
        roll_mean = data.rolling(window=roll_size).mean()
        roll_std = data.rolling(window=roll_size).std()

        # 시각화
        plt.figure(figsize=(11, 5))
        plt.plot(data, color='blue', alpha=0.25, label='Original')
        plt.plot(roll_mean, color='black', label='Rolling mean')
        plt.plot(roll_std, color='red', label='Rolling std')
        plt.title('Rolling Mean & Standard Deviation') 
        plt.ylabel("Values")
        plt.legend()
        st.pyplot(plt)

   
    # **1. 변수 선택, 로그 변환, 차분 옵션 설정**
    col1, col2, col3, col4 = st.columns(4)
    
    # 변수 선택
    with col1:
        col = st.selectbox("Select a variable for stationary test:", data.columns[1:6], index=data.columns.tolist().index(default_col))
    # 로그 변환 옵션
    with col2:
        log_option = st.selectbox("Log Transformation:", ["original", "log"], index=0)
    # 차분 옵션
    with col3:
        option = st.selectbox(
            "Select Data Transformation:",
            ["Original Data", "1st Differencing", "2nd Differencing"]
        )
    # Rolling Window Size
    with col4:
        roll_size = st.number_input("Rolling Window Size:", min_value=1, max_value=100, value=5, step=1)

    # **2. 데이터 변환**
    # 로그 변환
    if log_option == "log":
        col = f"log_{col}"
        if col not in data.columns:
            data[col] = np.log1p(data[default_col])  # 로그 변환

    # 차분 적용
    if option == "Original Data":
        selected_data = data[col]
        transformation = "Original Data"
    elif option == "1st Differencing":
        selected_data = data[col].diff().dropna()
        transformation = "1st Differencing"
    elif option == "2nd Differencing":
        selected_data = data[col].diff().diff().dropna()
        transformation = "2nd Differencing"

    # **3. Rolling Mean & Std 시각화**
    st.markdown(f"### Rolling Mean & Standard Deviation - {transformation}")
    plot_rolling(selected_data, roll_size)

    
    # **4. ACF 및 PACF 그래프**
    st.markdown(f"### ACF and PACF - {transformation}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(selected_data, ax=ax[0], lags=20, title=f"ACF - {transformation}")
    plot_pacf(selected_data, ax=ax[1], lags=20, title=f"PACF - {transformation}")
    st.pyplot(fig)
 
    # **5. ADF Test**
    st.markdown(f"### ADF Test Results - {transformation}")
    st.markdown("**Hypotheses:**") 
    st.markdown("- **H0**: 데이터는 Non-Stationary Data이다 (단위근이 존재한다).")
    st.markdown("- **H1**: 데이터는 Stationary Data이다. (단위근이 없다).")
    result_adf = adfuller(selected_data)
    adf_table = {
        "Statistic": [result_adf[0]],
        "p-value": [result_adf[1]],
        "Critical Value (1%)": [result_adf[4]["1%"]],
        "Critical Value (5%)": [result_adf[4]["5%"]],
        "Critical Value (10%)": [result_adf[4]["10%"]]
    }
    adf_df = pd.DataFrame(adf_table)
    st.table(adf_df)

    adf_result_text = (
        "Result: Stationary (p-value <= 0.05)" if result_adf[1] <= 0.05 
        else "Result: Non-Stationary (p-value > 0.05)"
    )
    st.markdown(f"<p style='color:green;'>{adf_result_text}</p>", unsafe_allow_html=True)

    # **6. KPSS Test**
    st.markdown(f"### KPSS Test Results - {transformation}")
    st.markdown("**Hypotheses:**")
    st.markdown("- **H0**: 데이터는 Stationary Data이다.")
    st.markdown("- **H1**: 데이터는 Non-Stationary Data이다.")

    result_kpss = kpss(selected_data, regression='c')
    kpss_table = {
        "Statistic": [result_kpss[0]],
        "p-value": [result_kpss[1]],
        "Critical Value (10%)": [result_kpss[3]["10%"]],
        "Critical Value (5%)": [result_kpss[3]["5%"]],
        "Critical Value (2.5%)": [result_kpss[3]["2.5%"]],
        "Critical Value (1%)": [result_kpss[3]["1%"]]
    }
    kpss_df = pd.DataFrame(kpss_table)
    st.table(kpss_df)

    kpss_result_text = (
        "Result: Stationary (p-value > 0.05)" if result_kpss[1] > 0.05 
        else "Result: Non-Stationary (p-value <= 0.05)"
    )
    st.markdown(f"<p style='color:green;'>{kpss_result_text}</p>", unsafe_allow_html=True)


def main():
    # <1> 주식 코드 입력
    stock = st.sidebar.text_input("주식 코드를 입력하세요", value='^GSPC')
    if not stock:
        st.sidebar.warning("주식 코드를 입력하세요.")
        return

    # 주식 이름 표시
    show_stock_name(stock)
    st.divider()

    # <2> Raw Data 불러오기 
    st.markdown("<h3 style='font-size:30px; color:blue;'>Raw Data</h3>", unsafe_allow_html=True)
    data = load_data(stock)

    # 날짜 범위 설정
    min_date = data['Date'].min().date()
    max_date = data['Date'].max().date()

    # **세션 상태 초기화**
    if "start_date" not in st.session_state:
        st.session_state.start_date = min_date
    if "end_date" not in st.session_state:
        st.session_state.end_date = max_date
    if "date_range" not in st.session_state:
        st.session_state.date_range = (min_date, max_date)

    # 슬라이더와 달력 동기화 함수
    def update_slider():
        st.session_state.date_range = (st.session_state.start_date, st.session_state.end_date)

    def update_calendar():
        st.session_state.start_date, st.session_state.end_date = st.session_state.date_range

    # 슬라이더 추가
    st.sidebar.write("### 날짜 선택 옵션")
    st.sidebar.slider(
        'Select a date range (Slider)',
        min_value=min_date,
        max_value=max_date,
        value=st.session_state.date_range,
        format="YYYY-MM-DD",
        on_change=update_calendar,
        key="date_range"
    )

    # 달력 추가 (가로 배치)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            min_value=min_date,
            max_value=max_date,
            on_change=update_slider,
            key="start_date"
        )
    with col2:
        st.date_input(
            "End Date",
            value=st.session_state.end_date,
            min_value=min_date,
            max_value=max_date,
            on_change=update_slider,
            key="end_date"
        )

    # 날짜 범위 유효성 검사
    if st.session_state.start_date > st.session_state.end_date:
        st.sidebar.error("Start date cannot be after end date.")
        return

    # 데이터 필터링
    filtered_data = data[
        (data['Date'] >= pd.to_datetime(st.session_state.start_date)) &
        (data['Date'] <= pd.to_datetime(st.session_state.end_date))
    ]

    if filtered_data.empty:
        st.warning("선택한 날짜 범위에 데이터가 없습니다. 다른 범위를 선택하세요.")
        return

    # 필터링된 데이터 표시
    st.dataframe(filtered_data, use_container_width=True)

    # 로그 데이터 생성
    filtered_data = log_scale(filtered_data)  
 
    # <3> 원본 데이터 히스토그램 및 검정 결과
    st.markdown("<h3 style='font-size:30px; color:blue;'>Data Histogram</h3>", unsafe_allow_html=True)
    original_fig, original_text_boxes, log_fig, log_text_boxes = show_combined_histogram_with_tests(filtered_data)
    st.plotly_chart(original_fig, use_container_width=True)
    st.plotly_chart(original_text_boxes, use_container_width=True)
    st.plotly_chart(log_fig, use_container_width=True)
    st.plotly_chart(log_text_boxes, use_container_width=True)

    # <4> 시계열 분해
    st.markdown("<h3 style='font-size:30px; color:blue;'>Time-series decomposition and visualization</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    # 컬럼 선택
    with col1:
        col = st.selectbox('Column', filtered_data.columns[1:6], index=0)

    # 로그 스케일 옵션 
    with col2: 
        log = st.selectbox('Log Scale', ['log', 'original'], index=0)

    # 가법/승법 모델 선택
    with col3:
        model = st.selectbox('Decomposition Model', ['additive', 'multiplicative'], index=0)

    # 시계열 분해 및 시각화
    fig, signal, trend, seasonal, residual = show_decomposition(filtered_data, col, log=log, model=model)
    st.plotly_chart(fig, use_container_width=True)
    
    # <5> 정상성 검정
    st.markdown("<h3 style='font-size:30px; color:blue;'>Stationary Test</h3>", unsafe_allow_html=True)
    stationary_test(filtered_data, col)

    # <6> Smoothing Visualization 
    st.markdown("<h3 style='font-size:30px; color:blue;'>Smoothing Methods</h3>", unsafe_allow_html=True)
    visualize_smoothing(filtered_data)

if __name__ == "__main__": 
    main() 

    
    
 