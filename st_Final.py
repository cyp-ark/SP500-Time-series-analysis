import streamlit as st
import EDA_Final
import Stock_Final
import Reg_Final
 
def sidebar_navigation():
    st.sidebar.title('Options')
  
    # 사이드바 버튼을 통해 세션 상태 설정
    if "session" not in st.session_state:
        st.session_state["session"] = "EDA_Final"  # 초기 세션 상태 설정

    # 항상 버튼이 표시되도록 구현
    if st.sidebar.button('EDA'):
        st.session_state["session"] = "EDA_Final"
    if st.sidebar.button('Stock'):
        st.session_state["session"] = "Stock_Final"
    if st.sidebar.button('Regression'):
        st.session_state["session"] = "Regression_Final"

def run():
    # 페이지 기본 설정
    st.set_page_config(layout='wide', page_title='Navigation App')

    # 사이드바 네비게이션
    sidebar_navigation() 

    # 세션 상태에 따라 각 모듈의 main 함수 실행
    if st.session_state["session"] == "Stock_Final":
        Stock_Final.main()
    elif st.session_state["session"] == "EDA_Final":
        # st.title("EDA")
        EDA_Final.main()
    elif st.session_state["session"] == "Regression_Final":
        # st.title("Regression")
        Reg_Final.main()

if __name__ == "__main__":
    run()
