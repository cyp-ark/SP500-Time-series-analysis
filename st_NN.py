import streamlit as st
import EDA_NN
import stock_NN

def sidebar_nagivation():
    st.sidebar.title('Options')
    st.session_state["session"]="EDA_NN"
    if st.sidebar.button('Stock'):
        st.session_state["session"]="Stock_NN"
        
def run():
    st.set_page_config(layout='wide') 
    
    # 사이드바 네비게이션
    sidebar_nagivation()
    
    if st.session_state.get("session","Stock") == "Stock_NN":
        stock_NN.main()
    if st.session_state.get("session","Stock") == "EDA_NN":
        st.title("EDA")
        # st.write("EDA is not implemented yet.")
        EDA_NN.main()
    
if __name__ == "__main__":
    run()
