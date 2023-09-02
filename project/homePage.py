import streamlit as st 

st.set_page_config(page_title="Home", page_icon="")

st.title(" Welcome to my Website Feel Free To Browse")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 