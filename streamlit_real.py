# import std libraries
import numpy as np 
import pandas as pd
import requests

from IPython.display import HTML
import pickle
import json

from prediction import main
import streamlit as st
#from st_aggrid import AgGrid

TEST_DATA = pd.read_csv("./Augmentation/test_df.csv")
TEST_DATA.rename(
    index=lambda x: x+1,
    inplace=True
    )
NAMES = ["~~~"] + list(TEST_DATA['patient'].sort_values()) 


# sidebar
with st.sidebar:
    # title
    st.title("Skin Cancer Detection")
    # image
    st.image('./Stimlit_Images/logo.png')
    # blank space
    st.write("")
    # selectbox
    page = st.selectbox(
        "Steps",
        [
            "Home",
            "How sensitive?",
            "Data collection",
            "Methodology of ml model (Concatinating)",
            "Lets see some results",
            "Further steps",
            "Ty and tools"
            ]
        ) 



if page == "Home":
    st.markdown("<h3 style='text-align: center; color: white;'>Berkay Bozkurt</h3>", unsafe_allow_html=True)
    st.write("*I regret taking so good care of my skin. --- said no one ever*")
    # blank space
    st.write("")
    # image
    st.image('./Stimlit_Images/skin-cancer-header.png')


elif page == "How sensitive?":
    # title
    st.title("How Sensitive Skin Can Be?")
    col1,col2,col3,col4 = st.columns([10,1,5,5])
    with col1:
        n = st.slider(
        label="How sensitive you are?",
        min_value=1,
        max_value=3,
        value=1
        ) 

    with col4:
        st.markdown("###")
        show_button = st.button(label="Give me the truth")


    st.markdown("###")
    if show_button:
        if n == 1:
            
            st.title("You are fine")
            st.image('./Stimlit_Images/first_level_burn.png', 
