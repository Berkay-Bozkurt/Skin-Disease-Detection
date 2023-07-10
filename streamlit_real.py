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
            st.image('./Stimlit_Images/first_level_burn.png', width=500)
        
        elif n == 2:
            st.title("Go to doctor, man",)
            col1,col2 = st.columns([10,5])
            with col1:
                st.image('./Stimlit_Images/second_level_burn_1.png')
            with col2:
                st.image('./Stimlit_Images/second_level_burn_2.png')

        elif n == 3:
            st.title("Call the ambulance!!!",)
            st.image('./Stimlit_Images/third_level_burn.png')
            
            
elif page == "Data collection":
    data_set = st.selectbox("Select Data Base",
                            ["ISAC", "PAD"])

    #col1,col2 = st.columns([3,3])
    if data_set == "ISAC":
        st.write("Huge dataset based on images")
        st.image('./Stimlit_Images/ISAC_data.png', width=600)
        
    if data_set == "PAD":
        st.write("Limited images BUT has detailed patient information")
        st.image('./Stimlit_Images/word-cloud.png', width=700)
        expand = st.button(label="Expand")
        if expand:
            st.markdown("<h6 style='text-align: white;'>Having a dataset with both images and patient information is a blessing and curse at some time.</h6>", unsafe_allow_html=True)
            #st.write("")
            st.write("")
            st.markdown("<h6 style='text-align: white;'>Patient informatin data was a mass.</h6>", unsafe_allow_html=True)
            
            st.write("")
            st.write("")
            st.markdown("<h3 style='text-align: center; color: white;'>Have a taste of it</h3>", unsafe_allow_html=True)
            st.image('./Stimlit_Images/raw_dataset.png')
            st.write("")
            st.markdown("<h6 style='text-align: white;'>But still after some future engineering. It gives such a great understanding of diseases.</h6>", unsafe_allow_html=True)
            st.write("")


        expand_ = st.button(label="Expand More")
        if expand_:
            st.markdown("<h6 style='text-align: white;'>But still after some future engineering. It gives such a great understanding of diseases.</h6>", unsafe_allow_html=True)
            col1,col2,col3,col4,col5 = st.columns([10,1,10,1,10])
            
            with col1:
                st.write("age_distribution")
                st.image('./DataWork/images/age_distribution.png')

            with col3:
                st.write("regions_per_diag")
                st.image('./DataWork/images/regions_per_diag.png')
                
            with col5:
                st.write("gender_per_diag")
                st.image('./DataWork/images/gender_per_diag.png')
            col1,col2,col3,col4,col5 = st.columns([10,1,10,1,8])

            with col1:
                st.write("region_frequency")
                st.image('./DataWork/images/region_frequency.png')
            with col3:
                st.write("feature_importances")
                st.image('./DataWork/images/feature_importances_using_permutation_on_full_model.png')
                
            with col5:
                st.write("cramer_v_corr")
                st.image('./DataWork/images/cramer_v_corr.png')




elif page == "Methodology of ml model (Concatinating)":
    st.markdown("###")
    col1, col2, col3 = st.columns(3)
    with col2:
      st.image('./Stimlit_Images/Emoji.png', width=100)

    st.markdown("<h6 style='text-align: white;'>If you have limited amount of data and it is not good enough to feed your model what would you do?</h6>", unsafe_allow_html=True)
    st.write("")
    show_button = st.button(label="Model")
    if show_button:
        st.markdown("<h6 style='text-align: white;'>An example of a CNN model</h6>", unsafe_allow_html=True)
        st.write("")
        st.image('./Stimlit_Images/CNN.png',width=400)

    show_button_ = st.button(label="Answer lies here")
    if show_button_:
        st.title("Concatenation")
        st.markdown("<h6 style='text-align: white;'>Savior of The Data People</h6>", unsafe_allow_html=True)
        st.write("")
        st.image('./Stimlit_Images/concat.png')
        # col1,col2,col3,col4,col5 = st.columns([10,1,10,1,10])
        
        # with col1:
        #     st.write("csv_ann_omly")
        #     st.image('./Stimlit_Images/csv_ann_omly.png')

        # with col3:
        #     st.write("ann_only")
        #     st.image('./Stimlit_Images/ann_only.png')
            
        # with col5:
        #     st.write("gender_per_diag")
        #     st.image('./Stimlit_Images/gender_per_diag.png')






