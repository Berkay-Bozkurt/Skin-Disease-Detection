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
    st.image('./Streamlit_Images/logo.png')
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
    st.image('./Streamlit_Images/skin-cancer-header.png')


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
            st.image('./Streamlit_Images/first_level_burn.png', width=500)
        
        elif n == 2:
            st.title("Go to doctor, man",)
            col1,col2 = st.columns([10,5])
            with col1:
                st.image('./Streamlit_Images/second_level_burn_1.png')
            with col2:
                st.image('./Streamlit_Images/second_level_burn_2.png')

        elif n == 3:
            st.title("Call the ambulance!!!",)
            st.image('./Streamlit_Images/third_level_burn.png')
            
            
elif page == "Data collection":
    data_set = st.selectbox("Select Data Base",
                            ["ISAC", "PAD"])

    #col1,col2 = st.columns([3,3])
    if data_set == "ISAC":
        st.write("Huge dataset based on images")
        st.image('./Streamlit_Images/ISAC_data.png', width=600)
        
    if data_set == "PAD":
        st.write("Limited images BUT has detailed patient information")
        st.image('./Streamlit_Images/word-cloud.png', width=700)
        expand = st.button(label="Expand")
        if expand:
            st.markdown("<h6 style='text-align: white;'>Having a dataset with both images and patient information is a blessing and curse at some time.</h6>", unsafe_allow_html=True)
            #st.write("")
            st.write("")
            st.markdown("<h6 style='text-align: white;'>Patient informatin data was a mass.</h6>", unsafe_allow_html=True)
            
            st.write("")
            st.write("")
            st.markdown("<h3 style='text-align: center; color: white;'>Have a taste of it</h3>", unsafe_allow_html=True)
            st.image('./Streamlit_Images/raw_dataset.png')
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
      st.image('./Streamlit_Images/Emoji.png', width=100)

    st.markdown("<h6 style='text-align: white;'>If you have limited amount of data and it is not good enough to feed your model what would you do?</h6>", unsafe_allow_html=True)
    st.write("")
    show_button = st.button(label="Model")
    if show_button:
        st.markdown("<h6 style='text-align: white;'>An example of a CNN model</h6>", unsafe_allow_html=True)
        st.write("")
        st.image('./Streamlit_Images/CNN.png',width=400)

    show_button_ = st.button(label="Answer lies here")
    if show_button_:
        st.title("Concatenation")
        st.markdown("<h6 style='text-align: white;'>Savior of The Data People</h6>", unsafe_allow_html=True)
        st.write("")
        st.image('./Streamlit_Images/concat.png')
        # col1,col2,col3,col4,col5 = st.columns([10,1,10,1,10])
        
        # with col1:
        #     st.write("csv_ann_omly")
        #     st.image('./Streamlit_Images/csv_ann_omly.png')

        # with col3:
        #     st.write("ann_only")
        #     st.image('./Streamlit_Images/ann_only.png')
            
        # with col5:
        #     st.write("gender_per_diag")
        #     st.image('./Streamlit_Images/gender_per_diag.png')





elif page == "Lets see some results":
    patient_name = st.selectbox("Pick a patient", NAMES)
    show_button = st.button(label="Make Prediction")
    if show_button:
        st.write("Don't worry I am getting the patient information and image for you")
        st.write("But be patient")
        prediction_result, prediction_image = main(patient_name)
        st.write(prediction_result)
        st.image(prediction_image, width=400)



elif page == "Further steps":
    col1,col2,col3,col4,col5 = st.columns([10,1,10,1,10])
            
    with col1:
        st.write("Collect more data")
        st.image('./Streamlit_Images/Data_collection.png')

    with col3:
      st.write("Another model for skin disease")
      st.image('./Streamlit_Images/skin.jpg')
        
    with col5:
       st.write("Mobile app for people to use it")
       st.image('./Streamlit_Images/mobile_app.png')
    




else:
    st.markdown("<h1 style='text-align: center; color: red;'>Tools I have used.</h1>", unsafe_allow_html=True)
    st.markdown("")
    col1,col2,col3,col4,col5 = st.columns([10,1,10,1,10])
    
    with col1:
        st.image('./Streamlit_Images/logo/Keras.png')

    with col3:
        st.image('./Streamlit_Images/logo/NumPy.png')
        
    with col5:
        st.image('./Streamlit_Images/logo/Pandas_logo.png')
    col1,col2,col3,col4,col5 = st.columns([10,1,10,1,8])

    with col1:
        st.image('./Streamlit_Images/logo/python-logo.png',width=150)

    with col3:
        st.image('./Streamlit_Images/logo/cuda.jpg')
        
    with col5:
        st.image('./Streamlit_Images/logo/sk_learn.jpg')

    col1,col2,col3,col4,col5 = st.columns([10,1,10,1,10])
    
    with col1:
        st.image('./Streamlit_Images/logo/sphx_glr_logo.png')

    with col3:
        st.image('./Streamlit_Images/logo/Streamlit-logo.jpeg')
        
    with col5:
        st.image('./Streamlit_Images/logo/tensorflow_.png')

    show_button = st.button(label="Thank You")
    if show_button:
        st.write("Thank you so much for your time")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("<h5 style='text-align: center; color: white;'>And always take care of your skin!!!</h5>", unsafe_allow_html=True)
        st.image('./Streamlit_Images/sunscreen.png')


