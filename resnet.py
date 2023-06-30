import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
# from PIL import Image
# import io
# import lime
# from lime import lime_image
from io import BytesIO
from model_methods import (
    load_img,
    show_original_img,
    image_preprocessing,
    predict_image,
    print_prediction
    )
CLASSES = ['Cataract', 'Diabetic retinopathy', 'Glaucoma', 'Normal']
st.title('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;👁️ Eye Disease Classifier')
image = Image.open('/Users/karimi/Desktop/datasets/eye-disease/main_dataset/app.png')
st.image(image,width = 600)

# Title and information about the app's functionality and limitations

# st.info('Only classifies **Cataract**, **Diabetic retinopathy**, **Glaucoma** or **Normal**.\n\nModel is restricted to giving **1** class at a time')

# Define the sidebar layout
sidebar = st.sidebar
logo_image = Image.open('/Users/karimi/Desktop/datasets/eye-disease/main_dataset/spiced2.png')
st.sidebar.image(logo_image, use_column_width=True)
new_img = sidebar.file_uploader('Upload photo') # File uploader widget in the sidebar
show_image_button = st.sidebar.button('Show Image') # Button to show and classify the image
classify_button = sidebar.button('Classify')


# Page configuration
# st.set_page_config(
#     layout='wide',
#     page_icon='👁️',
#     page_title='Eye Disease Classifier',
#     initial_sidebar_state='auto'
# )

# # Title and information about the app's functionality and limitations
# st.title('👁️ Eye Disease Classifier')
# st.info('Only classifies **Cataract**, **Diabetic retinopathy**, **Glaucoma** or **Normal**. \n\n Model is restricted to giving **1** class at a time')

# # Define the sidebar layout
# sidebar = st.sidebar
# show_image_button = st.sidebar.button('Show Image')
# classify_button = st.sidebar.button('Classify')


# # File uploader widget 
# new_img = st.file_uploader('Upload photo')



# Define default values for prediction variables
pred_prob = ""
pred_class = ""

if show_image_button and new_img is not None:
    col1, col2 = st.columns([1, 2])
    with col1:
        show_original_img(new_img)
    with col2:
        result = predict_image(new_img)  # Result is a probabilities array
        max_result = np.max(result) * 100  # Max probability
        pred_prob = np.format_float_positional(max_result, precision=2)  # Format probability
        pred_class = CLASSES[np.argmax(result)]  # Predicted class
        print_prediction(pred_prob, pred_class)

if classify_button and new_img is not None:
    col1, col2 = st.columns([1, 2])
    with col1:
        show_original_img(new_img)
    with col2:
        result = predict_image(new_img)  # Result is a probabilities array
        max_result = np.max(result) * 100  # Max probability
        pred_prob = np.format_float_positional(max_result, precision=2)  # Format probability
        pred_class = CLASSES[np.argmax(result)]  # Predicted class
        print_prediction(pred_prob, pred_class)
       








# show_image_button
# show_image_button = st.button(label = "show image")
# # classify button
# classify_button = st.button(label = "Classify")

# if new_img is not None and show_image_button:
#     show_original_img(new_img)
#     result = predict_image(new_img)  # Result is a probabilities array
#     max_result = np.max(result) * 100  # Max probability
#     pred_prob = np.format_float_positional(max_result, precision=2)  # Format probability
#     pred_class = CLASSES[np.argmax(result)]  # Predicted class


# # Calls the predict_upload function and displays the results in the sidebar 
# # if the "Classify" button is clicked and there is a new image uploaded
# if classify_button:
#     with st.sidebar:
#         print_prediction(pred_prob, pred_class)















# Explanation section
# if st.button('Explain') and new_img is not None:
#     img_array = np.array(Image.open(io.BytesIO(new_img.read())))
#     explain_prediction(img_array)
# if st.button('Explain') and new_img is not None:
#     with st.sidebar:
#         explain_prediction(new_img)




# if new_img is not None:
#     predictions, model = predict_image(new_img)  # Output tensor and model
#     plot_gradient_maps(predictions, model)  # Pass the predictions and model to the function
#     st.caption('Saliency map')


# with col2:
#     if new_img is not None:
#         input_im = predict_image(new_img)  # Output tensor
#         # Add your code here to plot gradient maps using input_im
#         st.caption('Saliency map')

# with col3:
#     if new_img is not None:
#         # Add your code here to generate Grad-CAM heatmap using new_img
#         st.caption('Activation heatmap')
# if new_img is None:
#         with st.sidebar: 
#              st.warning('''
#              Please upload retinal image for classification. 
#              \n\n Thank you 🙏
#              ''')








