import numpy as np

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image


import streamlit as st
from PIL import Image
# import io
# import lime
# from lime import lime_image

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# upload model
MODEL = tf.saved_model.load(
    '/Users/karimi/Desktop/datasets/eye-disease/main_dataset/finalefmodel4'
    )

def load_img(file):
    """
    """
    img = Image.open(file)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    return img
    
def show_original_img(file):
    """
    """
    try:
        img = load_img(file)
        st.image(img)
    except Image.UnidentifiedImageError as error:
        st.error(f"Invalid image file: {error}")

def image_preprocessing(file):
    """
    """
    img = load_img(file)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)
    input_tensor = preprocess_input(input_tensor)
    return input_tensor

def predict_image(file):
    """
    """
    try:
        input_tensor = image_preprocessing(file)
        # input_spec = tf.TensorSpec(
        #     shape=input_tensor.shape, dtype=input_tensor.dtype
        #     )
        prediction = MODEL.signatures["serving_default"](
            tf.constant(input_tensor, dtype=input_tensor.dtype)
            )
        output_tensor_name = list(prediction.keys())[0]
        predictions = prediction[output_tensor_name].numpy()
        return predictions
    except Image.UnidentifiedImageError as error:
        st.error(f"Invalid image file: {error}")

# Function to handle the prediction and display of results
def print_prediction(pred_prob, pred_class):
    """
    """
    st.write(f'### There is a')
    st.success(f'# {pred_prob}% probability')
    st.write(f'### that this retinal image shows')
    st.success(f'# {pred_class}')

# def predict_image(image):
#     model = tf.saved_model.load('/Users/karimi/Desktop/datasets/eye-disease/main_dataset/finalefmodel4')

#     try:
#         img = Image.open(image)
#         img = img.convert('RGB')
#         img = img.resize((224, 224))
#         img_array = tf.keras.preprocessing.image.img_to_array(img)
#         img_array = img_array / 255.0
#         input_tensor = np.expand_dims(img_array, axis=0)
#         input_tensor = tf.keras.applications.efficientnet.preprocess_input(input_tensor)
#         input_spec = tf.TensorSpec(shape=input_tensor.shape, dtype=input_tensor.dtype)
#         prediction = model.signatures["serving_default"](tf.constant(input_tensor, dtype=input_tensor.dtype))
#         output_tensor_name = list(prediction.keys())[0]
#         predictions = prediction[output_tensor_name].numpy()
#         return predictions
#     except Image.UnidentifiedImageError as error:
#         st.error("Invalid image file: {}".format(error))


# def predict_image(image):

#     # Load the Keras model
#     # model = tf.keras.models.load_model('/Users/karimi/Desktop/datasets/eye-disease/main_dataset/finalefmodel')
#     model = tf.saved_model.load('/Users/karimi/Desktop/datasets/eye-disease/main_dataset/finalefmodel4')
#     # Read the image and decode to a tensor
#     img = Image.open(io.BytesIO(image.read()))
#     img = img.convert('RGB')
#     # Resize the image to the desired size
#     img = img.resize((224, 224))
#     # Convert the image to a NumPy array
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     # Normalize the pixel values to the range of [0, 1]
#     img_array = img_array / 255.0
#     # Expand the dimensions to create a batch of size 1
#     input_tensor = np.expand_dims(img_array, axis=0)
#     # Preprocess the input tensor
#     input_tensor = tf.keras.applications.efficientnet.preprocess_input(input_tensor)
#     # Perform prediction
#     # Convert the input tensor to a TensorSpec
#     # Convert the input tensor to a TensorSpec
#     input_spec = tf.TensorSpec(shape=input_tensor.shape, dtype=input_tensor.dtype)

#     # Run the prediction
#     prediction = model.signatures["serving_default"](tf.constant(input_tensor, dtype=input_tensor.dtype))

#     # Get the output tensor name dynamically
#     output_tensor_name = list(prediction.keys())[0]

#     # Access the predicted output
#     predictions = prediction[output_tensor_name].numpy()

#     return predictions




    # predictions = model.predict(input_tensor)
    # return predictions

# Function to display the original image
# def display_original_image(image):
#     try:
#         img = Image.open(io.BytesIO(image.read()))
#         img = img.convert('RGB')
#         img = img.resize((224, 224))
#         st.image(img)
#     except PIL.UnidentifiedImageError as error:
#         st.error("Invalid image file: {}".format(error))


# def explain_prediction(image_path):
#     predictions, model = predict_image(image_path)
#     explainer = lime.lime_image.LimeImageExplainer(random_state=12)

#     def predict_fn(images):
#         return model.signatures["serving_default"](tf.constant(images, dtype=tf.float32))

#     explanation = explainer.explain_instance(
#         image_path,
#         predict_fn,
#         top_labels=1,
#         hide_color=0,
#         num_samples=1000
#     )

#     temp_1, mask_1 = explanation.get_image_and_mask(
#         explanation.top_labels[0],
#         positive_only=True,
#         num_features=5,
#         hide_rest=True
#     )
#     temp_2, mask_2 = explanation.get_image_and_mask(
#         explanation.top_labels[0],
#         positive_only=False,
#         num_features=10,
#         hide_rest=False
#     )

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
#     ax1.imshow(mark_boundaries(temp_1, mask_1))
#     ax2.imshow(mark_boundaries(temp_2, mask_2))
#     ax1.axis('off')
#     ax2.axis('off')
#     plt.show()


# def original_img(image):
#     try:
#         img = Image.open(io.BytesIO(image.read()))
#         img = img.convert('RGB')
#         img = img.resize((224, 224))
#         st.image(img)
#     except Image.UnidentifiedImageError as error:
#         st.error("Invalid image file: {}".format(error))



# # Function to normalize the image
# def normalize_image(img):
#     grads_norm = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
#     grads_norm = (grads_norm - tf.reduce_min(grads_norm)) / (tf.reduce_max(grads_norm) - tf.reduce_min(grads_norm))
#     return grads_norm

# def normalize_image(img):
#     img_norm = img / 255.0
#     return img_norm

# def generate_confusion_matrix(true_labels, predicted_labels, class_names):
#     cm = confusion_matrix(true_labels, predicted_labels)

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Confusion Matrix')
#     plt.show()




# def f1_score(true_labels, predicted_labels):
#     y_true = K.variable(true_labels)
#     y_pred = K.variable(predicted_labels)
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     recall = true_positives / (possible_positives + K.epsilon())
#     f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
#     return K.eval(f1_val)



# def plot_gradient_maps(input_im, model):
#     def normalize_image(image):
#         img_norm = image / 255.0
#         return img_norm

#     def plot_maps(gradient_map, input_image):
#         fig, ax = plt.subplots(figsize=(8, 8))
#         ax.imshow(gradient_map, cmap='hot')
#         ax.imshow(input_image, alpha=0.5)
#         plt.axis('off')
#         plt.show()

#     input_im = tf.convert_to_tensor(input_im)  # Convert to TensorFlow tensor

#     with tf.GradientTape() as tape:
#         tape.watch(input_im)
#         result_img = model(input_im)
#         max_idx = tf.argmax(result_img, axis=1)
#         max_score = tf.math.reduce_max(result_img[0, max_idx[0]])

#     grads = tape.gradient(max_score, input_im)
#     plot_maps(normalize_image(grads[0]), normalize_image(input_im[0]))



# def plot_gradient_maps(input_im, model):
#     with tf.GradientTape() as tape:
#         tape.watch(input_im)   
#         result_img = model(input_im)
#         max_idx = tf.argmax(result_img, axis=1)
#         max_score = tf.math.reduce_max(result_img[0, max_idx[0]])  # tensor max probability
#     grads = tape.gradient(max_score, input_im)
#     plot_maps(normalize_image(grads[0]), normalize_image(input_im[0]))

# def plot_maps(img1, img2, vmin=0.3, vmax=0.7, mix_val=2):
#     fig, ax = plt.subplots(figsize=(3.3, 3.3))
#     ax.imshow(img1 * mix_val + img2 / mix_val, cmap="terrain")
#     plt.axis("off")
#     fig.savefig("temp_fig.png", transparent=True, frameon=False, bbox_inches='tight', pad_inches=0)
#     image = Image.open('temp_fig.png')
#     st.image(image)
    # st.pyplot(fig)

# def gradCAM(orig, model):
#     img = Image.open(io.BytesIO(orig.getvalue()))
#     img = img.convert('RGB')
#     img = img.resize((224, 224))
#     x = tf.keras.preprocessing.image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = tf.keras.applications.efficientnet.preprocess_input(x)

#     last_conv_layer = model.get_layer('top_conv')
#     iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
#     model_out, last_conv_layer = iterate(x)
#     class_out = model_out[:, np.argmax(model_out[0])]
#     grads = tape.gradient(class_out, last_conv_layer)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
#     heatmap = heatmap.reshape((5, 5))

#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

#     img = heatmap * intensity + img
#     img = cv2.resize(img, (res, res))

#     cv2.imwrite('temporary.jpg', img)
#     st.image('temporary.jpg')