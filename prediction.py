import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.models import load_model
import os
import pandas as pd

def load_image(image_path, target_size=(224, 224)):
    """
    Loads an input image into PIL format of specified size
    """
    img = image.load_img(path=image_path, target_size=target_size)
    return img

def preprocess_image(loaded_image, model_name):
    """
    Preprocesses a loaded image based on the specified model
    """
    img_array = image.img_to_array(loaded_image)
    img_batch = np.expand_dims(img_array, axis=0)
    
    if model_name == "resnet":
        processed_img = preprocess_input_resnet50(img_batch)
    elif model_name == "densenet":
        processed_img = preprocess_input_densenet(img_batch)
    elif model_name == "mobilenet":
        processed_img = preprocess_input_mobilenet(img_batch)
    else:
        raise ValueError("Invalid model name")
    
    return processed_img

def show_preprocess_image(loaded_image, model_names):
    """
    Shows loaded image and preprocesses it for the specified models
    """
    # Display image
    plt.figure(figsize=(3, 3), dpi=100)
    plt.imshow(loaded_image)
    
    # Preprocess image
    processed_images = []
    for model_name in model_names:
        processed_img = preprocess_image(loaded_image, model_name)
        processed_images.append(processed_img)
    
    return processed_images

def image_predict(preprocessed_images, non_image_features, model):
    """
    Returns class probabilities for given preprocessed images and non-image features,
    based on the loaded model
    """
    predictions = model.predict([preprocessed_images, non_image_features])
    probabilities = np.round(predictions[0], 6)
    class_probabilities = dict(zip(CLASSES, probabilities))
    
    return class_probabilities

# Load the model
model = load_model("0.7463_owerfitted(a-bit).h5")

# Load the class labels
CLASSES = os.listdir('./images')

# Prompt user to enter the image file name
test_image_name = input("Enter the name of the test image file: ")

# Construct the full image path
test_image_path = os.path.join('./zr7vgbcyr2-1/cancer/phot', test_image_name)

# Load and preprocess the user-selected image
loaded_image = load_image(test_image_path)
model_names = ["resnet", "densenet", "mobilenet"]
processed_images = show_preprocess_image(loaded_image, model_names)

# Load non-image features
test_row = pd.read_csv("./DataWork/transformed_df_test_.csv")
non_image_features = test_row[test_row["img_id"] == test_image_name]
non_image_features = np.array(non_image_features.drop(["diagnostic", "img_id", "image_path"], axis=1))

# Replicate non_image_features to match the batch size
num_samples = processed_images[0].shape[0]
non_image_features = np.tile(non_image_features, (num_samples, 1))

# Make predictions on the test image
predictions = image_predict(processed_images, non_image_features, model)
print(predictions)


def main(test_image_name: str):
    # Load the model
    model = load_model("0.7463_owerfitted(a-bit).h5")

    # Load the class labels
    CLASSES = os.listdir('./images')

    # Construct the full image path
    test_image_path = os.path.join('./zr7vgbcyr2-1/cancer/phot', test_image_name)

    # Load and preprocess the user-selected image
    loaded_image = load_image(test_image_path)
    model_names = ["resnet", "densenet", "mobilenet"]
    processed_images = show_preprocess_image(loaded_image, model_names)

    # Load non-image features
    test_row = pd.read_csv("./Augmentation/test_df_names.csv")
    non_image_features = test_row[test_row["patient"] == test_image_name]
    non_image_features = np.array(non_image_features.drop(["diagnostic", "img_id", "image_path"], axis=1))

    # Replicate non_image_features to match the batch size
    num_samples = processed_images[0].shape[0]
    non_image_features = np.tile(non_image_features, (num_samples, 1))

    # Make predictions on the test image
    predictions = image_predict(processed_images, non_image_features, model)
    print(predictions)

if __name__ == "__main__":
    main()