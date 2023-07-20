import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.models import load_model
import os
import pandas as pd

CLASSES = {
    "ACK": "Actinic Keratosis",
    "MEL": "Malignant Melanoma",
    "BCC": "Basal Cell Carcinoma of skin",
    "SCC": "Squamous Cell Carcinoma",
    "SEK": "Seborrheic Keratosis",
    "NEV": "Nevus"
}

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


def main(patient_name: str):
    # Load the model
    model = load_model("model.h5")

    # Load non-image features
    test_row = pd.read_csv("./Augmentation/test_df.csv")
    test_row=test_row.sample(frac=1).reset_index()
    non_image_features = test_row[test_row["patient"] == patient_name]
    non_image_features = np.array(non_image_features.drop(["diagnostic", "img_id", "patient", "index", "level_0"], axis=1))

    # Construct the full image path
    img_id = test_row[test_row["patient"] == patient_name]
    img_id = img_id["img_id"].to_string(index=False).strip()
    test_image_path = os.path.join('./cancer/images/test', img_id)

    # Load and preprocess the user-selected image
    loaded_image = load_image(test_image_path)
    model_names = ["resnet", "densenet", "mobilenet"]
    processed_images = show_preprocess_image(loaded_image, model_names)


    # Replicate non_image_features to match the batch size
    num_samples = processed_images[0].shape[0]
    non_image_features = np.tile(non_image_features, (num_samples, 1))

    # Make predictions on the test image
    predictions = image_predict(processed_images, non_image_features, model)
    
    # Find the highest-scoring prediction
    max_prob = max(predictions.values())
    predicted_class = [k for k, v in predictions.items() if v == max_prob][0]

    # Get the full name of the predicted class
    predicted_class_name = CLASSES.get(predicted_class)

    # Display the result
    result_text = f"{patient_name} most likely has '{predicted_class_name}' "
    return result_text, loaded_image


if __name__ == "__main__":
    main("SEK_Samuel")