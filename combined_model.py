import pandas as pd
import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, concatenate, Input, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder


def load_data(csv_path: str, image_dir: str):
    """
    Load data from CSV and images from a directory.

    """
    # Load CSV data
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1).reset_index()
    df = df.drop(["index", "level_0"], axis=1)
    X = np.array(df.drop(["diagnostic", "img_id"], axis=1))
    y = np.array(df["diagnostic"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y)

    # Load images using ImageDataGenerator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    batch_size = 600
    target_size = (224, 224)
    class_mode = 'categorical'

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col="img_id",
        y_col="diagnostic",
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )

    xtrain, ytrain = next(train_generator)
    return xtrain, ytrain, X, y

def build_resnet_model(input_shape: tuple):
    """
    Build a ResNet model based on ResNet50 architecture.

    """
    BASE_MODEL = ResNet50(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=input_shape
    )
    BASE_MODEL.trainable = True

    for layer in BASE_MODEL.layers[:-4]:  # Freeze all layers except the last 4
        layer.trainable = False

    model = Sequential()
    model.add(BASE_MODEL)
    model.add(Dense(units=2048, activation="elu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=256, activation="elu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=128, activation="elu"))
    model.add(Dropout(rate=0.5))
    return model

def build_densenet_model(input_shape: tuple):
    """
    Build a DenseNet model based on DenseNet121 architecture.

    """
    BASE_MODEL_DENSENET = DenseNet121(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=input_shape
    )
    BASE_MODEL_DENSENET.trainable = True

    for layer in BASE_MODEL_DENSENET.layers[:-4]:  # Freeze all layers except the last 4
        layer.trainable = False

    model = Sequential()
    model.add(BASE_MODEL_DENSENET)
    model.add(Dense(units=512, activation="elu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=256, activation="elu"))
    model.add(Dropout(rate=0.5))
    return model

def build_mobilenet_model(input_shape: tuple):
    """
    Build a MobileNet model based on MobileNet architecture.

    """
    BASE_MODEL_MOBILENET = MobileNet(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=input_shape
    )
    BASE_MODEL_MOBILENET.trainable = True

    for layer in BASE_MODEL_MOBILENET.layers[:-4]:  # Freeze all layers except the last 4
        layer.trainable = False

    model = Sequential()
    model.add(BASE_MODEL_MOBILENET)
    model.add(Dense(units=512, activation="elu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=256, activation="elu"))
    model.add(Dropout(rate=0.5))
    return model

def build_ann_model(input_shape: np.array):
    """
    Build a CNN model for CSV file.

    """
    model = Sequential()
    input_ann = Input(shape=input_shape)
    ann_output = Dense(units=16, activation='elu')(input_ann)
    ann_output = Dropout(rate=0.5)(ann_output)
    model = Model(inputs=input_ann, outputs=ann_output)
    return model

def build_combined_model(model_1, model_2, model_3, model_4):
    """
    Build a combined model by concatenating ResNet, DenseNet, MobileNet, and CNN models.

    """
    concatenated = concatenate([model_1.output, model_2.output, model_3.output, model_4.output])
    combined_output = Dense(units=32, activation='elu')(concatenated)
    combined_output = Dense(units=6, activation='softmax')(combined_output)

    combined_model = Model(
        inputs=[model_1.input, model_2.input, model_3.input, model_4.input],
        outputs=[combined_output]
    )

    return combined_model

def train_model(model, xtrain, ytrain, X, y, batch_size, epochs):
    """
    Train the model.

    """
    opt = tf.keras.optimizers.SGD(lr=0.0003)
    model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80)
    
    history = model.fit([xtrain, xtrain, xtrain, X], ytrain, batch_size=batch_size, epochs=epochs,
                        callbacks=[stop_early], validation_split=0.2)
    
    return history

# Clear Keras session
K.clear_session()

# Load and preprocess the data
csv_path = "Augmentation/train_df.csv"
image_dir = "cancer/images/train"
xtrain, ytrain, X, y = load_data(csv_path, image_dir)

# Build individual models
cnn_model = build_resnet_model(input_shape=(224, 224, 3))
densenet_model = build_densenet_model(input_shape=(224, 224, 3))
mobilenet_model = build_mobilenet_model(input_shape=(224, 224, 3))
ann_model = build_ann_model(input_shape=X.shape[1])

# Build combined model
combined_model = build_combined_model(cnn_model, densenet_model, mobilenet_model, ann_model)
combined_model.summary()

# Train the model
history = train_model(combined_model, xtrain, ytrain, X, y, batch_size=16, epochs=1_000)

# Save the model
combined_model.save("model.h5")

# Save the model as pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(combined_model, f)
