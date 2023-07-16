# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Load data
train = pd.read_csv("Augmentation/train_df.csv")
train = train.sample(frac=1).reset_index()
train = train.drop(["index", "level_0"], axis=1)

# Preprocess data
CLASSES = sorted(train['diagnostic'].unique())
ohe = OneHotEncoder()
y_train = ohe.fit_transform(train['diagnostic'].values.reshape(-1, 1)).toarray()

# Data augmentation
data_gen = image.ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)
train_generator = data_gen.flow_from_dataframe(
    dataframe=train,
    directory="./cancer/images/train",
    x_col="img_id",
    y_col="diagnostic",
    target_size=(224, 224),
    batch_size=32,  # Adjust batch size based on your system's capabilities
    class_mode="categorical",
    shuffle=True  # Shuffle the training data
)

# Load ResNet50 base model
BASE_MODEL = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

# Freeze base model
BASE_MODEL.trainable = False

# Function to build the model
def build_model(units_2, units_3, units_4):
    model = keras.Sequential()
    model.add(BASE_MODEL)
    model.add(keras.layers.Dense(units=256, activation="elu"))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=units_2, activation='elu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=units_3, activation='elu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=units_4, activation='elu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(len(CLASSES), activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])
    return model

# Hyperparameter tuning using Keras Tuner
import keras_tuner as kt

def hypermodel(hyperparameters):
    units_2 = hyperparameters.Choice('units_2', values=[256, 128, 64, 32, 16])
    units_3 = hyperparameters.Choice('units_3', values=[128, 64, 32, 16, 8, 6])
    units_4 = hyperparameters.Choice('units_4', values=[128, 64, 32, 16, 8, 6])

    return build_model(units_2, units_3, units_4)

tuner = kt.RandomSearch(
    hypermodel=hypermodel,
    objective='val_loss',
    max_trials=10,  # Adjust the number of trials based on your computational resources
    directory='tuner_logs',
    project_name='cancer_classification'
)

# Define callbacks
stop_early = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10,
    restore_best_weights=True
)

# Perform hyperparameter search
tuner.search(
    train_generator,
    epochs=25,
    validation_split=0.2,
    callbacks=[stop_early]
)

# Get the best hyperparameters and build the final model
best_hp = tuner.get_best_hyperparameters(1)[0]
best_units_2 = best_hp.get('units_2')
best_units_3 = best_hp.get('units_3')
best_units_4 = best_hp.get('units_4')

best_model = build_model(best_units_2, best_units_3, best_units_4)
best_model.summary()
