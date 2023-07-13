# data analysis stack
import numpy as np
import pandas as pd

# data visualization stack
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# miscellaneous
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# deep learning stack
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
train = pd.read_csv("Augmentation/train_df.csv")
test = pd.read_csv("Augmentation/test_df.csv")

CLASSES = sorted(train['diagnostic'].unique())
ohe = OneHotEncoder()
y_train = ohe.fit_transform(
    train['diagnostic'].values.reshape(-1, 1)
)
y_train = np.array(y_train.todense())
 # resnet50
from tensorflow.keras.applications.resnet50 import (
     ResNet50,
     preprocess_input,
     decode_predictions
 )

# instantiate image data generator

data_gen = image.ImageDataGenerator(
    # preprocessing function for all images
    preprocessing_function=preprocess_input
)
train_generator = data_gen.flow_from_dataframe(
    dataframe=train,
    directory="./cancer/images/train",
    x_col="img_id",
    y_col="diagnostic",
    target_size=(224,224),
    batch_size=600,
    class_mode="categorical",
    shuffle=False
)
# load in all images at once

x_train = next(train_generator)[0]

BASE_MODEL = ResNet50(
     weights='imagenet', 
     include_top=False,  # removal of final dense layers
     pooling='avg',      # average pooling to last convolutional layer's ouput
     input_shape=(224,224,3) # ignored if input tensor is provided
 )
# freeze base model
BASE_MODEL.trainable = False
# base model summary
BASE_MODEL.summary()

def HyperModel(hyperparameters):
    '''
    complies a model by stacking dense layers on top of base model 
    '''
    # initialize the Sequential API to stack the layers
    model = keras.Sequential()
    
    # convolutional base 
    model.add(BASE_MODEL)
    
 # number of neurons in first dense layer
    model.add(keras.layers.Dense(units=256, activation="elu"))
    model.add(keras.layers.Dropout(rate=0.5))
 # number of neurons in first dense layer
    hp_units_2 = hyperparameters.Choice('units', values=[256, 128, 64, 32, 16])
 # number of neurons in first dense layer
    hp_units_3 = hyperparameters.Choice('units', values=[128, 64, 32, 16, 8, 6])
 # number of neurons in first dense layer
    hp_units_4 = hyperparameters.Choice('units', values=[128, 64, 32, 16, 8, 6])

    # first dense layer
    model.add(keras.layers.Dense(units=hp_units_2, activation='elu'))
    # dropout 
    model.add(keras.layers.Dropout(rate=0.5))
    
    # first dense layer
    model.add(keras.layers.Dense(units=hp_units_3, activation='elu'))
    # dropout 
    model.add(keras.layers.Dropout(rate=0.5))
    
    # first dense layer
    model.add(keras.layers.Dense(units=hp_units_4, activation='elu'))
    # dropout 
    model.add(keras.layers.Dropout(rate=0.5))

    # output layer with softmax activation function
    model.add(keras.layers.Dense(len(CLASSES),activation='softmax'))

    # learning rate for the optimizer
    #hp_learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.categorical_crossentropy,metrics=[keras.metrics.categorical_accuracy])

    return model

import keras_tuner as kt
#grid search
# instantiate hyperband

tuner = kt.GridSearch(
    hypermodel=HyperModel,
    objective='val_categorical_accuracy'
)
# hypertuning settings summary
tuner.search_space_summary() 
# early stopping

stop_early = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10
)
# Hypertuning

tuner.search(
    x_train, 
    y_train,
    epochs=25,
    validation_split=0.2,
    callbacks=[stop_early]
)

# best hyperparameters
best_hp = tuner.get_best_hyperparameters(1)[0]
best_hp.get('units')
h_model = tuner.hypermodel.build(best_hp)
h_model.summary()