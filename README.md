# Skin_Disease_Detection

This repository contains a machine learning algorithm with deep convolutional networks designed to predict six classes of skin cancer using the PAD-UFES-20 dataset, which includes raw images and patient information stored in a CSV file. The algorithm utilizes pre-trained ResNet152, DenseNet121 and MobileNet models as features such as visual semantics, edges, and object shapes can be transferable from the Imagenet dataset. Additionally, a dedicated CNN model is designed for processing the CSV file, capturing relevant information and improving the overall prediction accuracy. All code is implemented in Python.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [PAD-UFES-20](#pad-ufes-20)


## Overview

Detecting skin cancer in skin lesions has traditionally been a costly and time-consuming task. However, Computer-Aided Decision Support Systems can assist in timely and cost-effective diagnosis. This project aims to leverage multi-modal data, consisting of both images and meta-features of the patients, to aid in the detection of skin cancer.

## Dataset

The project utilizes the PAD-UFES-20 dataset, which has been tested and used for training the algorithm. The dataset contains images of different modalities, including smartphone-captured and clinical images. Additionally, the dataset includes various meta-features associated with the patients.

## PAD-UFES-20

The PAD-UFES-20 dataset can be accessed through the following link: [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

The dataset comprises 2,298 clinical skin lesion images captured using smartphone devices. It covers six lesion categories, namely Basal Cell Carcinoma, Melanoma, Squamous Cell Carcinoma, Actinic Keratosis, Nevus, and Seborrheic Keratosis. Furthermore, the dataset provides 21 patient clinical details, such as gender, age, and cancer history, among others.

## Technologies and Libraries
- Python 3.9
- Scikit-learn
- Keras
- TensorFlow
- CUDA
- pandas
- numpy

