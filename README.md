# Suicide_Risk_Analyzer
A deep learning model for detecting suicidal tendencies in text using LSTM and GloVe embeddings. Includes interactive text input with IPyWidgets for real-time analysis.

# Suicide Risk Detection Model
This project builds a machine learning model using an LSTM (Long Short-Term Memory) neural network for detecting suicidal tendencies in text data. The model is trained on a labeled dataset of text samples and uses pre-trained GloVe embeddings for word representations.

# Table of Contents
Overview
Dataset
Model Architecture
Installation
Usage
Preprocessing
Training
Prediction
Model and Tokenizer Saving
IPyWidgets for Interaction
Results
Contributing
License

# Overview
The goal of this project is to detect suicide-related tendencies in textual data using a deep learning model. The model is trained on a dataset that contains user-generated text labeled as either suicidal or non-suicidal.

The architecture utilizes GloVe embeddings for word representations and an LSTM layer for sequential text processing.

# Dataset
The dataset (Suicide_Detection.csv) consists of user-generated text labeled as either suicide or non-suicide. For this project, text cleaning and preprocessing are essential steps.

# Features:
Text: User-generated posts or comments.
Class: Labels indicating whether the text has suicidal tendencies (suicide) or not (non-suicide).
Important: You need to download or prepare a dataset and save it as Suicide_Detection.csv before running the code.

# Model Architecture
The model is a sequential deep learning model with the following layers:

Embedding Layer: Uses pre-trained GloVe embeddings (6B, 300-dimensional vectors) to transform words into dense vectors.
LSTM Layer: An LSTM with 64 units to capture temporal dependencies in the text.
GlobalMaxPooling1D Layer: To capture the most important features across the sequence.
Dense Layer: Fully connected layers for classification.
Dropout Layers: To prevent overfitting.
Output Layer: A single neuron with a sigmoid activation function for binary classification (suicidal or non-suicidal).

# Dependencies:
keras
tensorflow
numpy
pandas
matplotlib
sklearn
pickle

# License
This project is licensed under the MIT License - see the LICENSE file for details.
