
# Suicide Risk Detection Model

This project builds a machine learning model using an LSTM (Long Short-Term Memory) neural network for detecting suicidal tendencies in text data. The model is trained on a labeled dataset of text samples and uses pre-trained GloVe embeddings for word representations.
## Table of Contents
Table of Contents
- Overview
- Dataset
- Model Architecture
- Installation & Dependencies
- IPyWidgets for Interaction
- Comparison of Different Models


## Overview
The goal of this project is to detect suicide-related tendencies in textual data using a deep learning model. The model is trained on a dataset that contains user-generated text labeled as either suicidal or non-suicidal.

The architecture utilizes GloVe embeddings for word representations and an LSTM layer for sequential text processing.
## Dataset
The dataset [Suicide Detection Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data)(Suicide_Detection.csv) consists of user-generated text labeled as either suicide or non-suicide. For this project, text cleaning and preprocessing are essential steps.

or You can dowload the dataset from this drive link including the Glove Embedding(6B,300-Dimensional Vectors):

-Drive Link: [Suicide Detection Dataset](https://drive.google.com/drive/folders/1sXBMovYW2kjUq3HMUbDKv6cd5cIdwQaA?usp=sharing)

## Features
- Text: User-generated posts or comments.
- Class: Labels indicating whether the text has suicidal tendencies (suicide) or not (non-suicide).
- Important: You need to download or prepare a dataset and save it as Suicide_Detection.csv before running the code.
## Model Architecture

The model is a sequential deep learning model with the following layers:

- **Embedding Layer**: Uses pre-trained **GloVe embeddings (6B, 300-dimensional vectors)** to transform words into dense vectors.
- **LSTM Layer**: An LSTM with 64 units to capture temporal dependencies in the text.
- **GlobalMaxPooling1D Layer**: To capture the most important features across the sequence.
- **Dense Layer**: Fully connected layers for classification.
- **Dropout Layers**: To prevent overfitting.
- **Output Layer**: A single neuron with a sigmoid activation function for binary classification (suicidal or non-suicidal).
## Installation & Dependencies
Clone the repository:

```bash
git clone https://github.com/TajKab/Suicide_Risk_Analyzer.git
cd suicide-risk-detection
```
Install the necessary dependencies:
```bash

pip install -r requirements.txt
```
**Dependencies**:
- keras
- tensorflow
- numpy
- pandas
- matplotlib
- sklearn
- pickle



## Interactive Output
***IPyWidgets for Interaction***:

The code includes IPyWidgets for interactive text analysis. The widget allows users to input a sentence and receive an immediate analysis indicating whether the sentence suggests suicidal tendencies.

[![Op1.jpg](https://i.postimg.cc/G2G0nVm2/Op1.jpg)](https://postimg.cc/7GHtgB78)
## Comparison of Different Models 
Comparison of Models
In this project, we explored both deep learning and traditional machine learning models for suicide detection using text data.

- **LSTM-based NLP Model**:
Utilizes word embeddings (GloVe) and an LSTM layer for text classification.
Achieved high accuracy and recall for detecting suicidal tendencies.
- **Traditional Machine Learning Models**:
TF-IDF Vectorization: Text data was also transformed using TF-IDF, a common technique for converting text into numerical features.
Three traditional machine learning models were used: **Logistic Regression, Decision Tree, and Random Forest***.
The following table presents the performance metrics (Accuracy, F1 Score, Precision, and Recall) for all models:

[![Op2.jpg](https://i.postimg.cc/YSJcVMkZ/Op2.jpg)](https://postimg.cc/Ny7P2hKk)

The differences in outputs for traditional models stem primarily from how they handle the input features (TF-IDF vs. padded sequences)


